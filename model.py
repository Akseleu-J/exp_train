import math
import os
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List, Tuple
from jax.sharding import PartitionSpec as P

# ==========================================================================
# ФИКС (интеграция fused-Pallas backward B1-B6, atomic_ops/INTEGRATION_NOTES.md
# п. "Что стоит сделать" #3): раньше model.py импортировал ТОЛЬКО
# kernel_trainable.gdn2_pallas_forward_trainable ("читерский" backward --
# jax.vjp на чистом JAX-референсе). kernel_trainable_B6.py (честный
# fused-Pallas forward+backward B1->B2->B3->B4->B5, с финальной
# санитизацией градиентов на границе custom_vjp) провалидирован сравнением
# градиентов против kernel_trainable.py (atomic_ops/gdn2_backward_compare.py,
# compare_suite() по нескольким seed/размерам -- все finite, rel_diff < 5%)
# и подтверждён на реальном TPU. Переключаем основной импорт на него --
# это теперь единственный путь, которым обучается GDN-2 (и forward, и
# backward идут через кернелы Pallas A/B/C/D + B1-B5, никакого jax.vjp на
# JAX-референсе в горячем пути обучения больше нет).
#
# kernel_trainable.py (jax.vjp-на-референсе) остаётся в репозитории как
# cross-check / fallback -- см. его собственный docstring -- на случай, если
# понадобится снова сравнить градиенты при подозрении на регрессию.
# ==========================================================================
from atomic_ops.kernel_trainable_B6 import gdn2_pallas_forward_trainable
from atomic_ops.kernel_a_scores import BT as GDN2_PALLAS_BT
from atomic_ops.moe_gmm import GmmMoEJ

# ==========================================================================
# ФИКС (M6 -- интеграция Mamba2 SSD Pallas-пайплайна, atomic_ops/
# MAMBA2_PALLAS_PLAN.md): раньше Mamba2J вызывал напрямую
# mamba2_ssd_reference (чистый JAX, M1) -- временная "cheat forward"
# заглушка на время, пока сам Pallas-пайплайн (M2 cumdecay -> M3
# intra-chunk -> M4 inter-chunk scan) ещё разрабатывался и валидировался
# независимо (все три milestone'а подтверждены: rel_err на реальном TPU
# <1e-6 против mamba2_ssd_reference на multi-chunk/multi-batch кейсах).
# M5 обернул этот Pallas-пайплайн в custom_vjp (forward идёт через
# Pallas-кернели M2/M3/M4, backward -- через jax.vjp на mamba2_ssd_reference,
# тот же "честный forward + cheat backward первым" порядок, что уже был
# пройден для GDN-2 в kernel_trainable.py до появления B6) -- градиенты
# провалидированы против jax.vjp-на-референсе (rel_err ~1e-8 на fp32,
# все finite и корректного dtype на bfloat16). Переключаем импорт на этот
# обёрнутый Pallas-путь -- это теперь единственный путь, которым обучается
# Mamba2 (раньше был associative_scan, потом временно mamba2_ssd_reference
# напрямую; полностью честный fused-Pallas backward, B6-эквивалент для
# Mamba2 -- отдельный будущий milestone, не блокирует переход на это).
# ==========================================================================

from atomic_ops.kernel_mamba2_trainable_B6 import mamba2_pallas_forward_trainable_B6 as mamba2_pallas_forward_trainable

print("[MODEL] ⚙️ GDN-2: используется честный fused-Pallas forward+backward "
      "(kernel_d_pipeline.py + kernel_bwd_b1..b5, склеены в "
      "kernel_trainable_B6.py). jax.vjp-на-референсе backward "
      "(kernel_trainable.py) больше не используется в горячем пути "
      "обучения.")
print("[MODEL] ⚙️ Mamba2: используется Pallas forward (M2 cumdecay -> M3 "
      "intra-chunk -> M4 inter-chunk scan, kernel_mamba2_c_interchunk.py) + "
      "jax.vjp-на-референсе backward (kernel_mamba2_trainable.py, M5 -- "
      "тот же 'честный forward / cheat backward первым' порядок, что был "
      "пройден для GDN-2 до kernel_trainable_B6.py). Прямой вызов "
      "mamba2_ssd_reference из Mamba2J.__call__ больше не используется в "
      "горячем пути обучения (сам файл остаётся как cross-check/reference "
      "-- backward всё ещё дифференцирует через него).")

# ==========================================================================
# ФИКС (этот пасс -- диагностика сведена ТОЛЬКО к оптимизатору): все
# kernel/activation-level self.sow(...) вызовы (gdn2_input_*, gdn2_raw_out_*,
# gdn2_h_final_maxabs, gdn2_out_maxabs, gdn2_kernelstage_* -- через
# atomic_ops/kernel_diag.py, mamba2_input_*, mamba2_ssm_out_*, mla_input_*,
# mla_out_*, layer_delta_*, layer_resid_*, final_hidden_*) УДАЛЕНЫ из
# forward-пути. Причина: они (а) требовали ЛИШНЕГО пересчёта Kernel A/B/C
# под stop_gradient на каждый GDN-2 слой каждый шаг (kernel_diag.py's
# gdn2_kernel_stage_diagnostics, дорого и не нужно для стабильности самого
# обучения), (б) дублировали информацию, которую optimizer/train_setup уже
# дают на уровне градиентов и весов ПО ФИЗИЧЕСКОМУ СЛОЮ (см.
# train_setup.py's layer_grad_norm/layer_grad_maxabs/layer_grad_nonfinite +
# layer_w_norm/layer_w_maxabs/layer_w_nonfinite, diagnostics.py's
# build_leaf_stats_fn) -- то есть достаточно для диагностики "какая группа/
# слой нестабильны", без необходимости смотреть внутрь самого Pallas-ядра.
# Санитизация (clip/nan_to_num) НИГДЕ не тронута -- убрана только отчётность.
#
# GDN2_FWD_DIAG-диагностика ниже по файлу (_diag_maxabs/_diag_resid_breakdown,
# jax.debug.print под lax.cond) остаётся КАК ЕСТЬ, но выключена по
# умолчанию (env var) -- используйте её только для точечной ручной отладки
# одного прогона, не для постоянно включённого мониторинга.
# ==========================================================================

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as pallas_flash_attention,
        BlockSizes as FlashBlockSizes,
    )
    _PALLAS_FLASH_ATTENTION_IMPORT_ERROR = None
except Exception as _e:
    pallas_flash_attention = None
    FlashBlockSizes = None
    _PALLAS_FLASH_ATTENTION_IMPORT_ERROR = _e

_model_mesh = None
_batch_axis = None

def set_model_mesh(mesh, batch_axis=None):
    global _model_mesh, _batch_axis
    _model_mesh = mesh
    _batch_axis = batch_axis

def get_model_mesh():
    return _model_mesh

def get_batch_axis():
    return _batch_axis


# ==========================================================================
# ДИАГНОСТИКА (сквозная, переиспользует тот же флаг, что kernel_d_pipeline.py
# использует для GDN-2 -- GDN2_FWD_DIAG=1, по умолчанию OFF, нулевой оверхед
# в обычном обучении). Manual/point-in-time debugging only -- see module
# docstring above about why the always-on kernel/activation sow()s were
# removed in favor of the optimizer-side per-layer diagnostics.
# ==========================================================================
_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"


def _is_traced(x):
    return isinstance(x, jax.core.Tracer)


def _diag_maxabs(tag: str, x):
    """No-op (возвращает x без изменений) если GDN2_FWD_DIAG=0."""
    if not _GDN2_FWD_DIAG:
        return x

    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    if _is_traced(x):
        jax.debug.print(
            "[MAMBA2-FWD-DIAG] " + tag + ": all_finite={f}  max_abs={m:.3e}",
            f=all_finite, m=max_abs,
        )
    else:
        print(f"[MAMBA2-FWD-DIAG] {tag}: all_finite={bool(all_finite)}  "
              f"max_abs={float(max_abs):.3e}")
    return x


def _diag_resid_breakdown(tag: str, current_x_before, delta):
    if not _GDN2_FWD_DIAG:
        return

    cx_before_abs = jnp.max(jnp.abs(jnp.nan_to_num(current_x_before, nan=0.0, posinf=0.0, neginf=0.0)))
    delta_abs = jnp.max(jnp.abs(jnp.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)))

    if _is_traced(current_x_before) or _is_traced(delta):
        def _report():
            jax.debug.print(
                "[RESID-BREAKDOWN-DIAG] " + tag +
                ": current_x_before={cx:.3e}  delta={d:.3e}  "
                "(если current_x_before уже большой -- накопление ИЗ ПРЕДЫДУЩИХ слоёв; "
                "если delta большая, а current_x_before маленький -- источник ЭТОТ слой)",
                cx=cx_before_abs, d=delta_abs,
            )

        jax.lax.cond(
            jnp.maximum(cx_before_abs, delta_abs) > 5e2,
            _report,
            lambda: None,
        )
    else:
        cx_v = float(cx_before_abs)
        d_v = float(delta_abs)
        if max(cx_v, d_v) > 5e2:
            print(f"[RESID-BREAKDOWN-DIAG] {tag}: current_x_before={cx_v:.3e}  delta={d_v:.3e}  "
                  f"(если current_x_before уже большой -- накопление ИЗ ПРЕДЫДУЩИХ слоёв; "
                  f"если delta большая, а current_x_before маленький -- источник ЭТОТ слой)")


def make_grad_probe(tag: str):
    @jax.custom_vjp
    def _probe(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print("[BWD-DIAG] ⚠️ non-finite ВХОДЯЩИЙ градиент в узле: " + tag),
            lambda: None,
        )
        return (g,)

    _probe.defvjp(_fwd, _bwd)
    return _probe


def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
    """Как make_grad_probe, но не только печатает, а АКТИВНО чинит non-finite
    градиент (nan_to_num + клип) прежде чем отдать его дальше по backward
    графу."""
    @jax.custom_vjp
    def _sanitizer(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        finite = jnp.all(jnp.isfinite(g))
        jax.lax.cond(
            jnp.logical_not(finite),
            lambda: jax.debug.print("[BWD-FIX] 🩹 non-finite градиент в узле {t} -- санитизирован", t=tag),
            lambda: None,
        )
        g_safe = jnp.nan_to_num(jnp.clip(g, -clip_val, clip_val), nan=0.0, posinf=clip_val, neginf=-clip_val)
        return (g_safe,)

    _sanitizer.defvjp(_fwd, _bwd)
    return _sanitizer


@struct.dataclass
class ModelConfig:
    d_model: int = 768
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 8
    n_heads_ssm: int = 8          # NEW (M0): per-head decay grouping for Mamba2's A_log.
                                    # d_inner = d_model*expand must be divisible by this.
    d_latent: int = 384
    d_ff: int = 3072
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 21          # 7 blocks × 3 layers
    layers_per_block: int = 3
    vocab_size: int = 151936
    dropout_rate: float = 0.1
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0001
    moe_capacity_factor: float = 1.25
    tie_embeddings: bool = True
    label_smoothing: float = 0.05
    router_noise_std: float = 0.3
    use_flash_attention: bool = True
    deltanet_chunk_size: int = 256

    # Layer type schedule: "gdn2", "mamba2", "mla"
    # 21 layers: 16 gdn2, 2 mamba2, 3 mla
    layer_types: Tuple[str, ...] = (
        "gdn2", "gdn2", "mla",      # block 0
        "gdn2", "mamba2", "gdn2",   # block 1
        "gdn2", "gdn2", "gdn2",     # block 2
        "gdn2", "gdn2", "mla",      # block 3
        "gdn2", "mamba2", "gdn2",   # block 4
        "gdn2", "gdn2", "gdn2",     # block 5
        "gdn2", "gdn2", "mla",      # block 6
    )


# ==========================================
# RoPE
# ==========================================
class RoPEEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, seq_len):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2)[: (self.dim // 2)] / self.dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    cos = cos.astype(x.dtype)
    sin = sin.astype(x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated_x = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated_x * sin


# ==========================================
# Multi-head Latent Attention (MLA)
# ==========================================
class MLAJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        b, l, _ = x.shape
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_model // n_heads

        Q = nn.Dense(self.cfg.d_model, use_bias=False, name="W_q", dtype=jnp.bfloat16)(x)
        Q = Q.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        Q_rope = apply_rope(Q, cos[None, None, :, :d_head], sin[None, None, :, :d_head])

        kv_latent = nn.Dense(self.cfg.d_latent, use_bias=False, name="W_kv_down", dtype=jnp.bfloat16)(x)
        K = nn.Dense(self.cfg.d_model, use_bias=False, name="W_k_up", dtype=jnp.bfloat16)(kv_latent)
        V = nn.Dense(self.cfg.d_model, use_bias=False, name="W_v_up", dtype=jnp.bfloat16)(kv_latent)

        K = K.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        K_rope = apply_rope(K, cos[None, None, :, :d_head], sin[None, None, :, :d_head])
        V = V.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)

        def _sanitize(t):
            return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        Q_rope, K_rope, V = map(_sanitize, (Q_rope, K_rope, V))

        sm_scale = 1.0 / math.sqrt(d_head)

        if self.cfg.use_flash_attention:
            if pallas_flash_attention is None:
                raise ImportError(
                    "cfg.use_flash_attention=True but jax.experimental.pallas.ops.tpu."
                    "flash_attention failed to import -- original error: "
                    f"{_PALLAS_FLASH_ATTENTION_IMPORT_ERROR!r}."
                )

            def _flash_call(q_local, k_local, v_local):
                local_b = q_local.shape[0]
                block_sizes = FlashBlockSizes(
                    block_q=1024,
                    block_k_major=1024,
                    block_k=1024,
                    block_b=local_b,
                    block_q_major_dkv=1024,
                    block_k_major_dkv=1024,
                    block_k_dkv=256,
                    block_q_dkv=1024,
                    block_k_major_dq=512,
                    block_k_dq=256,
                    block_q_dq=1024,
                )
                return pallas_flash_attention(
                    q_local, k_local, v_local,
                    causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
                )
            mesh = get_model_mesh()
            batch_axis = get_batch_axis()

            if mesh is not None:
                spec = P(batch_axis, None, None, None)
                sharded_flash = jax.shard_map(
                    _flash_call,
                    mesh=mesh,
                    in_specs=spec,
                    out_specs=spec,
                    check_vma=False,
                )
                out = sharded_flash(Q_rope, K_rope, V).astype(x.dtype)
            else:
                out = _flash_call(Q_rope, K_rope, V).astype(x.dtype)
        else:
            scores = jnp.einsum("bhqd,bhkd->bhqk", Q_rope, K_rope) * sm_scale
            scores = jnp.where(causal_mask == 0, -1e9, scores)
            attn = jax.nn.softmax(scores, axis=-1)
            if not deterministic:
                dropout_rng = rngs['dropout'] if rngs is not None and 'dropout' in rngs else self.make_rng('dropout')
                keep_prob = 1.0 - self.cfg.dropout_rate
                mask_drop = jax.random.bernoulli(dropout_rng, keep_prob, attn.shape)
                attn = attn * mask_drop / keep_prob
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, V)

        out = out.transpose(0, 2, 1, 3).reshape(b, l, self.cfg.d_model)

        out = nn.RMSNorm(epsilon=1e-6, name="attn_out_norm")(out).astype(x.dtype)

        out = make_grad_sanitizer("mla_flash_attn_out")(out)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="W_o", dtype=jnp.bfloat16)(out)


# ==========================================
# Mamba-2 (SSM)
# ==========================================
class Mamba2J(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        d_inner = d * self.cfg.expand
        d_state = self.cfg.d_state
        n_heads_ssm = self.cfg.n_heads_ssm
        assert d_inner % n_heads_ssm == 0, (
            f"d_inner={d_inner} must be divisible by cfg.n_heads_ssm={n_heads_ssm}."
        )
        headdim = d_inner // n_heads_ssm

        in_proj = nn.Dense(d_inner * 2, use_bias=False, name="in_proj", dtype=jnp.bfloat16)(x)
        x_bc, res = jnp.split(in_proj, 2, axis=-1)

        conv_w = self.param("conv_w", nn.initializers.normal(stddev=0.02), (d_inner, self.cfg.d_conv))
        conv_b = self.param("conv_b", nn.initializers.zeros, (d_inner,))
        rhs = conv_w.T[:, None, :].astype(x_bc.dtype)
        res_conv = jax.lax.conv_general_dilated(
            lhs=x_bc, rhs=rhs, window_strides=(1,),
            padding=[(self.cfg.d_conv - 1, 0)],
            feature_group_count=d_inner,
            dimension_numbers=('NHC', 'HIO', 'NHC'),
        )
        x_conv = jax.nn.silu(res_conv + conv_b[None, None, :].astype(x_bc.dtype))

        # M0: A_log now (n_heads_ssm,) instead of (d_inner,) -- per-head decay,
        # matches atomic_ops/mamba2_ssd_reference.py's assumption.
        A_log = self.param("A_log", nn.initializers.uniform(scale=1.0), (n_heads_ssm,))
        A_log_safe = jnp.clip(A_log, -20.0, 20.0).astype(jnp.float32)
        A = -jnp.exp(A_log_safe)  # (n_heads_ssm,), float32 -- ssd kernels want this dtype

        B = nn.Dense(d_state, use_bias=False, name="B_proj", dtype=jnp.bfloat16)(x_bc)
        C = nn.Dense(d_state, use_bias=False, name="C_proj", dtype=jnp.bfloat16)(x_bc)
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj", dtype=jnp.bfloat16)(x_bc))
        dt = jnp.clip(dt, jnp.array(1e-2, dtype=dt.dtype), jnp.array(1.0, dtype=dt.dtype))

        def _sanitize(t, bound=1e3):
            bd = jnp.array(bound, dtype=t.dtype)
            return jnp.nan_to_num(jnp.clip(t, -bd, bd), nan=0.0, posinf=bound, neginf=-bound)

        dt, B, C, x_conv = map(_sanitize, (dt, B, C, x_conv))

        chunk_size = min(self.cfg.deltanet_chunk_size, l)
        if l % chunk_size != 0:
            raise ValueError(f"seq_len={l} must be divisible by deltanet_chunk_size={chunk_size}.")

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()
        dt_h = dt.reshape(b, l, n_heads_ssm, headdim)
        x_h = x_conv.reshape(b, l, n_heads_ssm, headdim)
        B_f = B
        C_f = C

        state0_zeros = jnp.zeros((b, n_heads_ssm, headdim, d_state), dtype=jnp.float32)

        def _mamba2_fixed(dt_, x_, B_, C_, A_, state0_):
            return mamba2_pallas_forward_trainable(
                dt_, x_, B_, C_, A_, chunk_size=chunk_size, state0=state0_
            )
        if mesh is not None:
            in_spec_dx = P(batch_axis, None, None, None)
            in_spec_bc = P(batch_axis, None, None)
            in_spec_a = P(None)
            in_spec_state0 = P(batch_axis, None, None, None)
            out_spec = (
                P(batch_axis, None, None, None),
                P(batch_axis, None, None, None),
            )
            sharded_mamba2 = jax.shard_map(
                _mamba2_fixed,
                mesh=mesh,
                in_specs=(in_spec_dx, in_spec_dx, in_spec_bc, in_spec_bc, in_spec_a, in_spec_state0),
                out_specs=out_spec,
                check_vma=False,
            )
            y_h, _state_final = sharded_mamba2(dt_h, x_h, B_f, C_f, A, state0_zeros)
        else:
            y_h, _state_final = _mamba2_fixed(dt_h, x_h, B_f, C_f, A, state0_zeros)

        y = y_h.reshape(b, l, d_inner).astype(x_bc.dtype)
        y = _diag_maxabs("mamba2_ssm_out_pre_norm", y)
        y = nn.RMSNorm(epsilon=1e-6, name="ssm_out_norm")(y).astype(x_bc.dtype)
        y = _diag_maxabs("mamba2_ssm_out_post_norm", y)

        out = y * jax.nn.silu(res)
        out = jnp.clip(out, jnp.array(-3e2, dtype=out.dtype), jnp.array(3e2, dtype=out.dtype))
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out)



# ==========================================
# Gated DeltaNet-2 (GDN-2)
# ==========================================
def _gdn2_recurrence_impl(k, e, z, alpha, q, dtype):
    """Reference-only fallback, not part of the active forward path -- see
    original file for full explanation. Left unchanged."""
    b, l, n_heads, d_head = k.shape

    def _to_time_major(t):
        return jnp.moveaxis(t, 1, 0)

    k_t, e_t, z_t, alpha_t, q_t = map(_to_time_major, (k, e, z, alpha, q))

    S0 = jnp.zeros((b, n_heads, d_head, d_head), dtype=jnp.float32)

    def _step(S, inputs):
        k_i, e_i, z_i, alpha_i, q_i = inputs
        k_f = k_i.astype(jnp.float32)
        e_f = e_i.astype(jnp.float32)
        z_f = z_i.astype(jnp.float32)
        alpha_f = alpha_i.astype(jnp.float32)
        q_f = q_i.astype(jnp.float32)

        S_bar = alpha_f[..., :, None] * S
        r = jnp.einsum('bhkv,bhk->bhv', S_bar, e_f)
        S_new = S_bar + jnp.einsum('bhk,bhv->bhkv', k_f, z_f - r)
        o = jnp.einsum('bhkv,bhk->bhv', S_new, q_f)
        return S_new, o.astype(dtype)

    _step = jax.checkpoint(_step)
    _, out_t = jax.lax.scan(_step, S0, (k_t, e_t, z_t, alpha_t, q_t))

    return jnp.moveaxis(out_t, 0, 1)


class GatedDeltaNet2J(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        n_heads = self.cfg.n_heads
        d_head = d // n_heads
        eps = 1e-6

        def short_causal_conv(name, u):
            conv_w = self.param(f"{name}_conv_w", nn.initializers.normal(stddev=0.02), (d, self.cfg.d_conv))
            conv_b = self.param(f"{name}_conv_b", nn.initializers.zeros, (d,))
            rhs = conv_w.T[:, None, :].astype(u.dtype)
            out = jax.lax.conv_general_dilated(
                lhs=u,
                rhs=rhs,
                window_strides=(1,),
                padding=[(self.cfg.d_conv - 1, 0)],
                feature_group_count=d,
                dimension_numbers=('NHC', 'HIO', 'NHC')
            )
            return out + conv_b[None, None, :].astype(u.dtype)

        q_lin = nn.Dense(d, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(x)
        k_lin = nn.Dense(d, use_bias=False, name="k_proj", dtype=jnp.bfloat16)(x)
        v_lin = nn.Dense(d, use_bias=False, name="v_proj", dtype=jnp.bfloat16)(x)

        q = jax.nn.silu(short_causal_conv("q", q_lin)).reshape(b, l, n_heads, d_head)
        k = jax.nn.silu(short_causal_conv("k", k_lin)).reshape(b, l, n_heads, d_head)
        v = jax.nn.silu(short_causal_conv("v", v_lin)).reshape(b, l, n_heads, d_head)
        v = jnp.clip(v, -50.0, 50.0)

        def _safe_normalize(t):
            return t * jax.lax.rsqrt(jnp.sum(t * t, axis=-1, keepdims=True) + eps ** 2)

        q = make_grad_sanitizer("gdn2_q_normalize")(_safe_normalize(q))
        k = make_grad_sanitizer("gdn2_k_normalize")(_safe_normalize(k))

        b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="erase_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="write_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,)).astype(jnp.float32)
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj", dtype=jnp.bfloat16)(x).reshape(b, l, n_heads, d_head)
        a_param_safe = jnp.clip(a_param, -20.0, 20.0)
        g = -jnp.exp(a_param_safe)[None, None, :, None] * jax.nn.softplus(f_proj.astype(jnp.float32))
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)
        alpha = jnp.exp(g)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate", dtype=jnp.bfloat16)(x)
        out_gate = jnp.clip(out_gate, -1e2, 1e2)

        def _sanitize(t):
            return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        q, k, v, w_gate, b_gate, g = map(_sanitize, (q, k, v, w_gate, b_gate, g))

        if l % GDN2_PALLAS_BT != 0:
            raise ValueError(
                f"seq_len={l} must be divisible by the Pallas GDN-2 chunk size "
                f"({GDN2_PALLAS_BT}); this is currently a separate constant from "
                f"cfg.deltanet_chunk_size, see kernel_a_scores.py."
            )

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()

        _gdn2_fixed = partial(gdn2_pallas_forward_trainable, scale=1.0, h0=None)

        in_spec = P(batch_axis, None, None, None)

        if mesh is not None:
            out_spec = (
                P(batch_axis, None, None, None),
                P(batch_axis, None, None, None),
            )
            sharded_gdn2 = jax.shard_map(
                _gdn2_fixed,
                mesh=mesh,
                in_specs=(in_spec, in_spec, in_spec, in_spec, in_spec, in_spec),
                out_specs=out_spec,
                check_vma=False,
            )
            out, _h_final = sharded_gdn2(q, k, v, w_gate, b_gate, g)
        else:
            out, _h_final = _gdn2_fixed(q, k, v, w_gate, b_gate, g)

        out = out.reshape(b, l, d)

        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out).astype(x.dtype)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out * jax.nn.silu(out_gate))


# ==========================================
# MoE
# ==========================================
class ExpertPack(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        h = nn.Dense(self.cfg.d_ff, name="w1", dtype=jnp.bfloat16)(x)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.cfg.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(self.cfg.d_model, name="w2", dtype=jnp.bfloat16)(h)


class MoEJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        E = self.cfg.num_experts

        router_logits = nn.Dense(E, use_bias=False, name="router", dtype=jnp.bfloat16)(flat_x)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape, dtype=router_logits.dtype
            )
        gate = jax.nn.softmax(router_logits, axis=-1)

        mean_probs = jnp.mean(gate, axis=0)
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * mean_probs))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(
            jax.scipy.special.logsumexp(router_logits, axis=-1))))
        self.sow("losses", "expert_utilization", mean_probs)

        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=E,
        )(cfg=self.cfg, name="experts_block")
        all_outputs = run_experts(flat_x, deterministic)

        weighted = jnp.einsum("te,etd->td", gate, all_outputs)
        return weighted.reshape(b, l, d)


class IntraBlockAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, sources, block_input):
        n_sources = len(sources)
        if n_sources == 1:
            return sources[0]

        b, l, d = block_input.shape

        q = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(block_input)

        k_list = []
        for i, src in enumerate(sources):
            k_i = nn.Dense(self.cfg.d_latent, use_bias=False, name=f"k_proj_{i}", dtype=jnp.bfloat16)(src)
            k_list.append(k_i)

        k_stack = jnp.stack(k_list, axis=0)

        scores = jnp.einsum("bld,nbld->nbl", q, k_stack) / jnp.sqrt(jnp.array(self.cfg.d_latent, dtype=q.dtype))

        weights = jax.nn.softmax(scores, axis=0)

        src_stack = jnp.stack(sources, axis=0)

        mixed = jnp.einsum("nbl,nbld->bld", weights, src_stack)
        return mixed.astype(block_input.dtype)


class HybridDARAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, current_x, all_sources):
        n = len(all_sources)
        if n == 0:
            return jnp.zeros_like(current_x)
        if n == 1:
            # ФИКС: единственный источник -- softmax по оси размера 1
            # тождественно даёт вес 1.0 независимо от q_proj/k_proj,
            # т.е. градиент через них структурно нулевой (не численно
            # маленький, а РОВНО ноль) -- k_proj/q_proj в этом вызове
            # никогда ничему не учатся, только тратят FLOPs и портят
            # Muon-диагностику (orth_resid для нулевого градиента ==
            # sqrt(dim), константный артефакт, всегда "выигрывает" argmax).
            # Тот же shortcut, что IntraBlockAttention уже применяет для
            # n_sources==1.
            return all_sources[0].astype(current_x.dtype)

        b, l, d = current_x.shape
        stack = jnp.stack(all_sources, axis=0)
        
        if n == 0:
            return jnp.zeros_like(current_x)

        b, l, d = current_x.shape
        stack = jnp.stack(all_sources, axis=0)

        flat = stack.reshape(n * b * l, d)
        k_flat = nn.Dense(
            self.cfg.d_latent, use_bias=False, name="k_proj", dtype=jnp.bfloat16
        )(flat)
        k = k_flat.reshape(n, b, l, self.cfg.d_latent)

        q = nn.Dense(
            self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16
        )(current_x)

        scores = jnp.einsum(
            "bld,nbld->nbl", q, k
        ) / jnp.sqrt(jnp.array(self.cfg.d_latent, dtype=q.dtype))

        weights = jax.nn.softmax(scores, axis=0)
        retrieved = jnp.einsum("nbl,nbld->bld", weights, stack)
        return retrieved.astype(current_x.dtype)


class SpecializedSublayer(nn.Module):
    cfg: ModelConfig
    layer_type: str

    @nn.compact
    def __call__(self, x, causal_mask=None, cos=None, sin=None, deterministic=True, rngs=None):
        if self.layer_type == "gdn2":
            return GatedDeltaNet2J(cfg=self.cfg, name="gdn2")(x)
        elif self.layer_type == "mamba2":
            return Mamba2J(cfg=self.cfg, name="mamba2")(x)
        elif self.layer_type == "mla":
            return MLAJ(cfg=self.cfg, name="mla")(
                x, causal_mask, cos, sin, deterministic=deterministic, rngs=rngs
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")


class BlockDARLayer(nn.Module):
    cfg: ModelConfig
    layer_type: str
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, x_input, block_input, local_deltas, history_blocks,
                 causal_mask, cos, sin, deterministic=True, rngs=None):
        b, l, d = current_x.shape

        dar_sources = []
        if history_blocks.shape[0] > 0:
            dar_sources.extend([history_blocks[j] for j in range(history_blocks.shape[0])])
        dar_sources.extend(local_deltas)

        retrieved = HybridDARAttention(cfg=self.cfg, name="dar")(
            current_x, dar_sources
        )
        retrieved = make_grad_probe(f"block{self.layer_idx}_dar_out")(retrieved)
        current_x = current_x + retrieved

        intra_sources = [block_input] + list(local_deltas)
        mixed = IntraBlockAttention(cfg=self.cfg, name="intra")(
            intra_sources, block_input
        )
        mixed = make_grad_probe(f"block{self.layer_idx}_intra_out")(mixed)

        mixed = nn.RMSNorm(epsilon=1e-6, name="pre_sublayer_norm")(mixed)

        delta = SpecializedSublayer(
            cfg=self.cfg, layer_type=self.layer_type, name="sublayer"
        )(mixed, causal_mask=causal_mask, cos=cos, sin=sin,
          deterministic=deterministic, rngs=rngs)

        delta = make_grad_sanitizer(f"delta_fanin_layer{self.layer_idx}_{self.layer_type}")(delta)

        delta = make_grad_probe(f"sublayer_out_layer{self.layer_idx}_{self.layer_type}")(delta)

        delta = jnp.nan_to_num(jnp.clip(delta, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        return delta

class BlockDAR(nn.Module):
    cfg: ModelConfig
    block_idx: int
    layer_idx_start: int

    @nn.compact
    def __call__(self, current_x, x_input, history_blocks, causal_mask, cos, sin,
                 deterministic=True, rngs=None):
        b, l, d = current_x.shape
        block_input = current_x

        local_deltas = []

        for i in range(self.cfg.layers_per_block):
            layer_idx = self.layer_idx_start + i
            layer_type = self.cfg.layer_types[layer_idx]

            delta = BlockDARLayer(
                cfg=self.cfg,
                layer_type=layer_type,
                layer_idx=layer_idx,
                name=f"layer_{layer_idx}"
            )(current_x, x_input, block_input, local_deltas, history_blocks,
              causal_mask, cos, sin, deterministic, rngs)

            delta_finite = jnp.all(jnp.isfinite(delta))
            jax.lax.cond(
                jnp.logical_not(delta_finite),
                lambda: jax.debug.print(
                    "[FWD-DIAG] ⚠️ non-finite delta: block={b} layer={l} type=" + layer_type,
                    b=self.block_idx, l=layer_idx,
                ),
                lambda: None,
            )

            _diag_resid_breakdown(f"block={self.block_idx}_layer={layer_idx}_{layer_type}",
                                   current_x, delta)

            _cx_next = current_x + delta
            cx_abs_max = jnp.max(jnp.abs(jnp.nan_to_num(_cx_next, nan=0.0, posinf=0.0, neginf=0.0)))
            jax.lax.cond(
                cx_abs_max > 1e3,
                lambda: jax.debug.print(
                    "[RESID-DIAG] ⚠️ current_x после layer={l} (block={b}) max|abs|={m} "
                    "ДО санитизации -- дрейф residual stream", b=self.block_idx, l=layer_idx, m=cx_abs_max,
                ),
                lambda: None,
            )

            local_deltas.append(delta)

            _resid_clip = 5e2 if layer_type == "mamba2" else 1e3
            current_x = jnp.nan_to_num(
                jnp.clip(_cx_next, -_resid_clip, _resid_clip),
                nan=0.0, posinf=_resid_clip, neginf=-_resid_clip,
            )

        # перед new_history:
        block_delta = sum(local_deltas)
        block_delta = make_grad_probe(f"block_delta_block{self.block_idx}")(block_delta)
        new_history = jnp.concatenate(
            [history_blocks, block_delta[None, ...]], axis=0
        )

        # в BlockDAR.__call__, после вычисления moe_out:
        norm_2 = nn.RMSNorm(epsilon=1e-6, name="norm_2")(current_x).astype(current_x.dtype)
        moe_out = GmmMoEJ(cfg=self.cfg, top_k=self.cfg.top_k, name="moe")(
            norm_2, deterministic=deterministic, rngs=rngs
        )
        moe_out = make_grad_probe(f"moe_out_block{self.block_idx}")(moe_out)
        moe_finite = jnp.all(jnp.isfinite(moe_out))
        jax.lax.cond(
         jnp.logical_not(moe_finite),
            lambda: jax.debug.print("[FWD-DIAG] ⚠️ non-finite moe_out: block={b}", b=self.block_idx),
             lambda: None,
        )

        output = jnp.nan_to_num(jnp.clip(current_x + moe_out, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        return output, new_history


class FullHybridMoEModel(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None, return_hidden: bool = False):
        b, l = input_ids.shape
        embed_layer = nn.Embed(
            num_embeddings=self.cfg.vocab_size,
            features=self.cfg.d_model,
            name="embed",
            dtype=jnp.bfloat16,
        )
        x_input = embed_layer(input_ids)
        # ФИКС: embed используется ДВАЖДЫ -- как input lookup (здесь) и как
        # выходная проекция логитов (tied, защищена ce_w_embed_or_lmhead в
        # optimizer.py). Это ДВА независимых градиентных пути в один и тот же
        # параметр -- градиенты суммируются. CE-барьеры защищают только путь #2;
        # non-finite, родившийся где угодно во ВСЕЙ остальной модели (любой блок,
        # любой слой), попадает в embed через путь #1, полностью минуя CE-барьеры
        # -- это объясняет, почему embed уходит non-finite, а
        # ce_logits_chunk/ce_w_embed_or_lmhead/final_hidden_pre_ce ни разу не
        # сработали.
        x_input = make_grad_sanitizer("embed_input_lookup", clip_val=1e3)(x_input)
        x = x_input
        causal_mask = jnp.tril(jnp.ones((l, l))).astype(jnp.bool_)[None, None, :, :]

        d_head = self.cfg.d_model // self.cfg.n_heads
        cos, sin = RoPEEmbedding(dim=d_head)(l)

        history_blocks = jnp.zeros((0, b, l, self.cfg.d_model), dtype=x.dtype)

        num_blocks = self.cfg.num_layers // self.cfg.layers_per_block

        RematBlock = nn.remat(
               BlockDAR,
               static_argnums=(6, 7),
            )

        for block_idx in range(num_blocks):
            layer_idx_start = block_idx * self.cfg.layers_per_block
            x, history_blocks = RematBlock(
                cfg=self.cfg, block_idx=block_idx, layer_idx_start=layer_idx_start,
                name=f"block_{block_idx}"
            )(x, x_input, history_blocks, causal_mask, cos, sin, deterministic, rngs)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)

        if return_hidden:
            return final

        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(
                self.cfg.vocab_size,
                use_bias=False,
                name="lm_head",
                dtype=jnp.bfloat16,
            )(final)
        return logits
