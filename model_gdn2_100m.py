"""
model_gdn2_100m.py -- GDN-2-only hybrid, ~100M parameters, byte-level (BPB).

Deliberately stripped down vs the full project (model.py): NO Mamba2, NO
MLA, NO MoE. Every sublayer is GDN-2 (atomic_ops.gdn2_pipeline's
gdn2_pallas_forward_trainable -- forward Pallas A->B->C->D, backward fused
Pallas B1-B5, both from your validated atomic_ops/ package). "Block delta"
architecture (BlockDAR/BlockDARLayer/HybridDARAttention/IntraBlockAttention)
is kept exactly as in the full project -- it's the DAG-of-blocks residual
routing structure, independent of which sublayer type sits inside each
block. Since there's no MoE here, each block's post-layer mixing step is
just the DAR-accumulated residual (no extra FFN) -- see BlockDAR below.

Byte-level vocab (256 byte values + no extra specials needed for raw BPB
training) means the loss is directly bits-per-byte:
    bpb = cross_entropy_nats / ln(2)
No BPE tokenizer needed -- this matches the project roadmap's "scale to
~100M using pure GDN-2 layers on TinyStories+TinyShakespeare+Kazakh" item,
using byte-level BPB as the standard cross-dataset/cross-tokenizer metric.

Sizing note: GDN-2 kernels (atomic_ops/gdn2_fwd.py, kernel_a_scores.py)
hard-assert d_head == 128 (MXU tile) and seq_len % BT(=256) == 0. Given
that constraint, d_model must be a multiple of 128. The config below
(d_model=1024, n_heads=8, num_layers=12, layers_per_block=3) lands close
to 100M -- exact count is printed at init time via `count_params`; tune
`num_layers`/`d_model` if you need to hit a different target.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from atomic_ops.gdn2_fwd import BT as GDN2_PALLAS_BT if False else None  # placeholder, see below

# atomic_ops.gdn2_fwd doesn't export BT directly at module level under that
# name in every version of this package -- grab it defensively from the
# config module instead (config.DEFAULT_CONFIG.bt), which is always present.
from atomic_ops.config import DEFAULT_CONFIG as _GDN2_CFG
GDN2_PALLAS_BT = _GDN2_CFG.bt


# ==========================================================================
# Model mesh plumbing (same pattern as the full project's model.py)
# ==========================================================================
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
# Config
# ==========================================================================
@struct.dataclass
class ModelConfig:
    d_model: int = 1024
    n_heads: int = 8                 # d_head = d_model / n_heads MUST equal 128 (Pallas MXU tile)
    d_latent: int = 512              # DAR/intra attention projection dim
    num_layers: int = 12             # all "gdn2"
    layers_per_block: int = 3
    vocab_size: int = 256            # raw bytes, 0-255
    dropout_rate: float = 0.0        # no dropout needed for a byte-level GDN-2-only stack
    tie_embeddings: bool = True
    label_smoothing: float = 0.0
    d_conv: int = 4


def make_grad_sanitizer(tag: str, clip_val: float = 1e3):
    """Active backward clip -- same convention as the full project's
    model.py/optimizer.py. Kept here (not imported) so this file has zero
    dependency on the full model.py."""
    @jax.custom_vjp
    def _sanitizer(x):
        return x

    def _fwd(x):
        return x, None

    def _bwd(_, g):
        g_safe = jnp.nan_to_num(jnp.clip(g, -clip_val, clip_val), nan=0.0,
                                 posinf=clip_val, neginf=-clip_val)
        return (g_safe,)

    _sanitizer.defvjp(_fwd, _bwd)
    return _sanitizer


# ==========================================================================
# GDN-2 sublayer (identical math to the full project's GatedDeltaNet2J,
# minus nothing -- this IS the "gdn2" layer_type, just the only one here)
# ==========================================================================
class GatedDeltaNet2J(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        n_heads = self.cfg.n_heads
        d_head = d // n_heads
        assert d_head == 128, (
            f"GDN-2 Pallas kernels require d_head==128 (MXU tile); got "
            f"d_model={d}, n_heads={n_heads} -> d_head={d_head}. Adjust "
            f"ModelConfig.d_model/n_heads so d_model == n_heads * 128."
        )
        eps = 1e-6

        def short_causal_conv(name, u):
            conv_w = self.param(f"{name}_conv_w", nn.initializers.normal(stddev=0.02),
                                 (d, self.cfg.d_conv))
            conv_b = self.param(f"{name}_conv_b", nn.initializers.zeros, (d,))
            rhs = conv_w.T[:, None, :].astype(u.dtype)
            out = jax.lax.conv_general_dilated(
                lhs=u, rhs=rhs, window_strides=(1,),
                padding=[(self.cfg.d_conv - 1, 0)],
                feature_group_count=d,
                dimension_numbers=('NHC', 'HIO', 'NHC'),
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

        b_gate = jax.nn.sigmoid(
            nn.Dense(d, use_bias=True, name="erase_gate", dtype=jnp.bfloat16)(x)
        ).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(
            nn.Dense(d, use_bias=True, name="write_gate", dtype=jnp.bfloat16)(x)
        ).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,)).astype(jnp.float32)
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj", dtype=jnp.bfloat16)(x)
        f_proj = f_proj.reshape(b, l, n_heads, d_head)
        a_param_safe = jnp.clip(a_param, -20.0, 20.0)
        g = -jnp.exp(a_param_safe)[None, None, :, None] * jax.nn.softplus(f_proj.astype(jnp.float32))
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate", dtype=jnp.bfloat16)(x)
        out_gate = jnp.clip(out_gate, -1e2, 1e2)

        def _sanitize(t):
            return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)

        q, k, v, w_gate, b_gate, g = map(_sanitize, (q, k, v, w_gate, b_gate, g))

        if l % GDN2_PALLAS_BT != 0:
            raise ValueError(
                f"seq_len={l} must be divisible by the Pallas GDN-2 chunk size "
                f"({GDN2_PALLAS_BT}). See atomic_ops/config.py DEFAULT_CONFIG.bt."
            )

        mesh = get_model_mesh()
        batch_axis = get_batch_axis()

        _gdn2_fixed = partial(gdn2_pallas_forward_trainable, scale=1.0, h0=None)
        in_spec = None
        if mesh is not None:
            from jax.sharding import PartitionSpec as P
            in_spec = P(batch_axis, None, None, None)
            out_spec = (in_spec, in_spec)
            sharded_gdn2 = jax.shard_map(
                _gdn2_fixed, mesh=mesh,
                in_specs=(in_spec, in_spec, in_spec, in_spec, in_spec, in_spec),
                out_specs=out_spec, check_vma=False,
            )
            out, _h_final = sharded_gdn2(q, k, v, w_gate, b_gate, g)
        else:
            out, _h_final = _gdn2_fixed(q, k, v, w_gate, b_gate, g)

        out = out.reshape(b, l, d)
        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out).astype(x.dtype)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(
            out * jax.nn.silu(out_gate)
        )


# ==========================================================================
# "Block delta" plumbing -- HybridDARAttention / IntraBlockAttention /
# BlockDARLayer / BlockDAR, unchanged in structure from the full project,
# minus the MoE tail (no MoE in this build -- block output is just the
# DAR-accumulated residual).
# ==========================================================================
class IntraBlockAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, sources, block_input):
        if len(sources) == 1:
            return sources[0]

        q = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(block_input)
        k_list = [
            nn.Dense(self.cfg.d_latent, use_bias=False, name=f"k_proj_{i}", dtype=jnp.bfloat16)(src)
            for i, src in enumerate(sources)
        ]
        k_stack = jnp.stack(k_list, axis=0)
        scores = jnp.einsum("bld,nbld->nbl", q, k_stack) / jnp.sqrt(
            jnp.array(self.cfg.d_latent, dtype=q.dtype)
        )
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
            return all_sources[0].astype(current_x.dtype)

        b, l, d = current_x.shape
        stack = jnp.stack(all_sources, axis=0)
        flat = stack.reshape(n * b * l, d)
        k_flat = nn.Dense(self.cfg.d_latent, use_bias=False, name="k_proj", dtype=jnp.bfloat16)(flat)
        k = k_flat.reshape(n, b, l, self.cfg.d_latent)
        q = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_proj", dtype=jnp.bfloat16)(current_x)
        scores = jnp.einsum("bld,nbld->nbl", q, k) / jnp.sqrt(
            jnp.array(self.cfg.d_latent, dtype=q.dtype)
        )
        weights = jax.nn.softmax(scores, axis=0)
        retrieved = jnp.einsum("nbl,nbld->bld", weights, stack)
        return retrieved.astype(current_x.dtype)


class BlockDARLayer(nn.Module):
    cfg: ModelConfig
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, block_input, local_deltas, history_blocks):
        dar_sources = []
        if history_blocks.shape[0] > 0:
            dar_sources.extend([history_blocks[j] for j in range(history_blocks.shape[0])])
        dar_sources.extend(local_deltas)

        retrieved = HybridDARAttention(cfg=self.cfg, name="dar")(current_x, dar_sources)
        current_x = current_x + retrieved

        intra_sources = [block_input] + list(local_deltas)
        mixed = IntraBlockAttention(cfg=self.cfg, name="intra")(intra_sources, block_input)
        mixed = nn.RMSNorm(epsilon=1e-6, name="pre_sublayer_norm")(mixed)

        delta = GatedDeltaNet2J(cfg=self.cfg, name="sublayer")(mixed)
        delta = make_grad_sanitizer(f"delta_fanin_layer{self.layer_idx}")(delta)
        delta = jnp.nan_to_num(jnp.clip(delta, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
        return delta


class BlockDAR(nn.Module):
    cfg: ModelConfig
    block_idx: int
    layer_idx_start: int

    @nn.compact
    def __call__(self, current_x, history_blocks):
        block_input = current_x
        local_deltas = []

        for i in range(self.cfg.layers_per_block):
            layer_idx = self.layer_idx_start + i
            delta = BlockDARLayer(cfg=self.cfg, layer_idx=layer_idx, name=f"layer_{layer_idx}")(
                current_x, block_input, local_deltas, history_blocks
            )
            local_deltas.append(delta)
            current_x = jnp.nan_to_num(
                jnp.clip(current_x + delta, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3
            )

        block_delta = sum(local_deltas)
        new_history = jnp.concatenate([history_blocks, block_delta[None, ...]], axis=0)

        # No MoE tail in this build -- "output" is just the accumulated
        # residual from the GDN-2 sublayers inside this block.
        output = current_x
        return output, new_history


class FullGDN2BlockDeltaModel(nn.Module):
    """The complete model: byte-embedding -> N blocks of (layers_per_block
    GDN-2 sublayers, wired via block-delta DAR/intra attention) -> RMSNorm
    -> tied output projection to byte logits."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None, return_hidden: bool = False):
        b, l = input_ids.shape
        embed_layer = nn.Embed(
            num_embeddings=self.cfg.vocab_size, features=self.cfg.d_model,
            name="embed", dtype=jnp.bfloat16,
        )
        x = embed_layer(input_ids)
        x = make_grad_sanitizer("embed_input_lookup", clip_val=1e3)(x)

        history_blocks = jnp.zeros((0, b, l, self.cfg.d_model), dtype=x.dtype)
        num_blocks = self.cfg.num_layers // self.cfg.layers_per_block

        RematBlock = nn.remat(BlockDAR, static_argnums=())

        for block_idx in range(num_blocks):
            layer_idx_start = block_idx * self.cfg.layers_per_block
            x, history_blocks = RematBlock(
                cfg=self.cfg, block_idx=block_idx, layer_idx_start=layer_idx_start,
                name=f"block_{block_idx}",
            )(x, history_blocks)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)
        if return_hidden:
            return final

        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head",
                               dtype=jnp.bfloat16)(final)
        return logits


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
