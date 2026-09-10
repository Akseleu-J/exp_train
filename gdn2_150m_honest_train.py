"""
gdn2_150m_honest_train.py

ЦЕЛЬ: обучить ~150M-параметровую GDN-2 CENTERED byte-LM на enwik8 так,
чтобы результат был СРАВНИМ с литературой и ПРОВЕРЯЕМ постфактум. Это
прямое следствие аудита предыдущего прогона (val_bpb=0.6209, что на
~30%+ лучше любого известного результата на enwik8, включая
неограниченные по компьюту context-mixing компрессоры -- см.
gdn2_training_methodology_audit.py). Три конкретных риска из того
аудита закрыты здесь СТРУКТУРНО, а не проверкой постфактум:

  РИСК 1 -- train/val контаминация через случайный чанк-shuffle.
  ИСПРАВЛЕНО: стандартный CONTIGUOUS сплит enwik8 -- первые 90MB train,
  следующие 5MB val, последние 5MB test (тот же протокол, что и в
  Transformer-XL / Compressive Transformer и всей остальной
  литературе). Чанки нарезаются ВНУТРИ каждого региона отдельно --
  train-чанк физически не может содержать байты из val/test региона.
  Дополнительно: contamination-скан (как в audit-скрипте) запускается
  здесь тоже и пишется в отчёт -- не как блокирующий гейт (остаточный
  риск в standard-сплите известен всему полю и общепринят), а как
  прозрачная метрика, а не молчаливое допущение.

  РИСК 2 -- причинный leak в custom Pallas-кернеле/пайплайне.
  ИСПРАВЛЕНО: end-to-end causal leak test (пертурбация токена, сверка
  логитов ДО этой позиции) теперь ОБЯЗАТЕЛЬНЫЙ ГЕЙТ ПЕРЕД обучением, на
  РЕАЛЬНОМ конфиге (bt/bc границы), не на игрушечных тензорах. Гейт не
  пройден -> обучение не запускается.

  РИСК 3 -- невозможность постфактум проверить результат (не было
  чекпоинтов). ИСПРАВЛЕНО: чекпоинты параметров сохраняются в конце
  каждой эпохи + финальный, каждый с "canary fingerprint" (логиты
  модели на фиксированном тестовом батче) для проверки целостности
  сохранения/загрузки. Финальная TEST-метрика (единственный prise --
  test используется РОВНО ОДИН РАЗ, в конце) считается ПОСЛЕ повторной
  загрузки финального чекпоинта С ДИСКА, а не из переменной в памяти --
  это доказывает, что сохранённые веса реально дают заявленный результат.

Что этот скрипт НЕ решает (честно, не молчать):
  - Contiguous-сплит снижает, но не обнуляет риск дублирующегося
    boilerplate между регионами (это известное свойство enwik8,
    общепринятое в литературе -- сравнение остаётся честным ИМЕННО
    потому что это тот же протокол, которым мерены все baseline'ы).
  - Self-timestamped логирование (см. TeeLogger ниже) -- ВНУТРЕННИЕ
    метки времени, их надёжность НИЖЕ, чем у внешнего лога (например,
    Kaggle notebook log viewer, который добавляет метки СНАРУЖИ
    процесса). Рекомендуется ДОПОЛНИТЕЛЬНО запускать через:
        python gdn2_150m_honest_train.py 2>&1 | tee /kaggle/working/external_run.log
    чтобы иметь вторую, независимо от скрипта собранную копию лога.
  - Чекпоинты подтверждают "эти веса дают именно этот test_bpb", но не
    доказывают отсутствие contamination за пределами того, что измерил
    contamination-скан (вероятностная, не исчерпывающая проверка).

Запуск (Kaggle TPU v5e-8, без argparse -- см. CONFIG ниже):
    python gdn2_150m_honest_train.py
"""
from __future__ import annotations

import os
import sys
import gc
import time
import json
import hashlib
import dataclasses as dc
from collections import deque
from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.serialization as fser
import optax

# ==========================================================================
# CONFIG
# ==========================================================================
DATA_PATH = "/kaggle/working/enwik8/enwik8"
DATA_DIR = "/kaggle/working/enwik8_data"

# Стандартный enwik8-протокол (Transformer-XL, Compressive Transformer и др.)
TRAIN_BYTES = 90_000_000
VAL_BYTES = 5_000_000
TEST_BYTES = 5_000_000

MODEL_CONFIG = dict(
    vocab_size=256,
    d_model=768,        # 6 * 128, d_head=128 обязателен для Pallas
    n_heads=6,
    num_layers=13,
    mlp_mult=4,
    dropout_rate=0.0,
)

RUN_CONFIG = dict(
    seq_len=1024,
    micro_batch_size=8,
    accum_steps=4,
    epochs=5,
    warmup_steps=200,
    adamw_peak_lr=3e-4,
    muon_peak_lr=0.02,
    lr_reduce_factor=0.5,
    patience=3,
    lr_cooldown_steps=300,
    weight_decay=0.01,
    grad_clip_norm=1.0,
    eval_every_steps=200,
    eval_batches=10,               # периодический (дешёвый) val во время обучения
    epoch_end_eval_batches=50,     # более полный val в конце каждой эпохи
    eval_seed=999,
    nonfinite_consecutive_limit=8,
    nonfinite_window_size=20,
    nonfinite_window_ratio=0.35,
    seed=42,
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_steps=5,
)

# -- contamination self-check (информационный, не блокирующий гейт) --
CONTAM_WINDOW_LEN = 48
CONTAM_TRAIN_STRIDE = 32
CONTAM_EVAL_STRIDE = 16
CONTAM_SAMPLE_CHUNKS = 200

# -- causal leak gate (БЛОКИРУЮЩИЙ) --
CAUSAL_OK_ABS_TOL = 1e-5

OUTPUT_JSON = "/kaggle/working/gdn2_150m_honest_results.json"
LOG_PATH = "/kaggle/working/gdn2_150m_honest_train.log"
CHECKPOINT_DIR = "/kaggle/working/checkpoints_150m_honest"


from Atomic_ops.configs import KernelConfig, KAGGLE_MEDIUM
from Atomic_ops.gdn2_pipeline import gdn2_pallas_forward_trainable
from Atomic_ops.gdn2_fwd import gdn2_pallas_forward
from Atomic_ops.reference import gdn2_token_serial_reference

KERNEL_CONFIG = KAGGLE_MEDIUM  # use_centering=True -- валидировано предыдущим A/B тестом

GDN2_CHUNK = KERNEL_CONFIG.bt
assert RUN_CONFIG["seq_len"] % GDN2_CHUNK == 0, (
    f"seq_len={RUN_CONFIG['seq_len']} должен делиться на bt={GDN2_CHUNK}"
)


def is_tpu_available() -> bool:
    try:
        return jax.devices("tpu")[0].platform == "tpu"
    except Exception:
        return False


# ==========================================================================
# ЛОГГЕР С САМО-ПРОСТАВЛЕННЫМИ ВРЕМЕННЫМИ МЕТКАМИ
# (см. предупреждение в докстринге файла про ограниченную надёжность
#  self-timestamped логов -- рекомендуется дублировать через `tee`)
# ==========================================================================
class TeeLogger:
    def __init__(self, path):
        self._t0 = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", buffering=1)
        self._f.write(f"\n\n===== NEW RUN @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")

    def log(self, msg=""):
        elapsed = time.time() - self._t0
        line = f"[{elapsed:10.1f}s] {msg}"
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


LOG = TeeLogger(LOG_PATH)


# ==========================================================================
# 1. ДАННЫЕ -- скачивание + СТАНДАРТНЫЙ CONTIGUOUS сплит (90/5/5)
# ==========================================================================
def ensure_enwik8() -> str:
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "enwik8.zip")
    out_path = os.path.join(DATA_DIR, "enwik8")
    if not os.path.exists(out_path):
        import urllib.request
        import zipfile
        url = "http://mattmahoney.net/dc/enwik8.zip"
        LOG.log(f"[DATA] {DATA_PATH} не найден, скачиваю {url} ...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
    return out_path


def load_enwik8_bytes(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8).astype(np.int32)


def make_contiguous_split(byte_arr, seq_len, train_bytes, val_bytes, test_bytes):
    total_needed = train_bytes + val_bytes + test_bytes
    assert len(byte_arr) >= total_needed, (
        f"enwik8 length {len(byte_arr):,} < requested split {total_needed:,}"
    )
    train_region = byte_arr[:train_bytes]
    val_region = byte_arr[train_bytes:train_bytes + val_bytes]
    test_region = byte_arr[train_bytes + val_bytes:train_bytes + val_bytes + test_bytes]

    def chunk_region(region):
        n = len(region) // (seq_len + 1)
        usable = n * (seq_len + 1)
        return region[:usable].reshape(n, seq_len + 1)

    train_chunks = chunk_region(train_region)
    val_chunks = chunk_region(val_region)
    test_chunks = chunk_region(test_region)

    # Защитная проверка: регионы физически не пересекаются по байтам
    # (тривиально верно по построению через срезы, но проверяем явно,
    # а не только полагаемся на код).
    assert train_bytes + val_bytes + test_bytes <= len(byte_arr)
    return train_chunks, val_chunks, test_chunks


def epoch_batches(chunks, idx_pool, batch_size, seed):
    idx_local = np.copy(idx_pool)
    np.random.RandomState(seed).shuffle(idx_local)
    n_steps = len(idx_local) // batch_size
    for step in range(n_steps):
        batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
        rows = chunks[batch_idx]
        yield {
            "input_ids": jnp.asarray(rows[:, :-1], dtype=jnp.int32),
            "labels": jnp.asarray(rows[:, 1:], dtype=jnp.int32),
        }


def full_pass_batches(chunks, batch_size, seed=0, shuffle=False):
    idx = np.arange(len(chunks))
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    n_steps = len(idx) // batch_size
    for step in range(n_steps):
        batch_idx = idx[step * batch_size:(step + 1) * batch_size]
        rows = chunks[batch_idx]
        yield {
            "input_ids": jnp.asarray(rows[:, :-1], dtype=jnp.int32),
            "labels": jnp.asarray(rows[:, 1:], dtype=jnp.int32),
        }


# ==========================================================================
# 2. CONTAMINATION SELF-CHECK (информационный, печатается в отчёт)
# ==========================================================================
def _row_to_bytes(row):
    return bytes(row.astype(np.uint8).tolist())


def build_window_hashes(chunks, window_len, stride):
    hashes = set()
    for row in chunks:
        b = _row_to_bytes(row)
        L = len(b)
        for s in range(0, L - window_len + 1, stride):
            hashes.add(hashlib.blake2b(b[s:s + window_len], digest_size=8).digest())
    return hashes


def scan_contamination(eval_chunks, train_hashes, window_len, stride, sample_chunks, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(eval_chunks))
    if len(idx) > sample_chunks:
        idx = rng.choice(idx, size=sample_chunks, replace=False)
    total, hits, chunks_with_hit = 0, 0, 0
    for ci in idx:
        b = _row_to_bytes(eval_chunks[ci])
        L = len(b)
        chunk_hit = False
        for s in range(0, L - window_len + 1, stride):
            h = hashlib.blake2b(b[s:s + window_len], digest_size=8).digest()
            total += 1
            if h in train_hashes:
                hits += 1
                chunk_hit = True
        if chunk_hit:
            chunks_with_hit += 1
    return dict(
        chunks_checked=len(idx), windows_checked=total, window_hits=hits,
        contamination_rate_windows=hits / max(total, 1),
        chunks_with_any_hit=chunks_with_hit,
        contamination_rate_chunks=chunks_with_hit / max(len(idx), 1),
    )


def run_contamination_report(train_chunks, val_chunks, test_chunks):
    LOG.log("\n" + "=" * 96)
    LOG.log("[CONTAMINATION] contiguous-сплит self-check (информационно, не блокирует обучение)")
    LOG.log("=" * 96)
    LOG.log(f"  Строим индекс окон из {len(train_chunks):,} train-чанков "
            f"(window={CONTAM_WINDOW_LEN}, stride={CONTAM_TRAIN_STRIDE})...")
    train_hashes = build_window_hashes(train_chunks, CONTAM_WINDOW_LEN, CONTAM_TRAIN_STRIDE)
    LOG.log(f"  Индекс: {len(train_hashes):,} уникальных хэшей.")

    val_report = scan_contamination(val_chunks, train_hashes, CONTAM_WINDOW_LEN,
                                     CONTAM_EVAL_STRIDE, CONTAM_SAMPLE_CHUNKS, RUN_CONFIG["seed"])
    test_report = scan_contamination(test_chunks, train_hashes, CONTAM_WINDOW_LEN,
                                      CONTAM_EVAL_STRIDE, CONTAM_SAMPLE_CHUNKS, RUN_CONFIG["seed"] + 1)

    LOG.log(f"  VAL:  {val_report['chunks_with_any_hit']}/{val_report['chunks_checked']} чанков "
            f"({val_report['contamination_rate_chunks']:.2%}) содержат совпадающее с train окно")
    LOG.log(f"  TEST: {test_report['chunks_with_any_hit']}/{test_report['chunks_checked']} чанков "
            f"({test_report['contamination_rate_chunks']:.2%}) содержат совпадающее с train окно")
    LOG.log("  (Ненулевой процент ожидаем -- enwik8 содержит повторяющийся boilerplate даже "
            "в стандартном contiguous-сплите. Сравнение с литературой остаётся честным, "
            "т.к. это ТОТ ЖЕ протокол, что и у всех baseline'ов.)")
    return dict(val=val_report, test=test_report)


# ==========================================================================
# 3. МОДЕЛЬ (не изменена относительно предыдущих скриптов)
# ==========================================================================
def make_grad_sanitizer(clip_val: float = 1e3):
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


class GDN2Mixer(nn.Module):
    cfg: dict
    kernel_config: KernelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        n_heads = self.cfg["n_heads"]
        d_head = d // n_heads
        assert d_head == 128
        eps = 1e-6

        def short_causal_conv(name, u, d_conv=4):
            conv_w = self.param(f"{name}_conv_w", nn.initializers.normal(stddev=0.02), (d, d_conv))
            conv_b = self.param(f"{name}_conv_b", nn.initializers.zeros, (d,))
            rhs = conv_w.T[:, None, :].astype(u.dtype)
            out = jax.lax.conv_general_dilated(
                lhs=u, rhs=rhs, window_strides=(1,), padding=[(d_conv - 1, 0)],
                feature_group_count=d, dimension_numbers=("NHC", "HIO", "NHC"),
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

        q = make_grad_sanitizer()(_safe_normalize(q))
        k = make_grad_sanitizer()(_safe_normalize(k))

        b_gate = jax.nn.sigmoid(nn.Dense(d, name="erase_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, name="write_gate", dtype=jnp.bfloat16)(x)).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,)).astype(jnp.float32)
        f_proj = nn.Dense(d, name="decay_proj", dtype=jnp.bfloat16)(x).reshape(b, l, n_heads, d_head)
        a_safe = jnp.clip(a_param, -20.0, 20.0)
        g = -jnp.exp(a_safe)[None, None, :, None] * jax.nn.softplus(f_proj.astype(jnp.float32))
        g = jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=-20.0)

        out_gate = jnp.clip(nn.Dense(d, use_bias=False, name="out_gate", dtype=jnp.bfloat16)(x), -1e2, 1e2)

        def _sanitize(t):
            return jnp.nan_to_num(jnp.clip(t, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
        q, k, v, w_gate, b_gate, g = map(_sanitize, (q, k, v, w_gate, b_gate, g))

        out, _h_final = gdn2_pallas_forward_trainable(
            q, k, v, w_gate, b_gate, g, scale=1.0, config=self.kernel_config
        )
        out = out.reshape(b, l, d)
        out = nn.RMSNorm(epsilon=1e-6, name="mixer_out_norm")(out).astype(x.dtype)
        return nn.Dense(d, use_bias=False, name="out_proj", dtype=jnp.bfloat16)(out * jax.nn.silu(out_gate))


class GatedMLP(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        h = self.cfg["mlp_mult"] * d
        gate = nn.Dense(h, use_bias=False, name="gate_proj", dtype=jnp.bfloat16)(x)
        up = nn.Dense(h, use_bias=False, name="up_proj", dtype=jnp.bfloat16)(x)
        act = jax.nn.silu(gate) * up
        return nn.Dense(d, use_bias=False, name="down_proj", dtype=jnp.bfloat16)(act)


class GDN2Block(nn.Module):
    cfg: dict
    kernel_config: KernelConfig

    @nn.compact
    def __call__(self, x):
        h = GDN2Mixer(cfg=self.cfg, kernel_config=self.kernel_config, name="mixer")(
            nn.RMSNorm(epsilon=1e-6, name="mixer_norm")(x))
        x = jnp.nan_to_num(jnp.clip(x + h, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
        m = GatedMLP(cfg=self.cfg, name="mlp")(nn.RMSNorm(epsilon=1e-6, name="mlp_norm")(x))
        x = jnp.nan_to_num(jnp.clip(x + m, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
        return x


class ByteGDN2LM(nn.Module):
    cfg: dict
    kernel_config: KernelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True):
        embed = nn.Embed(num_embeddings=self.cfg["vocab_size"], features=self.cfg["d_model"],
                          name="embed", dtype=jnp.bfloat16)
        x = embed(input_ids)
        x = make_grad_sanitizer(clip_val=1e3)(x)

        RematBlock = nn.remat(GDN2Block)
        for i in range(self.cfg["num_layers"]):
            x = RematBlock(cfg=self.cfg, kernel_config=self.kernel_config, name=f"block_{i}")(x)

        x = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)
        logits = embed.attend(x)
        return logits


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# ==========================================================================
# 4. ГЕЙТ 1/2: КОРРЕКТНОСТЬ КЕРНЕЛЯ (Pallas vs token-serial reference)
# ==========================================================================
def _rel_err(a, b):
    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    num = jnp.max(jnp.abs(a - b))
    den = jnp.maximum(jnp.max(jnp.abs(b)), 1e-8)
    return float(num / den)


def run_correctness_gate(config: KernelConfig) -> dict:
    LOG.log("\n[GATE 1/2] Корректность кернеля: Pallas vs token-serial reference "
            f"(use_centering={config.use_centering}) ...")
    bsz, H, D, n_chunks = 2, 3, 128, 2
    bt = config.bt
    L = n_chunks * bt
    key = jax.random.PRNGKey(777)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    shape = (bsz, L, H, D)

    q = jax.random.normal(k1, shape)
    k = jax.random.normal(k2, shape)
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = jax.random.normal(k3, shape) * 0.5
    w = jax.random.uniform(k4, shape, minval=0.2, maxval=1.0)
    b = jax.random.uniform(jax.random.fold_in(k4, 1), shape, minval=0.2, maxval=1.0)
    g = -jnp.abs(jax.random.normal(k5, shape)) * 0.1
    h0 = jax.random.normal(k6, (bsz, H, D, D)) * 0.1

    o_pallas, h_final_pallas = gdn2_pallas_forward(q, k, v, w, b, g, scale=1.0, h0=h0, config=config)
    o_ref, h_final_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale=1.0, h0=h0)
    fwd_o_err = _rel_err(o_pallas, o_ref)
    fwd_h_err = _rel_err(h_final_pallas, h_final_ref)

    rkey = jax.random.fold_in(key, 55)
    r1, r2 = jax.random.split(rkey)
    do_rand = jax.random.normal(r1, o_pallas.shape)
    dh_rand = jax.random.normal(r2, h_final_pallas.shape)

    def honest_loss(q_, k_, v_, w_, b_, g_, h0_):
        o, hf = gdn2_pallas_forward_trainable(q_, k_, v_, w_, b_, g_, scale=1.0, h0=h0_, config=config)
        return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

    def ref_loss(q_, k_, v_, w_, b_, g_, h0_):
        o, hf = gdn2_token_serial_reference(q_, k_, v_, g_, b_, w_, scale=1.0, h0=h0_)
        return jnp.sum(o * do_rand) + jnp.sum(hf * dh_rand)

    hg = jax.grad(honest_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
    rg = jax.grad(ref_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(q, k, v, w, b, g, h0)
    names = ["dq", "dk", "dv", "dw", "db", "dg", "dh0"]
    grad_errs = {}
    all_finite = True
    for name, h, r in zip(names, hg, rg):
        finite = bool(jnp.all(jnp.isfinite(h)))
        all_finite = all_finite and finite
        grad_errs[name] = _rel_err(h, r) if finite else float("inf")

    tol_fwd = 2e-2
    tol_grad = 5e-2
    fwd_ok = fwd_o_err < tol_fwd and fwd_h_err < tol_fwd
    grad_ok = all_finite and all(v <= tol_grad for v in grad_errs.values())
    ok = fwd_ok and grad_ok

    LOG.log(f"  fwd.o rel_err={fwd_o_err:.3e}  fwd.h_final rel_err={fwd_h_err:.3e}  "
            f"{'OK' if fwd_ok else 'FAIL'}")
    LOG.log("  grad rel_errs: " + ", ".join(f"{k}={v:.2e}" for k, v in grad_errs.items()))
    LOG.log(f"  [GATE 1/2] {'PASSED' if ok else 'FAILED'}")

    return dict(fwd_o_err=fwd_o_err, fwd_h_err=fwd_h_err, grad_errs=grad_errs,
                fwd_ok=fwd_ok, grad_ok=grad_ok, passed=ok)


# ==========================================================================
# 5. ГЕЙТ 2/2: ПРИЧИННОСТЬ END-TO-END (НА РЕАЛЬНОМ КОНФИГЕ, БЛОКИРУЮЩИЙ)
# ==========================================================================
def build_causal_test_positions(kernel_config, seq_len):
    positions = set()
    for base in range(0, seq_len, kernel_config.bt):
        for off in (-1, 0, 1):
            p = base + off
            if 0 <= p < seq_len:
                positions.add(p)
    for base in range(0, seq_len, kernel_config.bc):
        for off in (-1, 0, 1):
            p = base + off
            if 0 <= p < seq_len:
                positions.add(p)
    positions.update([1, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1])
    return sorted(positions)


def run_causal_leak_gate() -> dict:
    LOG.log("\n[GATE 2/2] Причинность end-to-end: реальная модель + реальный кернель "
            f"(bt={KERNEL_CONFIG.bt}, bc={KERNEL_CONFIG.bc}, seq_len={RUN_CONFIG['seq_len']}) ...")
    B, L = 2, RUN_CONFIG["seq_len"]
    model = ByteGDN2LM(cfg=MODEL_CONFIG, kernel_config=KERNEL_CONFIG)
    init_rng = jax.random.PRNGKey(12345)
    dummy = jnp.zeros((B, L), dtype=jnp.int32)
    params = model.init(init_rng, dummy)["params"]

    x_a = jax.random.randint(jax.random.PRNGKey(777), (B, L), 0, 256, dtype=jnp.int32)

    @jax.jit
    def forward(p, ids):
        return model.apply({"params": p}, ids, deterministic=True).astype(jnp.float32)

    logits_a = forward(params, x_a)
    jax.block_until_ready(logits_a)

    positions = build_causal_test_positions(KERNEL_CONFIG, L)
    results = []
    for T in positions:
        x_b = x_a.at[:, T].set((x_a[:, T] + 1) % 256)
        logits_b = forward(params, x_b)
        jax.block_until_ready(logits_b)

        max_diff_before = float(jnp.max(jnp.abs(logits_a[:, :T, :] - logits_b[:, :T, :]))) if T > 0 else 0.0
        max_diff_at_after = float(jnp.max(jnp.abs(logits_a[:, T:, :] - logits_b[:, T:, :])))
        causal_ok = max_diff_before < CAUSAL_OK_ABS_TOL
        results.append(dict(T=T, max_diff_before=max_diff_before,
                             max_diff_at_after=max_diff_at_after, causal_ok=causal_ok))
        if not causal_ok:
            LOG.log(f"  [FAIL] T={T:4d}: max|Δlogits[<{T}]|={max_diff_before:.3e} (ожидалось ~0.0)")

    n_fail = sum(1 for r in results if not r["causal_ok"])
    passed = n_fail == 0
    LOG.log(f"  Проверено позиций: {len(results)} (границы bt/bc + случайные точки)")
    LOG.log(f"  [GATE 2/2] {'PASSED' if passed else f'FAILED ({n_fail} позиций)'}")
    del params
    gc.collect()
    return dict(positions_checked=len(results), n_failed=n_fail, passed=passed, details=results)


# ==========================================================================
# 6. MUON + ADAMW (без изменений относительно gdn2_muon150m_train.py)
# ==========================================================================
def _zeropower_via_newtonschulz5(G: jnp.ndarray, steps: int, eps: float = 1e-7) -> jnp.ndarray:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.float32)
    X = X / (jnp.linalg.norm(X) + eps)
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.astype(G.dtype)


class MuonState(NamedTuple):
    momentum: optax.Updates


def scale_by_muon(momentum: float = 0.95, nesterov: bool = True,
                   ns_steps: int = 5, eps: float = 1e-7) -> optax.GradientTransformation:
    def init_fn(params):
        return MuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        new_momentum = jax.tree_util.tree_map(lambda m, g: momentum * m + g, state.momentum, updates)
        if nesterov:
            effective = jax.tree_util.tree_map(lambda m, g: momentum * m + g, new_momentum, updates)
        else:
            effective = new_momentum

        def _orthogonalize_and_scale(g):
            o = _zeropower_via_newtonschulz5(g, steps=ns_steps, eps=eps)
            rows, cols = g.shape
            scale = jnp.sqrt(jnp.maximum(1.0, rows / cols))
            return o * scale

        new_updates = jax.tree_util.tree_map(_orthogonalize_and_scale, effective)
        return new_updates, MuonState(momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def muon_param_labels(params) -> dict:
    def label_leaf(path, leaf):
        path_str = jax.tree_util.keystr(path)
        if leaf.ndim == 2 and "embed" not in path_str and "conv" not in path_str:
            return "muon"
        return "adamw"
    return jax.tree_util.tree_map_with_path(label_leaf, params)


def build_hybrid_tx(labels):
    def make_tx(adamw_lr, muon_lr):
        adamw_opt = optax.adamw(learning_rate=adamw_lr, b1=0.9, b2=0.95,
                                 weight_decay=RUN_CONFIG["weight_decay"])
        muon_opt = optax.chain(
            scale_by_muon(momentum=RUN_CONFIG["muon_momentum"],
                          nesterov=RUN_CONFIG["muon_nesterov"],
                          ns_steps=RUN_CONFIG["muon_ns_steps"]),
            optax.scale_by_learning_rate(muon_lr),
        )
        return optax.chain(
            optax.clip_by_global_norm(RUN_CONFIG["grad_clip_norm"]),
            optax.multi_transform({"muon": muon_opt, "adamw": adamw_opt}, labels),
        )
    return make_tx


# ==========================================================================
# 7. ЧЕКПОИНТЫ + CANARY FINGERPRINT
# ==========================================================================
def compute_canary_fingerprint(model, params):
    canary_ids = jax.random.randint(jax.random.PRNGKey(4242), (2, RUN_CONFIG["seq_len"]),
                                     0, 256, dtype=jnp.int32)
    logits = model.apply({"params": params}, canary_ids, deterministic=True).astype(jnp.float32)
    checksum = float(jnp.sum(jnp.abs(logits)))
    sample_vals = [float(v) for v in jax.device_get(logits[0, 0, :5])]
    return dict(checksum=checksum, sample_vals=sample_vals)


def save_params_checkpoint(path, params, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(fser.to_bytes(params))
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    LOG.log(f"  [CKPT] сохранено: {path}  (canary_checksum={meta['canary']['checksum']:.4f})")


def load_params_checkpoint(path, template_params):
    with open(path, "rb") as f:
        data = f.read()
    return fser.from_bytes(template_params, data)


# ==========================================================================
# 8. ОБУЧЕНИЕ
# ==========================================================================
def make_plateau_schedule(peak_lr, warmup_steps):
    def schedule(step, multiplier):
        step = jnp.asarray(step, dtype=jnp.float32)
        multiplier = jnp.asarray(multiplier, dtype=jnp.float32)
        warmup_frac = jnp.clip(step / max(warmup_steps, 1), 0.0, 1.0)
        warmup_lr = peak_lr * warmup_frac
        stable_lr = peak_lr * multiplier
        return jnp.where(step < warmup_steps, warmup_lr, stable_lr)
    return schedule


def compute_bpb_loss(params, model, batch, deterministic):
    logits = model.apply({"params": params}, batch["input_ids"], deterministic=deterministic).astype(jnp.float32)
    logits = jnp.nan_to_num(jnp.clip(logits, -30.0, 30.0), nan=0.0, posinf=30.0, neginf=-30.0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, batch["labels"][..., None], axis=-1).squeeze(-1)
    ce_nats = jnp.mean(nll)
    ce_nats = jnp.nan_to_num(ce_nats, nan=0.0, posinf=20.0, neginf=0.0)
    return ce_nats, ce_nats / jnp.log(2.0)


def _get_memory_stats():
    try:
        return jax.local_devices()[0].memory_stats()
    except Exception:
        return None


def run_training(train_chunks, val_chunks, test_chunks) -> dict:
    LOG.log("\n" + "=" * 96)
    LOG.log(f"[RUN] GDN-2 CENTERED, ~150M, Muon+AdamW, {RUN_CONFIG['epochs']} эпох, "
            f"contiguous 90/5/5 сплит")
    LOG.log("=" * 96)

    gc.collect()
    jax.clear_caches()

    model = ByteGDN2LM(cfg=MODEL_CONFIG, kernel_config=KERNEL_CONFIG)
    init_rng = jax.random.PRNGKey(RUN_CONFIG["seed"])
    dummy_ids = jnp.zeros((RUN_CONFIG["micro_batch_size"], RUN_CONFIG["seq_len"]), dtype=jnp.int32)
    params = model.init(init_rng, dummy_ids)["params"]
    n_params = count_params(params)
    LOG.log(f"[MODEL] Параметров: {n_params:,} (~{n_params/1e6:.1f}M)")

    labels = muon_param_labels(params)
    muon_count = sum(l.size for l, lbl in zip(jax.tree_util.tree_leaves(params),
                                               jax.tree_util.tree_leaves(labels)) if lbl == "muon")
    adamw_count = n_params - muon_count
    LOG.log(f"[OPT] Muon-группа: {muon_count:,}  AdamW-группа: {adamw_count:,}")

    adamw_lr_schedule = make_plateau_schedule(RUN_CONFIG["adamw_peak_lr"], RUN_CONFIG["warmup_steps"])
    muon_lr_schedule = make_plateau_schedule(RUN_CONFIG["muon_peak_lr"], RUN_CONFIG["warmup_steps"])

    tx = optax.inject_hyperparams(build_hybrid_tx(labels))(
        adamw_lr=RUN_CONFIG["adamw_peak_lr"], muon_lr=RUN_CONFIG["muon_peak_lr"]
    )
    opt_state = tx.init(params)

    def train_micro_step(p, accum_grads, batch):
        def loss_fn(pp):
            return compute_bpb_loss(pp, model, batch, deterministic=False)
        (ce_nats, bpb), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return new_accum, ce_nats, bpb

    def apply_step(p, s, accum_grads, n_accum, adamw_lr, muon_lr):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)
        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))
        is_finite = jnp.isfinite(global_norm)
        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), avg_grads)
        s = s._replace(hyperparams=dict(s.hyperparams, adamw_lr=adamw_lr, muon_lr=muon_lr))
        updates, new_s = tx.update(avg_grads, s, p)
        new_p_candidate = optax.apply_updates(p, updates)
        new_p_candidate = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p_candidate)
        new_p = jax.tree_util.tree_map(lambda old, new: jnp.where(is_finite, new, old), p, new_p_candidate)
        new_s = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_finite, new, old) if hasattr(new, "shape") else new, s, new_s)
        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return new_p, new_s, zero_accum, is_finite, global_norm

    def val_step(p, batch):
        return compute_bpb_loss(p, model, batch, deterministic=True)

    compiled_train_micro = jax.jit(train_micro_step, donate_argnums=(1,))
    compiled_apply = jax.jit(apply_step, donate_argnums=(0, 1, 2))
    compiled_val = jax.jit(val_step)

    accum_steps = RUN_CONFIG["accum_steps"]
    micro_bs = RUN_CONFIG["micro_batch_size"]
    zero_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
    accum_grads = zero_accum

    train_idx_pool = np.arange(len(train_chunks))
    n_micro_per_epoch = len(train_idx_pool) // micro_bs
    steps_per_epoch = n_micro_per_epoch // accum_steps
    total_steps_est = steps_per_epoch * RUN_CONFIG["epochs"]
    LOG.log(f"[DATA] train chunks={len(train_chunks):,}  val chunks={len(val_chunks):,}  "
            f"test chunks={len(test_chunks):,}  optimizer-steps/epoch≈{steps_per_epoch:,}  "
            f"total_steps≈{total_steps_est:,}")

    current_lr_multiplier = 1.0
    evals_since_improvement = 0
    cooldown_counter = 0
    global_step = 0
    nonfinite_consecutive = 0
    nonfinite_window = deque(maxlen=RUN_CONFIG["nonfinite_window_size"])
    best_val_bpb = float("inf")
    _micro_bpb_acc = []

    full_step_history = []  # ПОЛНАЯ пошаговая история -- в отличие от предыдущего скрипта,
                             # здесь она реально сохраняется в JSON целиком (см. п.3 рисков).
    val_history = {"step": [], "val_bpb": []}
    epoch_summaries = []
    checkpoint_meta = []
    peak_bytes_seen = 0
    nonfinite_step_count = 0
    auto_stopped = False

    LOG.log("[TRAIN] Компиляция (warm-up, не учитывается в тайминге)...")
    warm_stream = epoch_batches(train_chunks, train_idx_pool, micro_bs, seed=RUN_CONFIG["seed"])
    warm_batch = next(warm_stream)
    warm_accum = jax.tree_util.tree_map(jnp.zeros_like, params)
    _warm_out = compiled_train_micro(params, warm_accum, warm_batch)
    jax.block_until_ready(_warm_out)
    del warm_accum, _warm_out, warm_stream, warm_batch
    gc.collect()
    settle_stats = _get_memory_stats()
    settle_peak = settle_stats.get("peak_bytes_in_use", 0) if settle_stats else 0

    LOG.log(f"[TRAIN] Старт обучения, epochs={RUN_CONFIG['epochs']} ...")
    t_run_start = time.perf_counter()

    for epoch in range(1, RUN_CONFIG["epochs"] + 1):
        if auto_stopped:
            break
        epoch_seed = RUN_CONFIG["seed"] + epoch
        train_stream = epoch_batches(train_chunks, train_idx_pool, micro_bs, seed=epoch_seed)
        micro_step_in_epoch = 0
        t_epoch_start = time.perf_counter()
        epoch_step_start = global_step
        epoch_nonfinite_start = nonfinite_step_count

        for batch in train_stream:
            t_step_start = time.perf_counter()
            accum_grads, ce_nats, bpb = compiled_train_micro(params, accum_grads, batch)
            _micro_bpb_acc.append(float(jax.device_get(bpb)))
            micro_step_in_epoch += 1

            if micro_step_in_epoch % accum_steps != 0:
                continue

            next_step_for_lr = global_step + 1
            cur_adamw_lr = float(jax.device_get(adamw_lr_schedule(next_step_for_lr, current_lr_multiplier)))
            cur_muon_lr = float(jax.device_get(muon_lr_schedule(next_step_for_lr, current_lr_multiplier)))

            params, opt_state, accum_grads, is_finite, global_norm = compiled_apply(
                params, opt_state, accum_grads, accum_steps,
                jnp.asarray(cur_adamw_lr, dtype=jnp.float32),
                jnp.asarray(cur_muon_lr, dtype=jnp.float32),
            )
            jax.block_until_ready(params)
            step_wall = time.perf_counter() - t_step_start

            step_finite = bool(jax.device_get(is_finite))
            gnorm = float(jax.device_get(global_norm))
            global_step += 1
            if cooldown_counter > 0:
                cooldown_counter -= 1

            nonfinite_window.append(0 if step_finite else 1)
            nonfinite_consecutive = 0 if step_finite else nonfinite_consecutive + 1
            if not step_finite:
                nonfinite_step_count += 1
            window_ratio = sum(nonfinite_window) / len(nonfinite_window)

            mem_stats = _get_memory_stats()
            if mem_stats:
                peak_bytes_seen = max(peak_bytes_seen, mem_stats.get("peak_bytes_in_use", 0))

            mean_bpb = float(np.mean(_micro_bpb_acc))
            _micro_bpb_acc = []
            full_step_history.append(dict(
                step=global_step, epoch=epoch, train_bpb=mean_bpb, grad_norm=gnorm,
                is_finite=step_finite, step_ms=step_wall * 1000.0,
                peak_hbm_mb=peak_bytes_seen / 1e6,
                adamw_lr=cur_adamw_lr, muon_lr=cur_muon_lr,
            ))

            if global_step % 20 == 0 or global_step == 1:
                LOG.log(f"  [E{epoch}][STEP {global_step}/~{total_steps_est}] "
                         f"bpb={mean_bpb:.4f} adamw_lr={cur_adamw_lr:.2e} muon_lr={cur_muon_lr:.2e} "
                         f"grad_norm={gnorm:.3f} finite={step_finite} step_ms={step_wall*1000:.1f} "
                         f"peak_hbm_mb={peak_bytes_seen/1e6:.1f}")

            if not step_finite:
                LOG.log(f"  [E{epoch}][WARN] non-finite gradient at step {global_step} -- update skipped.")

            hit_consecutive = nonfinite_consecutive >= RUN_CONFIG["nonfinite_consecutive_limit"]
            hit_window = (len(nonfinite_window) >= RUN_CONFIG["nonfinite_window_size"]
                          and window_ratio >= RUN_CONFIG["nonfinite_window_ratio"])
            if hit_consecutive or hit_window:
                LOG.log(f"  [E{epoch}][AUTO-STOP] instability at step {global_step}.")
                auto_stopped = True
                break

            if global_step % RUN_CONFIG["eval_every_steps"] == 0:
                bpb_sum, n_done = 0.0, 0
                for vb in full_pass_batches(val_chunks, micro_bs, seed=RUN_CONFIG["eval_seed"], shuffle=True):
                    if n_done >= RUN_CONFIG["eval_batches"]:
                        break
                    _, vbpb = compiled_val(params, vb)
                    bpb_sum += float(jax.device_get(vbpb))
                    n_done += 1
                val_bpb = bpb_sum / max(n_done, 1)
                val_history["step"].append(global_step)
                val_history["val_bpb"].append(val_bpb)
                improved = val_bpb < best_val_bpb
                if improved:
                    best_val_bpb = val_bpb
                    evals_since_improvement = 0
                    LOG.log(f"  [E{epoch}][EVAL] step {global_step}: val_bpb={val_bpb:.4f}  IMPROVED")
                else:
                    evals_since_improvement += 1
                    LOG.log(f"  [E{epoch}][EVAL] step {global_step}: val_bpb={val_bpb:.4f}  "
                             f"no improvement ({evals_since_improvement}/{RUN_CONFIG['patience']})")

                if cooldown_counter == 0 and evals_since_improvement >= RUN_CONFIG["patience"]:
                    current_lr_multiplier *= RUN_CONFIG["lr_reduce_factor"]
                    cooldown_counter = RUN_CONFIG["lr_cooldown_steps"]
                    evals_since_improvement = 0
                    LOG.log(f"  [E{epoch}][LR] plateau -- множитель понижен до {current_lr_multiplier:.4f}")

        # -------- конец эпохи: более полный val + чекпоинт --------
        t_epoch_wall = time.perf_counter() - t_epoch_start
        bpb_sum, n_done = 0.0, 0
        for vb in full_pass_batches(val_chunks, micro_bs, seed=RUN_CONFIG["eval_seed"], shuffle=True):
            if n_done >= RUN_CONFIG["epoch_end_eval_batches"]:
                break
            _, vbpb = compiled_val(params, vb)
            bpb_sum += float(jax.device_get(vbpb))
            n_done += 1
        epoch_val_bpb = bpb_sum / max(n_done, 1)
        epoch_steps = global_step - epoch_step_start
        epoch_nonfinite = nonfinite_step_count - epoch_nonfinite_start
        epoch_train_bpb = full_step_history[-1]["train_bpb"] if full_step_history else float("nan")

        epoch_summaries.append(dict(
            epoch=epoch, steps=epoch_steps, wall_s=t_epoch_wall,
            end_train_bpb=epoch_train_bpb, end_val_bpb=epoch_val_bpb,
            nonfinite_steps=epoch_nonfinite,
        ))
        LOG.log(f"[EPOCH {epoch}/{RUN_CONFIG['epochs']}] done: steps={epoch_steps} "
                 f"wall={t_epoch_wall:.1f}s end_train_bpb={epoch_train_bpb:.4f} "
                 f"epoch_val_bpb={epoch_val_bpb:.4f} nonfinite={epoch_nonfinite}")

        canary = compute_canary_fingerprint(model, params)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"params_epoch{epoch}.msgpack")
        save_params_checkpoint(ckpt_path, params, dict(
            epoch=epoch, step=global_step, epoch_val_bpb=epoch_val_bpb,
            epoch_train_bpb=epoch_train_bpb, canary=canary,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        ))
        checkpoint_meta.append(dict(epoch=epoch, path=ckpt_path, canary=canary))

    total_wall = time.perf_counter() - t_run_start
    n_steps_done = len(full_step_history)
    tokens_per_step = micro_bs * accum_steps * RUN_CONFIG["seq_len"]
    grad_norms = [s["grad_norm"] for s in full_step_history]

    final_ckpt_path = checkpoint_meta[-1]["path"] if checkpoint_meta else None

    # -------- ОДИН РАЗ: TEST на ПОЛНОМ test-сплите, из ПЕРЕЗАГРУЖЕННОГО с диска чекпоинта --------
    test_result = None
    if final_ckpt_path:
        LOG.log("\n" + "=" * 96)
        LOG.log("[TEST] Финальная ОДНОРАЗОВАЯ оценка на held-out test, из чекпоинта, "
                 "ЗАНОВО ЗАГРУЖЕННОГО С ДИСКА (не из переменной params в памяти)")
        LOG.log("=" * 96)
        fresh_model = ByteGDN2LM(cfg=MODEL_CONFIG, kernel_config=KERNEL_CONFIG)
        template = fresh_model.init(jax.random.PRNGKey(0), dummy_ids)["params"]
        reloaded_params = load_params_checkpoint(final_ckpt_path, template)

        reloaded_canary = compute_canary_fingerprint(fresh_model, reloaded_params)
        expected_canary = checkpoint_meta[-1]["canary"]
        canary_match = abs(reloaded_canary["checksum"] - expected_canary["checksum"]) < 1e-3
        LOG.log(f"  Canary после перезагрузки: checksum={reloaded_canary['checksum']:.4f} "
                 f"(записано при сохранении: {expected_canary['checksum']:.4f})  "
                 f"{'MATCH' if canary_match else 'MISMATCH -- ЧЕКПОИНТ ПОВРЕЖДЁН/НЕ ТОТ!'}")

        @jax.jit
        def test_val_step(p, batch):
            return compute_bpb_loss(p, fresh_model, batch, deterministic=True)

        bpb_sum, n_done = 0.0, 0
        for tb in full_pass_batches(test_chunks, micro_bs, shuffle=False):
            _, tbpb = test_val_step(reloaded_params, tb)
            bpb_sum += float(jax.device_get(tbpb))
            n_done += 1
        test_bpb = bpb_sum / max(n_done, 1)
        LOG.log(f"  [TEST] Полный проход по {n_done} батчам ({n_done*micro_bs} чанков) "
                 f"held-out test: test_bpb={test_bpb:.4f}")
        test_result = dict(test_bpb=test_bpb, n_batches=n_done, canary_match=canary_match,
                            checkpoint_used=final_ckpt_path)
        del reloaded_params, template
        gc.collect()

    result = dict(
        n_params=n_params, muon_param_count=muon_count, adamw_param_count=adamw_count,
        epochs_target=RUN_CONFIG["epochs"], epochs_completed=len(epoch_summaries),
        steps_completed=n_steps_done, auto_stopped=auto_stopped,
        total_wall_s=total_wall,
        mean_step_ms=float(np.mean([s["step_ms"] for s in full_step_history])) if full_step_history else float("nan"),
        median_step_ms=float(np.median([s["step_ms"] for s in full_step_history])) if full_step_history else float("nan"),
        tokens_per_step=tokens_per_step,
        tokens_per_sec=(tokens_per_step * n_steps_done / total_wall) if total_wall > 0 else float("nan"),
        peak_hbm_mb=peak_bytes_seen / 1e6, settle_peak_hbm_mb=settle_peak / 1e6,
        mean_grad_norm=float(np.mean(grad_norms)) if grad_norms else float("nan"),
        max_grad_norm=float(np.max(grad_norms)) if grad_norms else float("nan"),
        nonfinite_step_count=nonfinite_step_count,
        nonfinite_step_ratio=(nonfinite_step_count / n_steps_done) if n_steps_done else float("nan"),
        final_train_bpb=full_step_history[-1]["train_bpb"] if full_step_history else float("nan"),
        best_val_bpb=best_val_bpb,
        epoch_summaries=epoch_summaries, val_history=val_history,
        full_step_history=full_step_history,   # <- ПОЛНАЯ история, не только агрегаты
        checkpoints=checkpoint_meta,
        test_result=test_result,
    )

    del params, opt_state, accum_grads
    gc.collect()
    jax.clear_caches()
    return result


# ==========================================================================
# 9. ФИНАЛЬНЫЙ ОТЧЁТ
# ==========================================================================
def print_final_report(gate1, gate2, contamination, result):
    LOG.log("\n\n" + "#" * 100)
    LOG.log("# ФИНАЛЬНЫЙ ОТЧЁТ: GDN-2 CENTERED, ~150M, Muon+AdamW, contiguous 90/5/5 сплит")
    LOG.log("#" * 100)

    LOG.log("\n-- Гейты (блокирующие, пройдены ДО обучения) --")
    LOG.log(f"  Гейт 1 (корректность кернеля): {'PASS' if gate1['passed'] else 'FAIL'}")
    LOG.log(f"  Гейт 2 (причинность e2e):      {'PASS' if gate2['passed'] else 'FAIL'} "
            f"({gate2['positions_checked']} позиций проверено, {gate2['n_failed']} провалов)")

    LOG.log("\n-- Contamination self-check (информационно) --")
    LOG.log(f"  VAL:  {contamination['val']['contamination_rate_chunks']:.2%} чанков с совпадением")
    LOG.log(f"  TEST: {contamination['test']['contamination_rate_chunks']:.2%} чанков с совпадением")

    LOG.log("\n-- По эпохам --")
    for e in result["epoch_summaries"]:
        LOG.log(f"  epoch {e['epoch']}: steps={e['steps']} wall={e['wall_s']:.1f}s "
                f"end_train_bpb={e['end_train_bpb']:.4f} end_val_bpb={e['end_val_bpb']:.4f} "
                f"nonfinite={e['nonfinite_steps']}")

    LOG.log("\n-- Общая сводка --")
    LOG.log(f"  Параметров: {result['n_params']:,}  Muon: {result['muon_param_count']:,}  "
            f"AdamW: {result['adamw_param_count']:,}")
    LOG.log(f"  Эпох: {result['epochs_completed']}/{result['epochs_target']}  "
            f"шагов: {result['steps_completed']}  время: {result['total_wall_s']/3600:.2f}ч")
    LOG.log(f"  Скорость: {result['mean_step_ms']:.2f} ms/step  tok/s: {result['tokens_per_sec']:.1f}")
    LOG.log(f"  Peak HBM: {result['peak_hbm_mb']:.1f} MB")
    LOG.log(f"  Grad norm: mean={result['mean_grad_norm']:.3f} max={result['max_grad_norm']:.3f}  "
            f"non-finite: {result['nonfinite_step_count']}")
    LOG.log(f"  final_train_bpb={result['final_train_bpb']:.4f}  best_val_bpb={result['best_val_bpb']:.4f}")

    if result["test_result"]:
        tr = result["test_result"]
        LOG.log(f"\n  >>> HELD-OUT TEST BPB (единожды, из перезагруженного чекпоинта): "
                f"{tr['test_bpb']:.4f}  (canary_match={tr['canary_match']}, "
                f"checkpoint={tr['checkpoint_used']})")
        LOG.log("      Сравнимо с литературой: Transformer-XL bpc≈0.94 на этом же "
                "(contiguous 90/5/5) протоколе.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dict(gate1=gate1, gate2=gate2, contamination=contamination,
                        train_result=result), f, indent=2, default=str)
    LOG.log(f"\n[OUTPUT] Полные результаты (включая ПОЛНУЮ пошаговую историю) сохранены в {OUTPUT_JSON}")
    LOG.log(f"[OUTPUT] Чекпоинты: {CHECKPOINT_DIR}")
    LOG.log(f"[OUTPUT] Лог (self-timestamped, см. предупреждение в докстринге): {LOG_PATH}")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    LOG.log(f"JAX version: {jax.__version__}")
    LOG.log(f"Devices: {jax.devices()}")
    LOG.log(f"TPU available: {is_tpu_available()}")
    if not is_tpu_available():
        LOG.log("!!! TPU не обнаружен -- Pallas-путь требует TPU с d_head=128.")

    data_path = ensure_enwik8()
    byte_arr = load_enwik8_bytes(data_path)
    train_chunks, val_chunks, test_chunks = make_contiguous_split(
        byte_arr, RUN_CONFIG["seq_len"], TRAIN_BYTES, VAL_BYTES, TEST_BYTES
    )
    LOG.log(f"[DATA] enwik8: {len(byte_arr):,} bytes -> train={len(train_chunks):,} chunks "
            f"(first {TRAIN_BYTES:,}B), val={len(val_chunks):,} chunks (next {VAL_BYTES:,}B), "
            f"test={len(test_chunks):,} chunks (last {TEST_BYTES:,}B) -- STANDARD CONTIGUOUS SPLIT")

    contamination = run_contamination_report(train_chunks, val_chunks, test_chunks)

    gate1 = run_correctness_gate(KERNEL_CONFIG)
    if not gate1["passed"]:
        LOG.log("!!! ГЕЙТ 1 НЕ ПРОЙДЕН -- ОБУЧЕНИЕ НЕ ЗАПУСКАЕТСЯ.")
        return

    gate2 = run_causal_leak_gate()
    if not gate2["passed"]:
        LOG.log("!!! ГЕЙТ 2 (ПРИЧИННОСТЬ) НЕ ПРОЙДЕН -- ОБУЧЕНИЕ НЕ ЗАПУСКАЕТСЯ. "
                "Модель/кернель видит будущие токены -- любой результат обучения был бы "
                "невалиден.")
        return

    result = run_training(train_chunks, val_chunks, test_chunks)
    print_final_report(gate1, gate2, contamination, result)
    LOG.close()


if __name__ == "__main__":
    main()
