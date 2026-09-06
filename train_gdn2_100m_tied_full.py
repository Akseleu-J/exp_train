"""
train_gdn2_100m_tied_full.py -- СТАДИЯ 2 (главная цель): полностью
weight-tied GDN-2-модель, ПЕРЕИСПОЛЬЗУЮЩАЯ 100% из 12 исходных слоёв
untied-модели (стадия 1, train_gdn2_100m.py) как ОДИН общий блок
(layers_per_block=12), применённый n_reasoning_cycles раз подряд.

Это НЕ то же самое, что H2-sweep (train_gdn2_100m_h2_sweep.py) или
latent_reasoning_gdn2.py:
  - H2-sweep обучает tied-блок С НУЛЯ и tied-блок содержит только
    layers_per_block=3 слоя (те же 3, что были бы в ОДНОМ untied-блоке).
  - latent_reasoning_gdn2.py warm-старт берёт только ПОСЛЕДНИЙ блок
    (block_7 / 3 слоя) исходной 0.7B модели.
  - ЗДЕСЬ: warm-старт объединяет ВСЕ 4 untied-блока (12 слоёв) стадии 1 в
    ОДИН shared_block (layer_0 .. layer_11 внутри одного module-scope) --
    100% исходных весов становятся общими и переиспользуются на каждом
    из n_reasoning_cycles проходов. Идея = "recurrent depth" / GPT-6 Astra:
    тот же compute-per-application, но эффективная глубина = num_layers *
    n_reasoning_cycles ценой количества уникальных параметров = 1 блок.

ИСТОЧНИК ВЕСОВ: HF-слот "gdn2_100m_untied/best_val", который пишет
train_gdn2_100m.py (см. HF_ROOT там). Если best_val пуст (best_val_bpb
никогда не улучшился за стадию 1) -- падаем обратно на "latest".
"""
from __future__ import annotations

import os
import time
import json
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from model_gdn2_100m import set_model_mesh, count_params, GDN2_PALLAS_BT
from model_gdn2_100m_tied import FullGDN2BlockDeltaModelTied, TiedModelConfig
from train_gdn2_100m import (
    load_config as load_stage1_config,
    load_enwik8_bytes, make_chunked_dataset, batch_iterator, make_tpu_mesh,
    HF_ROOT as STAGE1_HF_ROOT,
)
from checkpointing import make_manager, save_slot, upload_slot, download_slot, _HAS_HF

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# ==========================================================================
# RUN_CONFIG -- стадия 2 (tied, все 12 слоёв, recurrent-depth).
# ==========================================================================
RUN_CONFIG_TIED = dict(
    run_name=f"gdn2_100m_tied_full_{time.strftime('%Y%m%d_%H%M%S')}",
    seq_len=2048,
    micro_batch_size=8,
    accum_steps=4,
    total_train_steps=1500,
    warmup_steps=600,          # длиннее, чем stage1 -- weight-tying даёт более резкий градиентный ландшафт
    peak_lr=5e-5,              # ниже пика stage1 (2e-4) -- см. train_gdn2_100m_h2_sweep.py's TIED_LR_SCALE
    min_lr=1e-6,
    lr_reduce_factor=0.5,
    patience=2,
    lr_cooldown_steps=400,
    weight_decay=0.01,
    grad_clip_norm=0.3,        # ниже stage1 -- доп. подстраховка для tied-градиента
    eval_every_steps=150,
    eval_batches=20,
    eval_seed=999,
    ckpt_every_seconds=600,
    nonfinite_consecutive_limit=4,
    nonfinite_window_size=15,
    nonfinite_window_ratio=0.25,
    data_path="/kaggle/input/datasets/nightfury1103/enwik8/enwik8",
    val_split=0.02,
    seed=43,
)

# Сколько раз ПОВТОРНО применяется общий блок (все 12 слоёв) -- это и есть
# "лишняя" recurrent-глубина. 4 даёт тот же ЧИСЛО-forward-применений GDN2,
# что и вся стадия-1 модель (12 layers = 4 blocks * 3), т.е. compute-match
# по всем 12 слоям сразу, а не по одному блоку -- поднимите, если хотите
# намеренно ПРЕВЫСИТЬ compute стадии 1 ради дополнительной эффективной
# глубины (собственно Astra-идея).
N_REASONING_CYCLES = 4

# ==========================================================================
# ПАТЧ (deep supervision + smoothness): чистый "как у Astra" вариант --
# ТОЛЬКО final-loss, без явного per-цикл сигнала -- уже был опробован
# (300 шагов) и провалился: h1_probe_gdn2_100m_tied.py показал bpb,
# МОНОТОННО РАСТУЩИЙ с r (2.80->4.17) и Phi_conv~=0.0000 на каждом шаге
# (соседние состояния практически не коррелируют -- это drift/collapse
# необученного механизма, ровно baseline из RESEARCH_LOG §3.5, а НЕ
# emergent-прогрессия). При 100M/300-1500 шагов implicit-emergence в духе
# Astra не набирает нужный бюджет -- нужен явный сигнал.
#
# Явного gold-step датасета у нас нет (byte-level enwik8, не CoT) --
# поэтому вместо LOTUS gold-step берём deep supervision: ТОТ ЖЕ финальный
# CE-таргет применяется к КАЖДОМУ промежуточному y_r (не только к y_R),
# с весом, РАСТУЩИМ по r (не equal-weight!) -- equal-weight parallel
# supervision на игрушке уже дала другой, но тоже нежелательный эффект
# (§2.2: "реши сразу", accuracy плоская по глубине) -- растущий вес
# минимизирует этот риск, отдавая приоритет последнему циклу.
#
# Плюс smoothness-штраф на ||y_r - y_{r-1}||^2 -- напрямую бьёт по
# наблюдаемому Phi_conv~=0 (взрывной, не плавный переход между циклами).
# ==========================================================================
DEEP_SUPERVISION_ENABLED = True
DEEP_SUPERVISION_WEIGHT_MODE = "linear_increasing"  # веса r/R, r=1..R
SMOOTH_PENALTY_COEF = 1e-3   # маленький -- ведущий сигнал всё ещё CE, это только anti-explosion подстраховка
READOUT_CLIP = 30.0

CKPT_ROOT = "/kaggle/working/gdn2_100m_tied_full_ckpt"
HF_ROOT = "gdn2_100m_tied_full"
HF_LATEST_KEEP_N = 1

STAGE1_LOCAL_DOWNLOAD_DIR = "/kaggle/working/gdn2_100m_untied_downloaded"


def load_tied_config():
    base = load_stage1_config()
    return TiedModelConfig(
        d_model=base.d_model,
        n_heads=base.n_heads,
        d_latent=base.d_latent,
        num_layers=base.num_layers,     # 12 -- используется ТОЛЬКО для внутренней структуры блока
        layers_per_block=base.num_layers,  # ПАТЧ: = num_layers -> ОДИН блок содержит ВСЕ 12 слоёв
        vocab_size=base.vocab_size,
        tie_embeddings=base.tie_embeddings,
        label_smoothing=base.label_smoothing,
        d_conv=base.d_conv,
        tie_all_blocks=True,
        n_reasoning_cycles=N_REASONING_CYCLES,
    )


def _merge_all_untied_blocks_into_shared_block(stage1_params, num_layers, layers_per_block_stage1):
    """ПАТЧ (ключевая функция): собирает ВСЕ layer_i поддеревья из ВСЕХ
    block_j стадии 1 в один плоский dict -- это и есть 100%-переиспользуемый
    'shared_block' стадии 2. Работает напрямую благодаря тому, что
    BlockDARLayer именует себя ГЛОБАЛЬНЫМ layer_idx (f"layer_{layer_idx}"),
    а не локальным индексом внутри блока -- поэтому объединение НЕ требует
    переименования ключей."""
    num_blocks = num_layers // layers_per_block_stage1
    merged = {}
    for block_idx in range(num_blocks):
        block_key = f"block_{block_idx}"
        if block_key not in stage1_params:
            raise KeyError(
                f"Не нашёл {block_key} в восстановленных params стадии 1 -- "
                f"проверьте, что num_layers/layers_per_block совпадают с тем, "
                f"под чем чекпоинт реально обучался."
            )
        for k, v in stage1_params[block_key].items():
            if k.startswith("layer_"):
                merged[k] = v
    if not merged:
        raise RuntimeError("Слияние блоков стадии 1 дало пустой shared_block -- проверьте структуру чекпоинта.")
    return merged


def _try_copy_matching_shapes(target, source):
    if isinstance(target, dict):
        return {k: (_try_copy_matching_shapes(v, source[k]) if isinstance(source, dict) and k in source else v)
                for k, v in target.items()}
    if hasattr(target, "shape") and hasattr(source, "shape") and tuple(target.shape) == tuple(source.shape):
        return jnp.asarray(source, dtype=target.dtype)
    return target


def _readout_logits(y_r, embed_table):
    """Логиты промежуточного/финального состояния через ту же embed_table
    (tie_embeddings=True), БЕЗ дополнительной model-level RMSNorm -- та же
    схема, что h1_probe_gdn2_100m_tied.py уже использовал для измерения
    (числа получились содержательные: bpb 2.8..4.2), так что deep-
    supervision оптимизирует ТУ ЖЕ величину, что мы уже умеем мерить."""
    logits = jnp.einsum("bld,vd->blv", y_r.astype(jnp.float32), embed_table.astype(jnp.float32))
    return jnp.nan_to_num(jnp.clip(logits, -READOUT_CLIP, READOUT_CLIP),
                           nan=0.0, posinf=READOUT_CLIP, neginf=-READOUT_CLIP)


def deep_supervision_bpb_loss(params, model, batch, deterministic, rngs=None):
    """ПАТЧ: заменяет чистый final-only compute_bpb_loss для стадии 2.
    Возвращает (total_loss, bpb_final, ce_weighted, smooth_pen) --
    total_loss = то, что дифференцируется; bpb_final = bpb ПОСЛЕДНЕГО
    цикла (r=R) -- это реальная метрика качества, которую мы логируем и
    по которой решаем plateau/best_val (то, что модель реально отдаёт)."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    kwargs = {"deterministic": deterministic, "return_all_r_states": True}
    if rngs is not None:
        kwargs["rngs"] = rngs

    r_states = model.apply({"params": params}, input_ids, **kwargs)
    R = len(r_states)
    embed_table = params["embed"]["embedding"]

    ce_per_r = []
    for y_r in r_states:
        logits = _readout_logits(y_r, embed_table)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        ce_per_r.append(jnp.mean(nll))
    ce_stack = jnp.stack(ce_per_r)  # (R,)

    if DEEP_SUPERVISION_WEIGHT_MODE == "linear_increasing":
        weights = jnp.arange(1, R + 1, dtype=jnp.float32)
    else:  # "equal" -- НЕ рекомендуется, см. §2.2 в докстринге выше, оставлено для A/B
        weights = jnp.ones((R,), dtype=jnp.float32)
    weights = weights / jnp.sum(weights)

    ce_weighted = jnp.sum(ce_stack * weights)
    ce_weighted = jnp.nan_to_num(ce_weighted, nan=0.0, posinf=20.0, neginf=0.0)

    smooth_terms = []
    for i in range(1, R):
        delta = r_states[i].astype(jnp.float32) - r_states[i - 1].astype(jnp.float32)
        smooth_terms.append(jnp.mean(jnp.sum(delta * delta, axis=-1)))
    smooth_pen = jnp.mean(jnp.stack(smooth_terms)) if smooth_terms else jnp.asarray(0.0, dtype=jnp.float32)
    smooth_pen = jnp.nan_to_num(smooth_pen, nan=0.0, posinf=1e6, neginf=0.0)

    total_loss = ce_weighted + SMOOTH_PENALTY_COEF * smooth_pen
    bpb_final = ce_stack[-1] / jnp.log(2.0)
    return total_loss, bpb_final, ce_weighted, smooth_pen


def fetch_stage1_params(mesh, micro_batch_size, seq_len):
    """Качает лучший (или, если его нет, последний) чекпоинт стадии 1 с HF,
    восстанавливает его под АБСТРАКТНОЙ формой untied-модели, затем
    объединяет все 4 блока в один shared_block для tied-модели."""
    if not _HAS_HF:
        raise RuntimeError(
            "HF недоступен (_HAS_HF=False в checkpointing.py) -- стадия 2 не может "
            "скачать веса стадии 1. Проверьте HF_TOKEN/HF_REPO_ID в Kaggle Secrets."
        )

    os.makedirs(STAGE1_LOCAL_DOWNLOAD_DIR, exist_ok=True)
    step = download_slot(STAGE1_LOCAL_DOWNLOAD_DIR, f"{STAGE1_HF_ROOT}/best_val")
    used_slot = "best_val"
    if step is None:
        print(f"[FETCH] ⚠️ '{STAGE1_HF_ROOT}/best_val' пуст или недоступен -- пробую 'latest'.")
        step = download_slot(STAGE1_LOCAL_DOWNLOAD_DIR, f"{STAGE1_HF_ROOT}/latest")
        used_slot = "latest"
    if step is None:
        raise RuntimeError(
            f"Не удалось скачать чекпоинт стадии 1 ни из '{STAGE1_HF_ROOT}/best_val', "
            f"ни из '{STAGE1_HF_ROOT}/latest'. Запустите train_gdn2_100m.py первым."
        )
    print(f"[FETCH] ✅ Стадия 1: слот='{used_slot}', шаг={step}")

    stage1_cfg = load_stage1_config()

    # ПАТЧ (фикс): save_slot() пишет чекпоинт как {"params": ..., "opt_state": ...},
    # а restore() раньше запрашивался только с ключом "params" -- orbax сверяет
    # структуру restore-запроса со структурой НА ДИСКЕ и падает с
    # "Source: MISSING" на opt_state, т.к. запрошенное дерево уже дереву на
    # диске не соответствует (opt_state отсутствует в запросе). Нам не нужен
    # opt_state стадии 1 вообще (стадия 2 инициализирует свой оптимизатор
    # с нуля) -- проще всего восстановить БЕЗ явной target-структуры (orbax
    # сам восстановит то, что реально лежит на диске, params+opt_state), а
    # затем взять из результата только "params".
    mngr = ocp.CheckpointManager(STAGE1_LOCAL_DOWNLOAD_DIR, ocp.StandardCheckpointer())
    raw = mngr.restore(step)
    stage1_params = raw["params"]

    replicate = NamedSharding(mesh, P())
    stage1_params = jax.tree_util.tree_map(lambda p: jax.device_put(p, replicate), stage1_params)

    merged_shared_block = _merge_all_untied_blocks_into_shared_block(
        stage1_params, stage1_cfg.num_layers, stage1_cfg.layers_per_block
    )
    embed_params = stage1_params["embed"]
    final_norm_params = stage1_params.get("final_norm", None)
    return merged_shared_block, embed_params, final_norm_params


def _hf_checkpoint_slot(mngr, local_dir, hf_subdir, step, params, opt_state,
                         best_val_loss, best_train_loss, train_loss=None, keep_last_n=1, tag=""):
    try:
        save_slot(mngr, local_dir, step, params, opt_state, epoch=0,
                   best_val_loss=best_val_loss, best_train_loss=best_train_loss,
                   train_loss=train_loss)
    except Exception as e:
        print(f"[CKPT] ⚠️ Локальный save_slot ({tag}) не удался на шаге {step}: {e}")
        return
    if not _HAS_HF:
        return
    try:
        upload_slot(local_dir, hf_subdir, step, msg=tag, keep_last_n=keep_last_n)
    except Exception as e:
        print(f"[HF] ⚠️ upload_slot ({tag}) не удался на шаге {step}: {e}")


def make_plateau_schedule(peak_lr, warmup_steps):
    def schedule(step, multiplier):
        step = jnp.asarray(step, dtype=jnp.float32)
        multiplier = jnp.asarray(multiplier, dtype=jnp.float32)
        warmup_frac = jnp.clip(step / max(warmup_steps, 1), 0.0, 1.0)
        warmup_lr = peak_lr * warmup_frac
        stable_lr = peak_lr * multiplier
        return jnp.where(step < warmup_steps, warmup_lr, stable_lr)
    return schedule


def main():
    cfg_run = RUN_CONFIG_TIED
    os.makedirs(CKPT_ROOT, exist_ok=True)

    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]
    set_model_mesh(mesh, batch_axis="tpu_nodes")
    print(f"[TPU] Устройств: {n_devices}")

    if cfg_run["seq_len"] % GDN2_PALLAS_BT != 0:
        raise ValueError(f"seq_len={cfg_run['seq_len']} должен делиться на GDN2_PALLAS_BT={GDN2_PALLAS_BT}")
    if cfg_run["micro_batch_size"] % n_devices != 0:
        raise ValueError("micro_batch_size должен делиться на число устройств.")

    model_cfg = load_tied_config()
    model = FullGDN2BlockDeltaModelTied(cfg=model_cfg)
    print(f"[MODEL] СТАДИЯ 2: tie_all_blocks=True, layers_per_block={model_cfg.layers_per_block} "
          f"(=100% исходных слоёв), n_reasoning_cycles={model_cfg.n_reasoning_cycles}")

    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))
    scalar_sharding = NamedSharding(mesh, P())

    init_rng = jax.random.PRNGKey(cfg_run["seed"])
    abstract_params = jax.eval_shape(
        lambda: model.init(init_rng, jnp.zeros((cfg_run["micro_batch_size"], cfg_run["seq_len"]), dtype=jnp.int32))
    )["params"]

    def _shard_leaf(path, param):
        if not hasattr(param, "shape") or param.ndim == 0:
            return NamedSharding(mesh, P())
        for i, size in enumerate(param.shape):
            if size % n_devices == 0 and (size // n_devices) >= 64:
                spec = [None] * param.ndim
                spec[i] = "tpu_nodes"
                return NamedSharding(mesh, P(*spec))
        return NamedSharding(mesh, P(*([None] * param.ndim)))

    param_sharding = jax.tree_util.tree_map_with_path(_shard_leaf, abstract_params)

    init_fn = jax.jit(
        lambda rng: model.init(rng, jnp.zeros((cfg_run["micro_batch_size"], cfg_run["seq_len"]), dtype=jnp.int32))["params"],
        out_shardings=param_sharding,
    )
    params = init_fn(init_rng)

    # ---- ПАТЧ: warm-start shared_block/embed из стадии 1 (100% слоёв) ----
    merged_shared_block, stage1_embed, stage1_final_norm = fetch_stage1_params(
        mesh, cfg_run["micro_batch_size"], cfg_run["seq_len"]
    )
    params = dict(params)
    params["shared_block"] = _try_copy_matching_shapes(params["shared_block"], merged_shared_block)
    params["embed"] = _try_copy_matching_shapes(params["embed"], stage1_embed)
    if stage1_final_norm is not None and "final_norm" in params:
        params["final_norm"] = _try_copy_matching_shapes(params["final_norm"], stage1_final_norm)
    params = jax.tree_util.tree_map_with_path(
        lambda path, p: jax.device_put(p, _shard_leaf(path, p)) if hasattr(p, "shape") else p, params
    )
    print("[WARM-START] ✅ shared_block инициализирован ВСЕМИ 12 слоями стадии 1, embed скопирован.")

    n_params = count_params(params)
    print(f"[MODEL] Уникальных параметров стадии 2: {n_params:,} (≈ {n_params/1e6:.1f}M) -- "
          f"эффективная глубина = num_layers * n_reasoning_cycles = "
          f"{model_cfg.num_layers} * {model_cfg.n_reasoning_cycles} слой-применений на forward.")

    lr_schedule = make_plateau_schedule(cfg_run["peak_lr"], cfg_run["warmup_steps"])

    def make_tx(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(cfg_run["grad_clip_norm"]),
            optax.adamw(learning_rate=learning_rate, weight_decay=cfg_run["weight_decay"]),
        )

    tx = optax.inject_hyperparams(make_tx)(learning_rate=cfg_run["peak_lr"])
    opt_state_abstract = jax.eval_shape(tx.init, abstract_params)
    opt_state_sharding = jax.tree_util.tree_map_with_path(_shard_leaf, opt_state_abstract)
    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    current_lr_multiplier = 1.0
    evals_since_improvement = 0
    cooldown_counter = 0
    total_reduces_done = 0

    byte_arr = load_enwik8_bytes(cfg_run["data_path"])
    trimmed, train_idx, val_idx = make_chunked_dataset(
        byte_arr, cfg_run["seq_len"], cfg_run["val_split"], cfg_run["seed"]
    )
    train_stream = batch_iterator(
        trimmed, train_idx, cfg_run["micro_batch_size"], data_sharding,
        seed=cfg_run["seed"], shuffle=True,
    )

    def val_stream_factory():
        return batch_iterator(
            trimmed, val_idx, cfg_run["micro_batch_size"], data_sharding,
            seed=cfg_run["eval_seed"], shuffle=True,
        )

    accum_steps = cfg_run["accum_steps"]

    def train_micro_step(p, accum_grads, batch, rng):
        def loss_fn(param):
            total_loss, bpb_final, ce_weighted, smooth_pen = deep_supervision_bpb_loss(
                param, model, batch, deterministic=False, rngs={"dropout": rng}
            )
            return total_loss, (bpb_final, ce_weighted, smooth_pen)
        (total_loss, (bpb_final, ce_weighted, smooth_pen)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        micro_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return new_accum, bpb_final, smooth_pen, micro_grad_norm

    compiled_train_micro = jax.jit(
        train_micro_step,
        donate_argnums=(1,),
        in_shardings=(param_sharding, param_sharding,
                       {"input_ids": data_sharding, "labels": data_sharding},
                       NamedSharding(mesh, P(None))),
        out_shardings=(param_sharding, NamedSharding(mesh, P()),
                        NamedSharding(mesh, P()), NamedSharding(mesh, P())),
    )

    def apply_step(p, s, accum_grads, n_accum, lr):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)
        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))
        is_finite = jnp.isfinite(global_norm)
        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), avg_grads
        )
        s = s._replace(hyperparams=dict(s.hyperparams, learning_rate=lr))
        updates, new_s = tx.update(avg_grads, s, p)
        new_p_candidate = optax.apply_updates(p, updates)
        new_p_candidate = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p_candidate,
        )
        new_p = jax.tree_util.tree_map(lambda old, new: jnp.where(is_finite, new, old), p, new_p_candidate)
        new_s = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_finite, new, old) if hasattr(new, "shape") else new, s, new_s,
        )
        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return new_p, new_s, zero_accum, is_finite, global_norm

    compiled_apply = jax.jit(
        apply_step,
        donate_argnums=(0, 1, 2),
        in_shardings=(param_sharding, opt_state_sharding, param_sharding,
                       NamedSharding(mesh, P()), scalar_sharding),
        out_shardings=(param_sharding, opt_state_sharding, param_sharding,
                        NamedSharding(mesh, P()), NamedSharding(mesh, P())),
    )

    def val_step(p, batch):
        # ФИКС: eval теперь считает bpb ТОЙ ЖЕ функцией, что training (deep
        # supervision readout, r=R) -- иначе plateau/best_val сравнивал бы
        # величину, отличную от той, что реально оптимизируется.
        _total_loss, bpb_final, _ce_w, _smooth = deep_supervision_bpb_loss(
            p, model, batch, deterministic=True
        )
        return bpb_final

    compiled_val = jax.jit(
        val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )

    mngr = ocp.CheckpointManager(
        os.path.join(CKPT_ROOT, "latest"),
        ocp.StandardCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=2, create=True, enable_async_checkpointing=False),
    )
    hf_latest_dir = os.path.join(CKPT_ROOT, "hf_latest_slot")
    hf_best_dir = os.path.join(CKPT_ROOT, "hf_best_val_slot")
    mngr_hf_latest = make_manager(hf_latest_dir, max_to_keep=2)
    mngr_hf_best = make_manager(hf_best_dir, max_to_keep=1)
    if _HAS_HF:
        print(f"[HF] Стадия 2 будет писать в '{HF_ROOT}/latest' и '{HF_ROOT}/best_val'.")

    zero_accum = jax.jit(lambda p: jax.tree_util.tree_map(jnp.zeros_like, p), out_shardings=param_sharding)(params)
    accum_grads = zero_accum

    global_rng = jax.random.PRNGKey(cfg_run["seed"] + 1)
    global_step = 0
    nonfinite_consecutive = 0
    nonfinite_window = deque(maxlen=cfg_run["nonfinite_window_size"])
    best_val_bpb = float("inf")
    last_ckpt_time = time.perf_counter()
    _micro_bpb_acc, _micro_smooth_acc, _micro_grad_norm_acc = [], [], []

    print(f"[TRAIN] 🚀 Старт СТАДИИ 2 (tied, 100% слоёв, R={model_cfg.n_reasoning_cycles}, "
          f"deep_supervision={DEEP_SUPERVISION_ENABLED}, weight_mode={DEEP_SUPERVISION_WEIGHT_MODE}, "
          f"smooth_coef={SMOOTH_PENALTY_COEF}).")

    micro_step = 0
    while global_step < cfg_run["total_train_steps"]:
        batch = next(train_stream)
        global_rng, step_rng = jax.random.split(global_rng)

        accum_grads, bpb, smooth_pen, micro_grad_norm = compiled_train_micro(params, accum_grads, batch, step_rng)
        _micro_bpb_acc.append(float(jax.device_get(bpb)))
        _micro_smooth_acc.append(float(jax.device_get(smooth_pen)))
        _micro_grad_norm_acc.append(float(jax.device_get(micro_grad_norm)))
        micro_step += 1

        if micro_step % accum_steps == 0:
            next_step_for_lr = global_step + 1
            cur_lr = float(jax.device_get(lr_schedule(next_step_for_lr, current_lr_multiplier)))
            lr_arr = jax.device_put(jnp.asarray(cur_lr, dtype=jnp.float32), scalar_sharding)

            params, opt_state, accum_grads, is_finite, global_norm = compiled_apply(
                params, opt_state, accum_grads, accum_steps, lr_arr
            )
            step_finite = bool(jax.device_get(is_finite))
            global_step += 1

            if cooldown_counter > 0:
                cooldown_counter -= 1

            nonfinite_window.append(0 if step_finite else 1)
            nonfinite_consecutive = 0 if step_finite else nonfinite_consecutive + 1
            window_ratio = sum(nonfinite_window) / len(nonfinite_window)

            mean_bpb = float(np.mean(_micro_bpb_acc))
            mean_smooth = float(np.mean(_micro_smooth_acc))
            _micro_bpb_acc, _micro_smooth_acc, _micro_grad_norm_acc = [], [], []

            print(f"[STAGE2 STEP {global_step}/{cfg_run['total_train_steps']}] "
                  f"bpb(r=R)={mean_bpb:.4f} smooth_pen={mean_smooth:.4f} lr={cur_lr:.2e} "
                  f"global_grad_norm={float(jax.device_get(global_norm)):.4f} "
                  f"finite={step_finite} nonfinite_window={window_ratio:.2%} "
                  f"lr_mult={current_lr_multiplier:.4f} cooldown={cooldown_counter}")

            if not step_finite:
                print(f"[WARNING] ⚠️ Non-finite градиент на шаге {global_step} -- обновление ПРОПУЩЕНО.")

            hit_consecutive = nonfinite_consecutive >= cfg_run["nonfinite_consecutive_limit"]
            hit_window = (len(nonfinite_window) >= cfg_run["nonfinite_window_size"]
                          and window_ratio >= cfg_run["nonfinite_window_ratio"])
            if hit_consecutive or hit_window:
                print(f"🛑 [AUTO-STOP] СТАДИЯ 2: нестабильность на шаге {global_step}. Сохраняю и останавливаюсь.")
                mngr.save(global_step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
                mngr.wait_until_finished()
                _hf_checkpoint_slot(
                    mngr_hf_latest, hf_latest_dir, f"{HF_ROOT}/latest", global_step,
                    params, opt_state, best_val_bpb, best_val_bpb,
                    keep_last_n=HF_LATEST_KEEP_N, tag="auto_stop",
                )
                break

            if global_step % cfg_run["eval_every_steps"] == 0:
                vstream = val_stream_factory()
                bpb_sum, n_done = 0.0, 0
                for _ in range(cfg_run["eval_batches"]):
                    vb = next(vstream)
                    vbpb = compiled_val(params, vb)
                    bpb_sum += float(jax.device_get(vbpb))
                    n_done += 1
                val_bpb = bpb_sum / max(n_done, 1)

                improved = val_bpb < best_val_bpb
                if improved:
                    best_val_bpb = val_bpb
                    evals_since_improvement = 0
                    print(f"[EVAL] Step {global_step}: val_bpb={val_bpb:.4f} 🟢 УЛУЧШЕНИЕ (best={best_val_bpb:.4f})")
                    _hf_checkpoint_slot(
                        mngr_hf_best, hf_best_dir, f"{HF_ROOT}/best_val", global_step,
                        params, opt_state, best_val_bpb, best_val_bpb,
                        train_loss=mean_bpb, keep_last_n=1, tag=f"val_bpb={val_bpb:.4f}",
                    )
                else:
                    evals_since_improvement += 1
                    print(f"[EVAL] Step {global_step}: val_bpb={val_bpb:.4f} 🔴 Нет улучшения "
                          f"({evals_since_improvement}/{cfg_run['patience']}) best={best_val_bpb:.4f}")

                if cooldown_counter == 0 and evals_since_improvement >= cfg_run["patience"]:
                    new_multiplier = current_lr_multiplier * cfg_run["lr_reduce_factor"]
                    if new_multiplier * cfg_run["peak_lr"] >= cfg_run["min_lr"]:
                        current_lr_multiplier = new_multiplier
                        cooldown_counter = cfg_run["lr_cooldown_steps"]
                        total_reduces_done += 1
                        print(f"[PLATEAU] ⬇️ LR reduce #{total_reduces_done}: multiplier={current_lr_multiplier:.4f}")
                    else:
                        print(f"[PLATEAU] ⛔ Достигнут min_lr, снижение заблокировано.")

            now = time.perf_counter()
            if now - last_ckpt_time >= cfg_run["ckpt_every_seconds"]:
                mngr.save(global_step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
                mngr.wait_until_finished()
                with open(os.path.join(CKPT_ROOT, "meta.json"), "w") as f:
                    json.dump({
                        "global_step": global_step, "best_val_bpb": best_val_bpb,
                        "lr_multiplier": current_lr_multiplier, "total_lr_reduces": total_reduces_done,
                    }, f)
                _hf_checkpoint_slot(
                    mngr_hf_latest, hf_latest_dir, f"{HF_ROOT}/latest", global_step,
                    params, opt_state, best_val_bpb, best_val_bpb,
                    train_loss=mean_bpb, keep_last_n=HF_LATEST_KEEP_N, tag="periodic",
                )
                print(f"[CKPT] Стадия 2 сохранена на шаге {global_step}")
                last_ckpt_time = now

    print(f"[DONE] СТАДИЯ 2 завершена на шаге {global_step}. best_val_bpb={best_val_bpb:.4f}")
    _hf_checkpoint_slot(
        mngr_hf_latest, hf_latest_dir, f"{HF_ROOT}/latest", global_step,
        params, opt_state, best_val_bpb, best_val_bpb,
        keep_last_n=HF_LATEST_KEEP_N, tag="final",
    )
    return {"global_step": global_step, "best_val_bpb": best_val_bpb}


if __name__ == "__main__":
    main()
