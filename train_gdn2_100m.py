"""
train_gdn2_100m.py -- обучение FullGDN2BlockDeltaModel (~100M, byte-level,
enwik8/BPB). Plateau-adaptive LR schedule: warmup -> peak -> reduce on plateau.
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

from model_gdn2_100m import (
    FullGDN2BlockDeltaModel, ModelConfig, set_model_mesh, count_params,
    GDN2_PALLAS_BT,
)
from utils import path_to_str
from diagnostics import (
    make_leaf_layer_map, param_layer_tags, build_leaf_stats_fn, build_leaf_raw_stats_fn,
)

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# ==========================================================================
# W&B -- тот же паттерн, что wandb_logging.py: молча отключается, если
# секретов нет, никогда не роняет обучение.
# ==========================================================================
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    import wandb
    _WANDB_KEY = _secrets.get_secret("WANDB_API_KEY")
    _HAS_WANDB = bool(_WANDB_KEY)
    if _HAS_WANDB:
        wandb.login(key=_WANDB_KEY)
        print("[WANDB] ✅ Ключ найден, логирование включено.")
except Exception as e:
    _HAS_WANDB = False
    print(f"[WANDB] ⚠️ Недоступен ({type(e).__name__}: {e}) -- продолжаю без W&B.")

_run = None


def wandb_log(step, metrics):
    if not _HAS_WANDB or _run is None:
        return
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        print(f"[WANDB] ⚠️ log failed на шаге {step}: {e}")


# ==========================================================================
# RUN_CONFIG -- редактировать здесь, никакого argparse (Kaggle sys.argv).
# ==========================================================================
RUN_CONFIG = dict(
    run_name=f"gdn2_100m_enwik8_{time.strftime('%Y%m%d_%H%M%S')}",
    seq_len=2048,
    micro_batch_size=8,
    accum_steps=4,
    total_train_steps=4500,
    warmup_steps=300,
    peak_lr=2e-4,
    min_lr=1e-6,
    lr_reduce_factor=0.5,
    patience=2,
    lr_cooldown_steps=400,
    weight_decay=0.01,
    grad_clip_norm=1.0,
    eval_every_steps=150,
    eval_batches=20,
    eval_seed=999,
    ckpt_every_seconds=600,
    nonfinite_consecutive_limit=4,
    nonfinite_window_size=15,
    nonfinite_window_ratio=0.25,
    data_path="/kaggle/input/datasets/nightfury1103/enwik8/enwik8",
    val_split=0.02,
    seed=42,
)

CKPT_ROOT = "/kaggle/working/gdn2_100m_ckpt"


def load_config():
    cfg = ModelConfig(
        d_model=1024,
        n_heads=8,
        d_latent=512,
        num_layers=12,
        layers_per_block=3,
        vocab_size=256,
        tie_embeddings=True,
        label_smoothing=0.0,
        d_conv=4,
    )
    return cfg


# ==========================================================================
# Данные: enwik8 как сырые байты.
# ==========================================================================
def load_enwik8_bytes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Не найден {path}. Скачайте enwik8 (обычно как Kaggle dataset) "
            f"и поправьте RUN_CONFIG['data_path']."
        )
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    print(f"[DATA] enwik8: {len(arr):,} байт")
    return arr


def make_chunked_dataset(byte_arr, seq_len, val_split, seed):
    n_total = len(byte_arr) // (seq_len + 1)
    usable = n_total * (seq_len + 1)
    trimmed = byte_arr[:usable].reshape(n_total, seq_len + 1)
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_val = max(1, int(n_total * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    print(f"[DATA] Чанков всего: {n_total:,} (train={len(train_idx):,}, val={len(val_idx):,}), "
          f"seq_len={seq_len}")
    return trimmed, train_idx, val_idx


def batch_iterator(trimmed, idx_pool, batch_size, data_sharding, seed, shuffle=True):
    local_rng = np.random.RandomState(seed)
    idx_local = np.copy(idx_pool)
    while True:
        if shuffle:
            local_rng.shuffle(idx_local)
        n_steps = len(idx_local) // batch_size
        for step in range(n_steps):
            batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
            rows = trimmed[batch_idx]
            input_ids = rows[:, :-1]
            labels = rows[:, 1:]
            yield {
                "input_ids": jax.device_put(jnp.asarray(input_ids, dtype=jnp.int32), data_sharding),
                "labels": jax.device_put(jnp.asarray(labels, dtype=jnp.int32), data_sharding),
            }
        if not shuffle:
            break


# ==========================================================================
# Plateau-adaptive schedule.
#   - warmup: линейный рост 0 -> peak_lr
#   - после warmup: peak_lr * current_multiplier
#   - current_multiplier уменьшается вручную при обнаружении plateau
#
# ВАЖНО: эта функция вызывается на ЧИСТОМ Python-уровне (вне jit) каждый
# эффективный шаг, чтобы получить конкретное скалярное значение lr, которое
# затем передаётся в compiled_apply как обычный (трассируемый) аргумент.
# Раньше lr передавался как Python-замыкание внутрь оптимизатора,
# скомпилированного через jax.jit -- JAX трассирует такую лямбду только
# один раз при первой компиляции и "замораживает" значение
# current_lr_multiplier, бывшее на тот момент (1.0), навсегда. Plateau-reduce
# после этого продолжал печататься в логах (лог считается вне jit), но
# РЕАЛЬНЫЙ lr внутри optax.adamw никогда не менялся. Передача lr явным
# трассируемым аргументом устраняет эту проблему.
# ==========================================================================
def make_plateau_schedule(peak_lr, warmup_steps):
    def schedule(step, multiplier):
        """multiplier -- scalar (обычно float), управляется извне."""
        step = jnp.asarray(step, dtype=jnp.float32)
        multiplier = jnp.asarray(multiplier, dtype=jnp.float32)
        warmup_frac = jnp.clip(step / max(warmup_steps, 1), 0.0, 1.0)
        warmup_lr = peak_lr * warmup_frac
        stable_lr = peak_lr * multiplier
        lr = jnp.where(step < warmup_steps, warmup_lr, stable_lr)
        return lr
    return schedule


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def compute_bpb_loss(params, model, batch, deterministic, rngs=None):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    kwargs = {"deterministic": deterministic}
    if rngs is not None:
        kwargs["rngs"] = rngs
    logits = model.apply({"params": params}, input_ids, **kwargs).astype(jnp.float32)
    logits = jnp.nan_to_num(jnp.clip(logits, -30.0, 30.0), nan=0.0, posinf=30.0, neginf=-30.0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    ce_nats = jnp.mean(nll)
    ce_nats = jnp.nan_to_num(ce_nats, nan=0.0, posinf=20.0, neginf=0.0)
    bpb = ce_nats / jnp.log(2.0)
    return ce_nats, bpb


def main():
    cfg_run = RUN_CONFIG
    os.makedirs(CKPT_ROOT, exist_ok=True)

    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]
    set_model_mesh(mesh, batch_axis="tpu_nodes")
    print(f"[TPU] Устройств: {n_devices}")

    if RUN_CONFIG["seq_len"] % GDN2_PALLAS_BT != 0:
        raise ValueError(
            f"seq_len={RUN_CONFIG['seq_len']} должен делиться на GDN2_PALLAS_BT={GDN2_PALLAS_BT}"
        )
    if RUN_CONFIG["micro_batch_size"] % n_devices != 0:
        raise ValueError("micro_batch_size должен делиться на число устройств.")

    model_cfg = load_config()
    model = FullGDN2BlockDeltaModel(cfg=model_cfg)

    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))
    scalar_sharding = NamedSharding(mesh, P())

    # ---- init params ----
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
    n_params = count_params(params)
    print(f"[MODEL] Параметров: {n_params:,} (≈ {n_params/1e6:.1f}M)")

    # ---- diagnostics ----
    leaf_layer_map = make_leaf_layer_map(abstract_params)
    layer_tags = param_layer_tags(leaf_layer_map)
    layer_grad_stats_fn = build_leaf_stats_fn(leaf_layer_map, layer_tags)
    layer_grad_raw_stats_fn = build_leaf_raw_stats_fn(leaf_layer_map, layer_tags)
    layer_w_stats_fn = build_leaf_stats_fn(leaf_layer_map, layer_tags)
    print(f"[DIAG] По-слойных тегов: {len(layer_tags)} -> {layer_tags}")

    # ---- optimizer: AdamW + plateau-adaptive schedule + clip ----
    lr_schedule = make_plateau_schedule(cfg_run["peak_lr"], cfg_run["warmup_steps"])

    # inject_hyperparams должен оборачивать ФАБРИКУ transformации (функцию,
    # которая по значению learning_rate строит optax.chain заново), а не
    # уже готовый объект optax.chain(...) -- см. историю ошибки:
    # "GradientTransformationExtraArgs(...) is not a callable object".
    def make_tx(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(cfg_run["grad_clip_norm"]),
            optax.adamw(learning_rate=learning_rate, weight_decay=cfg_run["weight_decay"]),
        )

    # Начальное значение learning_rate здесь чисто формальное (для init) --
    # реальное значение подставляется explicit-аргументом `lr` в apply_step
    # на каждом шаге, см. ниже.
    tx = optax.inject_hyperparams(make_tx)(learning_rate=cfg_run["peak_lr"])

    # ВАЖНО: abstract_params передаётся как настоящий аргумент eval_shape, а
    # не через замыкание в zero-arg лямбде (как это было раньше:
    # `jax.eval_shape(lambda: tx.init(abstract_params))`). В таком виде
    # внутрь tx.init попадали бы буквальные объекты jax.ShapeDtypeStruct как
    # обычные Python-значения (а не трассируемые abstract-tracers), и
    # optax.tree.dtype(...) падал на jnp.asarray(ShapeDtypeStruct(...)) с
    # TypeError. Передача аргументом заставляет eval_shape подставить
    # корректные abstract-значения нужной формы/типа.
    opt_state_abstract = jax.eval_shape(tx.init, abstract_params)
    opt_state_sharding = jax.tree_util.tree_map_with_path(_shard_leaf, opt_state_abstract)
    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    # ---- plateau state (plain Python, обновляется после eval) ----
    current_lr_multiplier = 1.0
    evals_since_improvement = 0
    cooldown_counter = 0          # шагов до конца cooldown
    total_reduces_done = 0

    print(f"[LR-SCHEDULE] Plateau-adaptive: warmup=[0,{cfg_run['warmup_steps']}) "
          f"peak_lr={cfg_run['peak_lr']} min_lr={cfg_run['min_lr']} "
          f"factor={cfg_run['lr_reduce_factor']} patience={cfg_run['patience']} "
          f"cooldown={cfg_run['lr_cooldown_steps']} steps")

    # ---- data ----
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

    # ---- compiled steps ----
    accum_steps = cfg_run["accum_steps"]

    def train_micro_step(p, accum_grads, batch, rng):
        def loss_fn(param):
            ce_nats, bpb = compute_bpb_loss(param, model, batch, deterministic=False, rngs={"dropout": rng})
            return ce_nats, bpb
        (ce_nats, bpb), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        micro_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return new_accum, ce_nats, bpb, micro_grad_norm

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

        layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite = layer_grad_stats_fn(avg_grads)
        layer_grad_raw_maxabs, layer_grad_nonfinite_count = layer_grad_raw_stats_fn(avg_grads)

        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), avg_grads
        )

        # `lr` приходит как обычный трассируемый аргумент jitted-функции (не
        # Python-замыкание) -- поэтому меняется КАЖДЫЙ вызов вместе с
        # изменением current_lr_multiplier снаружи. Подставляем его в
        # hyperparams состояния перед tx.update, как задумано
        # optax.inject_hyperparams.
        s = s._replace(hyperparams=dict(s.hyperparams, learning_rate=lr))

        updates, new_s = tx.update(avg_grads, s, p)
        new_p_candidate = optax.apply_updates(p, updates)
        new_p_candidate = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p_candidate,
        )

        new_p = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_finite, new, old), p, new_p_candidate
        )
        new_s = jax.tree_util.tree_map(
            lambda old, new: jnp.where(is_finite, new, old) if hasattr(new, "shape") else new,
            s, new_s,
        )

        layer_w_norms, layer_w_maxabs, layer_w_nonfinite = layer_w_stats_fn(new_p)
        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)

        return (new_p, new_s, zero_accum, is_finite, global_norm,
                layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite,
                layer_grad_raw_maxabs, layer_grad_nonfinite_count,
                layer_w_norms, layer_w_maxabs, layer_w_nonfinite)

    compiled_apply = jax.jit(
        apply_step,
        donate_argnums=(0, 1, 2),
        in_shardings=(param_sharding, opt_state_sharding, param_sharding,
                       NamedSharding(mesh, P()), scalar_sharding),
        out_shardings=(
            param_sharding, opt_state_sharding, param_sharding,
            NamedSharding(mesh, P()), NamedSharding(mesh, P()),
            NamedSharding(mesh, P(None)), NamedSharding(mesh, P(None)), NamedSharding(mesh, P(None)),
            NamedSharding(mesh, P(None)), NamedSharding(mesh, P(None)),
            NamedSharding(mesh, P(None)), NamedSharding(mesh, P(None)), NamedSharding(mesh, P(None)),
        ),
    )

    def val_step(p, batch):
        ce_nats, bpb = compute_bpb_loss(p, model, batch, deterministic=True)
        return ce_nats, bpb

    compiled_val = jax.jit(
        val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())),
    )

    # ---- checkpoint manager ----
    mngr = ocp.CheckpointManager(
        os.path.join(CKPT_ROOT, "latest"),
        ocp.StandardCheckpointer(),
        ocp.CheckpointManagerOptions(max_to_keep=2, create=True, enable_async_checkpointing=False),
    )

    global _run
    if _HAS_WANDB:
        _run = wandb.init(project="gdn2-100m-enwik8", name=cfg_run["run_name"], config=cfg_run)

    zero_accum = jax.jit(lambda p: jax.tree_util.tree_map(jnp.zeros_like, p), out_shardings=param_sharding)(params)
    accum_grads = zero_accum

    global_rng = jax.random.PRNGKey(cfg_run["seed"] + 1)
    global_step = 0
    nonfinite_consecutive = 0
    nonfinite_window = deque(maxlen=cfg_run["nonfinite_window_size"])
    best_val_bpb = float("inf")
    last_ckpt_time = time.perf_counter()

    _micro_bpb_acc, _micro_grad_norm_acc = [], []

    print("[TRAIN] 🚀 Старт обучения. Plateau-adaptive LR. Каждый эффективный шаг -- полная диагностика.")

    micro_step = 0
    while global_step < cfg_run["total_train_steps"]:
        batch = next(train_stream)
        global_rng, step_rng = jax.random.split(global_rng)

        accum_grads, ce_nats, bpb, micro_grad_norm = compiled_train_micro(params, accum_grads, batch, step_rng)
        _micro_bpb_acc.append(float(jax.device_get(bpb)))
        _micro_grad_norm_acc.append(float(jax.device_get(micro_grad_norm)))
        micro_step += 1

        if micro_step % accum_steps == 0:
            # lr считается на Python-уровне ДО вызова compiled_apply и
            # передаётся как явный аргумент -- это то самое значение, которое
            # реально применится внутри optax.adamw на этом шаге (см.
            # комментарий у make_plateau_schedule).
            next_step_for_lr = global_step + 1
            cur_lr = float(jax.device_get(lr_schedule(next_step_for_lr, current_lr_multiplier)))
            lr_arr = jax.device_put(jnp.asarray(cur_lr, dtype=jnp.float32), scalar_sharding)

            (params, opt_state, accum_grads, is_finite, global_norm,
             layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite,
             layer_grad_raw_maxabs, layer_grad_nonfinite_count,
             layer_w_norms, layer_w_maxabs, layer_w_nonfinite) = compiled_apply(
                params, opt_state, accum_grads, accum_steps, lr_arr
            )

            step_finite = bool(jax.device_get(is_finite))
            global_step += 1

            # cooldown считаем в шагах (не eval-ах)
            if cooldown_counter > 0:
                cooldown_counter -= 1

            nonfinite_window.append(0 if step_finite else 1)
            nonfinite_consecutive = 0 if step_finite else nonfinite_consecutive + 1
            window_ratio = sum(nonfinite_window) / len(nonfinite_window)

            mean_bpb = float(np.mean(_micro_bpb_acc))
            max_grad = float(np.max(_micro_grad_norm_acc))
            _micro_bpb_acc, _micro_grad_norm_acc = [], []

            print(f"[STEP {global_step}/{cfg_run['total_train_steps']}] "
                  f"bpb={mean_bpb:.4f} lr={cur_lr:.2e} global_grad_norm={float(jax.device_get(global_norm)):.4f} "
                  f"finite={step_finite} nonfinite_window={window_ratio:.2%} "
                  f"lr_mult={current_lr_multiplier:.4f} cooldown={cooldown_counter}")

            step_metrics = {
                "train/bpb": mean_bpb,
                "train/lr": cur_lr,
                "train/global_grad_norm": float(jax.device_get(global_norm)),
                "train/max_micro_grad_norm": max_grad,
                "train/step_skipped_nonfinite": int(not step_finite),
                "train/nonfinite_window_ratio": window_ratio,
                "train/lr_multiplier": current_lr_multiplier,
                "train/cooldown_remaining": cooldown_counter,
            }

            _lgn = jax.device_get(layer_grad_norms)
            _lgm_raw = jax.device_get(layer_grad_raw_maxabs)
            _lgnf_cnt = jax.device_get(layer_grad_nonfinite_count)
            _lwn = jax.device_get(layer_w_norms)
            for tag, gn, gm_raw, nf_cnt, wn in zip(layer_tags, _lgn, _lgm_raw, _lgnf_cnt, _lwn):
                step_metrics[f"layer_grad_norm/{tag}"] = float(gn)
                step_metrics[f"layer_grad_raw_maxabs/{tag}"] = float(gm_raw)
                step_metrics[f"layer_w_norm/{tag}"] = float(wn)
                if int(nf_cnt) > 0:
                    print(f"    ⚠️ [LAYER-DIAG] {tag}: {int(nf_cnt)} non-finite элементов в градиенте "
                          f"(raw maxabs по конечной части={float(gm_raw):.3e})")

            wandb_log(global_step, step_metrics)

            if not step_finite:
                print(f"[WARNING] ⚠️ Non-finite градиент на шаге {global_step} -- обновление ПРОПУЩЕНО.")

            hit_consecutive = nonfinite_consecutive >= cfg_run["nonfinite_consecutive_limit"]
            hit_window = (len(nonfinite_window) >= cfg_run["nonfinite_window_size"]
                          and window_ratio >= cfg_run["nonfinite_window_ratio"])
            if hit_consecutive or hit_window:
                print(f"🛑 [AUTO-STOP] Системная нестабильность на шаге {global_step} "
                      f"(consecutive={nonfinite_consecutive}, window_ratio={window_ratio:.2%}). "
                      f"Сохраняю и останавливаюсь.")
                mngr.save(global_step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
                mngr.wait_until_finished()
                wandb_log(global_step, {"train/auto_stopped": 1})
                break

            # ---- eval + plateau logic ----
            if global_step % cfg_run["eval_every_steps"] == 0:
                vstream = val_stream_factory()
                bpb_sum, n_done = 0.0, 0
                for _ in range(cfg_run["eval_batches"]):
                    vb = next(vstream)
                    _, vbpb = compiled_val(params, vb)
                    bpb_sum += float(jax.device_get(vbpb))
                    n_done += 1
                val_bpb = bpb_sum / max(n_done, 1)

                improved = val_bpb < best_val_bpb
                if improved:
                    best_val_bpb = val_bpb
                    evals_since_improvement = 0
                    print(f"[EVAL] Step {global_step}: val_bpb={val_bpb:.4f} 🟢 УЛУЧШЕНИЕ (best={best_val_bpb:.4f})")
                else:
                    evals_since_improvement += 1
                    print(f"[EVAL] Step {global_step}: val_bpb={val_bpb:.4f} 🔴 Нет улучшения "
                          f"({evals_since_improvement}/{cfg_run['patience']}) best={best_val_bpb:.4f}")

                wandb_log(global_step, {"eval/val_bpb": val_bpb, "eval/best_val_bpb": best_val_bpb})

                if improved:
                    wandb_log(global_step, {"eval/best_val_bpb": best_val_bpb})

                # --- plateau check (только вне cooldown) ---
                plateau_triggered = False
                if cooldown_counter == 0 and evals_since_improvement >= cfg_run["patience"]:
                    new_multiplier = current_lr_multiplier * cfg_run["lr_reduce_factor"]
                    if new_multiplier * cfg_run["peak_lr"] >= cfg_run["min_lr"]:
                        current_lr_multiplier = new_multiplier
                        cooldown_counter = cfg_run["lr_cooldown_steps"]
                        total_reduces_done += 1
                        plateau_triggered = True
                        print(f"[PLATEAU] ⬇️ LR reduce #{total_reduces_done}: multiplier={current_lr_multiplier:.4f} "
                              f"new_lr={cfg_run['peak_lr']*current_lr_multiplier:.2e} "
                              f"cooldown={cooldown_counter} шагов")
                        wandb_log(global_step, {
                            "train/lr_reduced": 1,
                            "train/lr_multiplier": current_lr_multiplier,
                            "train/total_lr_reduces": total_reduces_done,
                        })
                    else:
                        print(f"[PLATEAU] ⛔ Достигнут min_lr ({cfg_run['min_lr']:.2e}), "
                              f"дальнейшее снижение заблокировано.")
                        wandb_log(global_step, {"train/min_lr_reached": 1})

            # ---- checkpoint по времени ----
            now = time.perf_counter()
            if now - last_ckpt_time >= cfg_run["ckpt_every_seconds"]:
                mngr.save(global_step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
                mngr.wait_until_finished()
                with open(os.path.join(CKPT_ROOT, "meta.json"), "w") as f:
                    json.dump({
                        "global_step": global_step,
                        "best_val_bpb": best_val_bpb,
                        "lr_multiplier": current_lr_multiplier,
                        "total_lr_reduces": total_reduces_done,
                    }, f)
                print(f"[CKPT] Сохранено на шаге {global_step}")
                last_ckpt_time = now

    print(f"[DONE] Обучение завершено на шаге {global_step}. best_val_bpb={best_val_bpb:.4f} "
          f"final_lr_mult={current_lr_multiplier:.4f} total_reduces={total_reduces_done}")
    if _HAS_WANDB and _run is not None:
        wandb.summary["final/best_val_bpb"] = best_val_bpb
        wandb.summary["final/global_step"] = global_step
        wandb.summary["final/lr_multiplier"] = current_lr_multiplier
        wandb.summary["final/total_lr_reduces"] = total_reduces_done
        wandb.finish()


if __name__ == "__main__":
    main()
