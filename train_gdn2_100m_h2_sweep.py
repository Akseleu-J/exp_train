"""
train_gdn2_100m_h2_sweep.py -- проверка H2 (recurrent depth = дешёвая
эффективная глубина) на вашем уже обученном enwik8-сетапе.

ВАЖНО: ваш текущий чекпоинт (3 эпохи, bpb~1.3) обучался как UNTIED
(12 разных слоёв). Здесь мы НЕ дообучаем его дальше -- tied-архитектура
имеет другую параметризацию (общие веса), поэтому TIED-прогоны стартуют
С НУЛЯ (init_fn), а untied-baseline в этом файле -- это КОНТРОЛЬНЫЙ
повторный прогон с нуля с ТЕМ ЖЕ бюджетом шагов, чтобы сравнение было
честным (сравнивать tied-с-нуля против вашего уже сошедшегося 3-эпочного
чекпоинта было бы некорректно -- разный "возраст" обучения).

ДВА РЕЖИМА (переключаются RUN_MODE ниже):

  "compute_match": один prognon tied (R=n_reasoning_cycles) и один
      untied (num_blocks=R) с ОДИНАКОВЫМ числом forward-применений GDN2 и
      ОДИНАКОВЫМ total_train_steps -- главный H2-тест. Если tied
      финиширует с близким bpb -- H2 подтверждён (recurrent-depth valid
      substitute for unique weights per layer at this scale/budget).

  "r_curve": фиксируем tied-веса ОДНОГО shared_block, меняем ТОЛЬКО R
      (n_reasoning_cycles) между независимыми прогонами (2,4,8,12), тем
      же total_train_steps каждый -- строим bpb(R). Каждый R -- ОТДЕЛЬНЫЙ
      прогон с независимой инициализацией (не warm-start между ними,
      чтобы не путать "R помогает" с "дольше обучались").

Оба режима используют ОДИН тренировочный луп (train_one_run), собранный
1-в-1 по образцу вашего train_gdn2_100m.py (plateau-adaptive LR не
переносим сюда -- для короткого sweep-бюджета фиксированный cosine-decay
после warmup достаточно и не добавляет лишнюю степень свободы в
сравнение; если понадобится -- добавьте _ReduceLROnPlateauController
отдельно, как в основном скрипте).
"""
from __future__ import annotations

import os
import time
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model_gdn2_100m_tied import (
    FullGDN2BlockDeltaModelTied, TiedModelConfig, count_params, GDN2_PALLAS_BT,
    set_model_mesh,
)
from train_gdn2_100m import (
    load_enwik8_bytes, make_chunked_dataset, batch_iterator, make_tpu_mesh,
    compute_bpb_loss,
)

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# ==========================================================================
# ВЫБЕРИТЕ РЕЖИМ ЗДЕСЬ
# ==========================================================================
RUN_MODE = "compute_match"  # "compute_match" | "r_curve"

DATA_PATH = "/kaggle/input/datasets/nightfury1103/enwik8/enwik8"
OUT_DIR = "/kaggle/working/h2_sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

COMMON = dict(
    seq_len=2048,
    micro_batch_size=8,
    accum_steps=4,
    total_train_steps=1500,      # короче полного прогона -- это sweep, не финальное обучение
    warmup_steps=150,
    peak_lr=2e-4,
    min_lr=1e-6,
    weight_decay=0.01,
    grad_clip_norm=1.0,
    eval_every_steps=150,
    eval_batches=20,
    eval_seed=999,
    nonfinite_consecutive_limit=4,
    nonfinite_window_size=15,
    nonfinite_window_ratio=0.25,
    val_split=0.02,
    seed=42,
)

BASE_MODEL_KW = dict(
    d_model=1024, n_heads=8, d_latent=512, layers_per_block=3,
    vocab_size=256, tie_embeddings=True, label_smoothing=0.0, d_conv=4,
)

# compute_match: baseline untied num_blocks == tied n_reasoning_cycles
COMPUTE_MATCH_R = 4          # соответствует вашему текущему num_layers=12 (12//3=4 блока)

# r_curve: набор R для отдельных с-нуля прогонов
R_CURVE_VALUES = [2, 4, 8, 12]


def cosine_schedule(step, peak_lr, warmup_steps, total_steps, min_lr):
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup_frac = jnp.clip(step / max(warmup_steps, 1), 0.0, 1.0)
    warmup_lr = peak_lr * warmup_frac
    progress = jnp.clip((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0, 1.0)
    decay_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < warmup_steps, warmup_lr, decay_lr)


def build_model(tied: bool, num_layers: int, n_cycles: int):
    cfg = TiedModelConfig(
        num_layers=num_layers,
        tie_all_blocks=tied,
        n_reasoning_cycles=n_cycles,
        **BASE_MODEL_KW,
    )
    model = FullGDN2BlockDeltaModelTied(cfg=cfg)
    return model, cfg


def train_one_run(run_name: str, model, cfg, run_cfg, trimmed, train_idx, val_idx, mesh, n_devices):
    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))
    scalar_sharding = NamedSharding(mesh, P())
    init_rng = jax.random.PRNGKey(run_cfg["seed"])

    abstract_params = jax.eval_shape(
        lambda: model.init(init_rng, jnp.zeros((run_cfg["micro_batch_size"], run_cfg["seq_len"]), dtype=jnp.int32))
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
        lambda rng: model.init(rng, jnp.zeros((run_cfg["micro_batch_size"], run_cfg["seq_len"]), dtype=jnp.int32))["params"],
        out_shardings=param_sharding,
    )
    params = init_fn(init_rng)
    n_params = count_params(params)
    print(f"[{run_name}] Параметров: {n_params:,} (≈{n_params/1e6:.1f}M)")

    # ФИКС (тот же, что в train_gdn2_100m.py): inject_hyperparams должен
    # оборачивать ФАБРИКУ трансформации (функцию learning_rate -> tx), а не
    # готовый optax.chain(...) с lr-замыканием внутри jit -- иначе JAX
    # трассирует лямбду один раз при компиляции и "замораживает" lr на
    # начальном значении навсегда (реальный lr внутри optax.adamw после
    # этого никогда не меняется, хотя расписание печатается в логах верно).
    # Здесь используем lr явным трассируемым аргументом apply_step, cosine
    # расписание считается на Python-уровне ДО вызова compiled_apply.
    def make_tx(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(run_cfg["grad_clip_norm"]),
            optax.adamw(learning_rate=learning_rate, weight_decay=run_cfg["weight_decay"]),
        )

    tx = optax.inject_hyperparams(make_tx)(learning_rate=run_cfg["peak_lr"])

    opt_state_abstract = jax.eval_shape(tx.init, abstract_params)
    opt_state_sharding = jax.tree_util.tree_map_with_path(_shard_leaf, opt_state_abstract)
    opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    train_stream = batch_iterator(
        trimmed, train_idx, run_cfg["micro_batch_size"], data_sharding,
        seed=run_cfg["seed"], shuffle=True,
    )

    def val_stream_factory():
        return batch_iterator(
            trimmed, val_idx, run_cfg["micro_batch_size"], data_sharding,
            seed=run_cfg["eval_seed"], shuffle=True,
        )

    accum_steps = run_cfg["accum_steps"]

    def train_micro_step(p, accum_grads, batch, rng):
        def loss_fn(param):
            ce_nats, bpb = compute_bpb_loss(param, model, batch, deterministic=False, rngs={"dropout": rng})
            return ce_nats, bpb
        (ce_nats, bpb), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return new_accum, ce_nats, bpb

    compiled_train_micro = jax.jit(
        train_micro_step,
        donate_argnums=(1,),
        in_shardings=(param_sharding, param_sharding,
                       {"input_ids": data_sharding, "labels": data_sharding},
                       NamedSharding(mesh, P(None))),
        out_shardings=(param_sharding, NamedSharding(mesh, P()), NamedSharding(mesh, P())),
    )

    def apply_step(p, s, accum_grads, n_accum, lr):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)
        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))
        is_finite = jnp.isfinite(global_norm)
        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), avg_grads
        )
        # lr приходит как обычный трассируемый аргумент -- подставляем в
        # hyperparams состояния перед tx.update (см. комментарий выше).
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
        return compute_bpb_loss(p, model, batch, deterministic=True)

    compiled_val = jax.jit(
        val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())),
    )

    zero_accum = jax.jit(lambda p: jax.tree_util.tree_map(jnp.zeros_like, p), out_shardings=param_sharding)(params)
    accum_grads = zero_accum
    global_rng = jax.random.PRNGKey(run_cfg["seed"] + 1)
    global_step = 0
    micro_step = 0
    nonfinite_consecutive = 0
    from collections import deque
    nonfinite_window = deque(maxlen=run_cfg["nonfinite_window_size"])
    best_val_bpb = float("inf")
    history = []

    print(f"[{run_name}] 🚀 Старт, total_train_steps={run_cfg['total_train_steps']}")
    t0 = time.perf_counter()

    while global_step < run_cfg["total_train_steps"]:
        batch = next(train_stream)
        global_rng, step_rng = jax.random.split(global_rng)
        accum_grads, ce_nats, bpb = compiled_train_micro(params, accum_grads, batch, step_rng)
        micro_step += 1

        if micro_step % accum_steps == 0:
            # lr считается на Python-уровне ДО compiled_apply и передаётся
            # явным трассируемым аргументом -- то самое значение, которое
            # реально применится внутри optax.adamw на этом шаге.
            next_step_for_lr = global_step + 1
            cur_lr = float(cosine_schedule(
                next_step_for_lr, run_cfg["peak_lr"], run_cfg["warmup_steps"],
                run_cfg["total_train_steps"], run_cfg["min_lr"],
            ))
            lr_arr = jax.device_put(jnp.asarray(cur_lr, dtype=jnp.float32), scalar_sharding)

            params, opt_state, accum_grads, is_finite, global_norm = compiled_apply(
                params, opt_state, accum_grads, accum_steps, lr_arr
            )
            step_finite = bool(jax.device_get(is_finite))
            global_step += 1
            nonfinite_window.append(0 if step_finite else 1)
            nonfinite_consecutive = 0 if step_finite else nonfinite_consecutive + 1
            window_ratio = sum(nonfinite_window) / len(nonfinite_window)

            hit_consecutive = nonfinite_consecutive >= run_cfg["nonfinite_consecutive_limit"]
            hit_window = (len(nonfinite_window) >= run_cfg["nonfinite_window_size"]
                          and window_ratio >= run_cfg["nonfinite_window_ratio"])
            if hit_consecutive or hit_window:
                print(f"[{run_name}] 🛑 AUTO-STOP на шаге {global_step} (нестабильность). Прерываю прогон.")
                break

            if global_step % run_cfg["eval_every_steps"] == 0 or global_step == run_cfg["total_train_steps"]:
                vstream = val_stream_factory()
                bpb_sum, n_done = 0.0, 0
                for _ in range(run_cfg["eval_batches"]):
                    vb = next(vstream)
                    _, vbpb = compiled_val(params, vb)
                    bpb_sum += float(jax.device_get(vbpb))
                    n_done += 1
                val_bpb = bpb_sum / max(n_done, 1)
                best_val_bpb = min(best_val_bpb, val_bpb)
                elapsed = time.perf_counter() - t0
                print(f"[{run_name}] step {global_step}/{run_cfg['total_train_steps']} "
                      f"val_bpb={val_bpb:.4f} best={best_val_bpb:.4f} [{elapsed:.1f}s]")
                history.append({"step": global_step, "val_bpb": val_bpb})

    result = {
        "run_name": run_name,
        "n_params": n_params,
        "best_val_bpb": best_val_bpb,
        "final_step": global_step,
        "history": history,
    }
    with open(os.path.join(OUT_DIR, f"{run_name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{run_name}] ✅ Готово. best_val_bpb={best_val_bpb:.4f}")
    return result


def main():
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]
    set_model_mesh(mesh, batch_axis="tpu_nodes")
    print(f"[TPU] Устройств: {n_devices}")

    if COMMON["seq_len"] % GDN2_PALLAS_BT != 0:
        raise ValueError(f"seq_len должен делиться на GDN2_PALLAS_BT={GDN2_PALLAS_BT}")
    if COMMON["micro_batch_size"] % n_devices != 0:
        raise ValueError("micro_batch_size должен делиться на число устройств.")

    byte_arr = load_enwik8_bytes(DATA_PATH)
    trimmed, train_idx, val_idx = make_chunked_dataset(
        byte_arr, COMMON["seq_len"], COMMON["val_split"], COMMON["seed"]
    )

    results = []

    if RUN_MODE == "compute_match":
        R = COMPUTE_MATCH_R
        # untied baseline: num_layers = R * layers_per_block -> R блоков, разные веса
        model_u, cfg_u = build_model(tied=False, num_layers=R * BASE_MODEL_KW["layers_per_block"], n_cycles=R)
        res_u = train_one_run(f"untied_R{R}", model_u, cfg_u, COMMON, trimmed, train_idx, val_idx, mesh, n_devices)
        results.append(res_u)

        # tied: тот же R применений одного и того же блока
        model_t, cfg_t = build_model(tied=True, num_layers=BASE_MODEL_KW["layers_per_block"], n_cycles=R)
        res_t = train_one_run(f"tied_R{R}", model_t, cfg_t, COMMON, trimmed, train_idx, val_idx, mesh, n_devices)
        results.append(res_t)

        print("\n===== H2 COMPUTE-MATCH РЕЗУЛЬТАТ =====")
        print(f"untied (R={R}, {res_u['n_params']/1e6:.1f}M params): best_val_bpb={res_u['best_val_bpb']:.4f}")
        print(f"tied   (R={R}, {res_t['n_params']/1e6:.1f}M params): best_val_bpb={res_t['best_val_bpb']:.4f}")
        gap = res_t["best_val_bpb"] - res_u["best_val_bpb"]
        print(f"gap (tied - untied) = {gap:+.4f} bpb")
        print("Малый gap -> H2 подтверждена на этом бюджете (recurrent depth ~ valid substitute).")
        print("Большой gap -> уникальные веса на слой всё ещё важны на этом масштабе.")

    elif RUN_MODE == "r_curve":
        for R in R_CURVE_VALUES:
            model_t, cfg_t = build_model(tied=True, num_layers=BASE_MODEL_KW["layers_per_block"], n_cycles=R)
            res = train_one_run(f"tied_Rcurve_R{R}", model_t, cfg_t, COMMON, trimmed, train_idx, val_idx, mesh, n_devices)
            results.append(res)

        print("\n===== H2 R-CURVE РЕЗУЛЬТАТ =====")
        for r_val, res in zip(R_CURVE_VALUES, results):
            print(f"R={r_val:>3}: best_val_bpb={res['best_val_bpb']:.4f}")
        print("Монотонное убывание с насыщением -> лишний compute-per-token помогает сам по себе (Astra-like).")
        print("Плоско/шумно -> depth reuse не даёт выигрыша на этом масштабе без reasoning-сигнала.")

    else:
        raise ValueError(f"Неизвестный RUN_MODE={RUN_MODE!r}")

    with open(os.path.join(OUT_DIR, f"summary_{RUN_MODE}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
