"""
train_setup.py -- диагностика non-finite ПО ОПТИМИЗАТОРСКИМ ГРУППАМ, TPU
mesh / шардинг / компиляция train-step'ов, multi-source dataloader.

ФИКС (этот пасс -- диагностика сведена ТОЛЬКО к оптимизатору, см. чат):
раньше _DIAG_GROUPS группировала параметры по АРХИТЕКТУРНОМУ типу ядра
(gdn2/mamba2/mla/moe/embed/other, через _classify_leaf_group -- ручное
сопоставление по подстрокам пути). Это дублировало то, что optimizer.py
и так уже решает через _label_leaf (какой ИМЕННО optax-трансформ
(muon/lion/adamw_decay/adamw_nodecay/frozen) обновляет этот лист), и не
отвечало напрямую на вопрос "какой оптимизатор сейчас нестабилен" --
приходилось смотреть и на группу, и отдельно на muon_orth_resid.

Теперь _DIAG_GROUPS == те же метки, что использует
optimizer.py's multi_transform (тот же `_label_leaf`, импортирован как
`_optimizer_label_leaf`). "non-finite в группе muon" теперь означает
буквально "не-finite в подветке параметров, которую обновляет Muon" -- без
необходимости смотреть внутрь конкретного Pallas-ядра (GDN-2/Mamba2/MLA).

Плюс: build_group_norms_fn -- НОВОЕ, даёт L2-норму градиента И весов ПО
КАЖДОЙ optimizer-группе каждый эффективный шаг (не только булевый
nonfinite-флаг) -- видно, что РАСТЁТ, до того как оно реально сломается.
distributed_apply_step теперь возвращает 19 значений вместо 17 --
добавлены group_grad_norms, group_weight_norms В САМОМ КОНЦЕ.

ФИКС (этот пасс -- критический краш компиляции): aux_info_sharding раньше
содержал ~30 ключей для kernel/activation-level sow()-значений
(gdn2_kernelstage_*, mamba2_ssm_out_*, mla_*, layer_delta_*, layer_resid_*,
final_hidden_*, ...), которых compute_loss() (optimizer.py) БОЛЬШЕ НЕ
возвращает -- соответствующие self.sow(...) вызовы удалены из model.py
(см. его докстринг "диагностика сведена ТОЛЬКО к оптимизатору").
jax.jit(..., out_shardings=aux_info_sharding) требует, чтобы дерево
out_shardings СТРУКТУРНО совпадало с реальным выходом функции -- лишние
ключи в out_shardings ломают компиляцию compiled_train_micro при первом
же реальном вызове (не при импорте модуля -- падает ровно в тот момент,
когда пытаешься реально запустить обучение). Оставлены только ключи,
которые compute_loss реально кладёт в aux_info при return_aux=True.

ФИКС (локализация Muon orth_resid, предыдущий пасс, без изменений в этом
пассе): построение _MUON_LEAF_PATHS через optimizer.build_muon_leaf_paths
-- модульный атрибут, экспортируемый для train.py, чтобы КАЖДЫЙ шаг
расшифровывать worst_leaf_idx (число из jit-графа, см.
MuonState.worst_leaf_idx) в реальный путь параметра, без offline-скриптов.

Остальное -- без изменений логики относительно предыдущего пасса
(ФИКС #1-10, см. прежний докстринг ниже).
"""
from __future__ import annotations

import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from model import FullHybridMoEModel, ModelConfig, set_model_mesh
from optimizer import (
    compute_loss, make_hybrid_optimizer, extract_zclip_diagnostics,
    extract_muon_diagnostics, build_muon_leaf_paths, _label_leaf as _optimizer_label_leaf,
)
from utils import path_to_str
from diagnostics import (
    make_leaf_layer_map, param_layer_tags, layer_tags_in_sow_order, build_leaf_stats_fn, build_leaf_raw_stats_fn,
)
from diagnostics import build_leaf_raw_stats_fn

DATASET_FRACTION = 1
DATASET_FRACTION_SEED = 777

NONFINITE_CONSECUTIVE_LIMIT = 4
NONFINITE_WINDOW_SIZE = 15
NONFINITE_WINDOW_RATIO = 0.25

ROUTER_TEMP_DECAY_RATE = 0.02
ROUTER_TEMP_INIT = 10.0


def _router_temp_decay_leaf(path, param):
    path_str = path_to_str(path)
    return ROUTER_TEMP_DECAY_RATE if "router_temp" in path_str else 0.0


def apply_router_temp_decay(new_params, decay_map):
    return jax.tree_util.tree_map(
        lambda p, rate: p - rate * (p - ROUTER_TEMP_INIT),
        new_params, decay_map,
    )


EXPERT_BIAS_GAMMA = 0.02


def _build_expert_bias_index_map(abstract_params):
    counter = {"i": 0}

    def _mark(path, leaf):
        path_str = path_to_str(path)
        if "expert_bias" in path_str and hasattr(leaf, "shape"):
            idx = counter["i"]
            counter["i"] += 1
            return idx
        return -1

    return jax.tree_util.tree_map_with_path(_mark, abstract_params)


def apply_expert_bias_update(new_params, bias_index_map, assignment_frac_stacked, gamma=EXPERT_BIAS_GAMMA):
    if assignment_frac_stacked is None:
        return new_params

    def _update(p, idx):
        if idx < 0:
            return p
        frac = assignment_frac_stacked[idx]
        e_routed = p.shape[-1]
        target = 1.0 / e_routed
        overloaded = frac > target
        return p + gamma * jnp.where(overloaded, -1.0, 1.0)

    return jax.tree_util.tree_map(_update, new_params, bias_index_map)


SESSION_TIME_BUDGET_SECONDS = 9 * 3600 - 5 * 60

# ==========================================================================
# ФИКС (этот пасс -- диагностика сведена ТОЛЬКО к оптимизатору): группы
# теперь = метки optax multi_transform (см. optimizer.py's _label_leaf),
# не архитектурный тип ядра. "moe"/"muon_decay" отдельных групп больше нет
# -- MoE-эксперты и roter уже классифицируются как lion/adamw_nodecay
# самим _label_leaf (см. optimizer.py), так что их nonfinite/norm-статус
# виден через ТУ ЖЕ группу, что реально их обновляет.
# ==========================================================================
_DIAG_GROUPS = ("muon", "lion", "adamw_decay", "adamw_nodecay", "frozen")


def make_grad_group_map(params):
    """ФИКС (этот пасс): группировка листьев теперь по ОПТИМИЗАТОРСКОЙ
    метке (_label_leaf из optimizer.py -- та же функция, что
    multi_transform реально использует для выбора трансформа), а не по
    архитектурному типу ядра. Один источник истины для "какая группа
    отвечает за какие параметры" -- optimizer.py, не дублируется здесь."""
    return jax.tree_util.tree_map_with_path(_optimizer_label_leaf, params)


def build_group_nonfinite_flags(grad_group_map):
    leaves_g, _ = jax.tree_util.tree_flatten(grad_group_map)

    def _flags(avg_grads):
        leaves_grad = jax.tree_util.tree_leaves(avg_grads)
        flags = []
        for group in _DIAG_GROUPS:
            idxs = [i for i, g in enumerate(leaves_g) if g == group]
            if not idxs:
                flags.append(jnp.array(False))
                continue
            group_flags = jnp.stack([
                jnp.logical_not(jnp.all(jnp.isfinite(leaves_grad[i]))) for i in idxs
            ])
            flags.append(jnp.any(group_flags))
        return jnp.stack(flags)

    return _flags


def build_group_norms_fn(grad_group_map, groups):
    """НОВОЕ (этот пасс): L2-норма ПО КАЖДОЙ оптимизаторской группе --
    те же группы, что build_group_nonfinite_flags, но с реальной
    величиной, а не только булевым флагом. Одна и та же функция
    применяется и к avg_grads (-> group_grad_norms), и к new_p
    (-> group_weight_norms) -- обе имеют ту же листовую структуру, что и
    grad_group_map. Даёт видимость "что растёт" в конкретной группе
    оптимизатора ДО того, как она реально уйдёт в non-finite."""
    leaves_g, _ = jax.tree_util.tree_flatten(grad_group_map)
    idx_by_group = {grp: [i for i, gg in enumerate(leaves_g) if gg == grp] for grp in groups}

    def _norms(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        norms = []
        for grp in groups:
            idxs = idx_by_group[grp]
            if not idxs:
                norms.append(jnp.array(0.0, dtype=jnp.float32))
                continue
            safe = [jnp.nan_to_num(leaves[i].astype(jnp.float32), nan=0.0, posinf=0.0, neginf=0.0) for i in idxs]
            sq = sum(jnp.sum(jnp.square(s)) for s in safe)
            norms.append(jnp.sqrt(sq))
        return jnp.stack(norms)

    return _norms


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


_PARAM_LAYER_TAGS = None
_SOW_LAYER_TAGS = None
# ФИКС (локализация Muon orth_resid, см. модульный докстринг выше):
# список путей muon-параметров, ПОРЯДОК СОВПАДАЕТ с worst_leaf_idx из
# MuonState -- заполняется внутри make_shard_and_compile, экспортируется
# для train.py.
_MUON_LEAF_PATHS = None


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int,
                           seq_len: int = 8192, accum_steps: int = 1,
                           warmup_freeze_step=None, muon_diagnostic_disable: bool = False):
    """warmup_freeze_step прокидывается напрямую в
    make_hybrid_optimizer(warmup_freeze_step=...). None -- обычный полный
    warmup/cosine-decay без заморозки; int -- LR-schedule заморожена на
    этом шаге.

    muon_diagnostic_disable: прокидывается в make_hybrid_optimizer -- если
    True, ВСЕ muon-параметры переводятся на adamw_nodecay (полное
    временное отключение Muon, см. чат)."""
    global _PARAM_LAYER_TAGS, _SOW_LAYER_TAGS, _MUON_LEAF_PATHS

    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by n_devices={n_devices}."
        )

    batch_axis = "tpu_nodes"
    set_model_mesh(mesh, batch_axis=batch_axis)

    _opt_result = make_hybrid_optimizer(
        total_steps=total_steps, warmup_freeze_step=warmup_freeze_step,
        muon_diagnostic_disable=muon_diagnostic_disable,
    )
    if isinstance(_opt_result, (optax.GradientTransformation, optax.GradientTransformationExtraArgs)):
        tx = _opt_result
        lr_schedule = None
        print("[OPTIMIZER] ⚠️ make_hybrid_optimizer вернул голый tx (без lr_schedule).")
    else:
        tx, lr_schedule = _opt_result
    model = FullHybridMoEModel(cfg=config)

    init_rng = jax.random.PRNGKey(0)
    abstract_params = jax.eval_shape(
        lambda: model.init(init_rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
    )["params"]

    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))

    MIN_SHARD_SIZE = 128

    def _get_shard_spec(path, param):
        if not hasattr(param, "shape") or param.ndim == 0:
            return NamedSharding(mesh, P())
        path_str = path_to_str(path)
        if ("experts_block" in path_str or "routed_experts" in path_str
                or "routed_w1" in path_str or "routed_w2" in path_str):
            return NamedSharding(mesh, P(*([None] * param.ndim)))
        best_axis, best_size = None, -1
        for i, size in enumerate(param.shape):
            if size % n_devices == 0 and (size // n_devices) >= MIN_SHARD_SIZE and size > best_size:
                best_axis, best_size = i, size
        if best_axis is None:
            return NamedSharding(mesh, P(*([None] * param.ndim)))
        spec = [None] * param.ndim
        spec[best_axis] = "tpu_nodes"
        return NamedSharding(mesh, P(*spec))

    param_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, abstract_params)

    # ФИКС (этот пасс -- диагностика сведена ТОЛЬКО к оптимизатору):
    # grad_group_map теперь строится через _optimizer_label_leaf (см.
    # make_grad_group_map выше), и добавлены _group_grad_norms_fn /
    # _group_weight_norms_fn -- по-группная L2-норма, не только флаг.
    grad_group_map = make_grad_group_map(abstract_params)
    _group_nonfinite_flags = build_group_nonfinite_flags(grad_group_map)
    _group_grad_norms_fn = build_group_norms_fn(grad_group_map, _DIAG_GROUPS)
    _group_weight_norms_fn = build_group_norms_fn(grad_group_map, _DIAG_GROUPS)
    print(f"[DIAG] По-оптимизаторская диагностика: группы {_DIAG_GROUPS} "
          f"(nonfinite-флаг + grad/weight L2-норма на группу, каждый эффективный шаг).")

    grad_layer_map = make_leaf_layer_map(abstract_params)
    _PARAM_LAYER_TAGS = param_layer_tags(grad_layer_map)
    _SOW_LAYER_TAGS = layer_tags_in_sow_order(config)
    _layer_grad_stats_fn = build_leaf_stats_fn(grad_layer_map, _PARAM_LAYER_TAGS)
    _layer_grad_raw_stats_fn = build_leaf_raw_stats_fn(grad_layer_map, _PARAM_LAYER_TAGS)
    _layer_weight_stats_fn = build_leaf_stats_fn(grad_layer_map, _PARAM_LAYER_TAGS)
    print(f"[DIAG] По-слойная диагностика: {len(_PARAM_LAYER_TAGS)} физических "
          f"тегов параметров ({_PARAM_LAYER_TAGS}), {len(_SOW_LAYER_TAGS)} sown-тегов активаций.")

    # ФИКС (локализация Muon orth_resid, см. модульный докстринг выше):
    # строим список путей muon-параметров ОДИН РАЗ здесь -- нужен train.py
    # для расшифровки worst_leaf_idx (число из jit-графа) в реальный путь.
    try:
        _whole_tree_label_fn = lambda p: jax.tree_util.tree_map_with_path(_optimizer_label_leaf, p)
        _MUON_LEAF_PATHS = build_muon_leaf_paths(abstract_params, _whole_tree_label_fn)
        print(f"[MUON-DIAG] Локализация включена: {len(_MUON_LEAF_PATHS)} muon-параметров "
              f"проиндексированы для расшифровки worst_leaf_idx каждый шаг.")
    except Exception as e:
        _MUON_LEAF_PATHS = None
        print(f"[MUON-DIAG] ⚠️ Не удалось построить _MUON_LEAF_PATHS ({e}) -- "
              f"worst_leaf_idx будет логироваться как число без расшифровки пути.")

    def _decay_scale_leaf(path, param):
        path_str = path_to_str(path)
        return 0.2 if ("decay_a" in path_str or "a_log" in path_str) else 1.0
    def _router_scale_leaf(path, param):
        path_str = path_to_str(path)
        return 0.3 if "router" in path_str else 1.0

    _router_grad_scale = jax.tree_util.tree_map_with_path(_router_scale_leaf, abstract_params)
    _decay_grad_scale = jax.tree_util.tree_map_with_path(_decay_scale_leaf, abstract_params)
    _router_temp_decay_map = jax.tree_util.tree_map_with_path(
        _router_temp_decay_leaf, abstract_params
    )
    _expert_bias_index_map = _build_expert_bias_index_map(abstract_params)

    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_shard_spec, opt_state_abstract)

    def model_apply_wrapped(variables, input_ids, rngs=None, deterministic=True, **kwargs):
        return model.apply(
            variables, input_ids,
            rngs=rngs, deterministic=deterministic,
            **kwargs
        )

    def distributed_train_step_micro(p, s, b, r, accum_grads, collinearity_coef=None):
        def loss_fn(param):
            return compute_loss(
                param, model_apply_wrapped, b, config,
                rngs={"dropout": r},
                deterministic=False, return_aux=True,
                ce_chunk_size=2048,
                collinearity_coef=collinearity_coef,
            )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        micro_grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        )
        new_accum = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)
        return p, s, new_accum, loss, aux_info, micro_grad_norm

    def distributed_apply_step(p, s, accum_grads, n_accum, assignment_frac_stacked=None):
        avg_grads = jax.tree_util.tree_map(lambda g: g / n_accum, accum_grads)

        group_nonfinite_flags = _group_nonfinite_flags(avg_grads)
        group_grad_norms = _group_grad_norms_fn(avg_grads)
        layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite = _layer_grad_stats_fn(avg_grads)
        # ФИКС (диагностика замаскированных non-finite, см. чат): существующий
        # build_leaf_stats_fn применяет nan_to_num ДО вычисления normы/maxabs
        # -- это маскирует реальную величину NaN/inf-выброса, из-за чего
        # layer_grad_norm выглядел "здоровым" (0.1-0.3) даже на шагах, где
        # nonfinite-флаг для этого же слоя сработал. Отдельная функция считает
        # maxabs ТОЛЬКО по конечной части + отдельно количество non-finite
        # элементов -- различает "один залётный NaN" от "массового обвала".
                # ФИКС: сырая диагностика ДО nan_to_num'а внутри avg_grads (см.
        # чуть ниже -- avg_grads реально санитизируется через несколько
        # строк, здесь мы ещё смотрим на "живые" значения ПОСЛЕ деления
        # на n_accum, но ДО принудительной замены NaN на 0).
        layer_grad_raw_maxabs, layer_grad_nonfinite_count = _layer_grad_raw_stats_fn(avg_grads)
        global_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(avg_grads)))
        is_finite = jnp.isfinite(global_norm)
        safe_norm = jnp.where(is_finite, global_norm, 1.0)
        clip_factor = jnp.where(is_finite, jnp.minimum(1.0, 1.0 / (safe_norm + 1e-6)), 0.0)

        avg_grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
            avg_grads,
        )
        avg_grads = jax.tree_util.tree_map(lambda g, s: g * s, avg_grads, _decay_grad_scale)
        avg_grads = jax.tree_util.tree_map(lambda g, s: g * s, avg_grads, _router_grad_scale)

        updates, new_s = tx.update(avg_grads, s, p)
        new_p = optax.apply_updates(p, updates)

        new_p = jax.tree_util.tree_map(
            lambda pp: jnp.nan_to_num(jnp.clip(pp, -1e2, 1e2), nan=0.0, posinf=1e2, neginf=-1e2),
            new_p,
        )

        new_p = apply_router_temp_decay(new_p, _router_temp_decay_map)
        new_p = apply_expert_bias_update(new_p, _expert_bias_index_map, assignment_frac_stacked)

        layer_w_norms, layer_w_maxabs, layer_w_nonfinite = _layer_weight_stats_fn(new_p)
        group_weight_norms = _group_weight_norms_fn(new_p)

        # ФИКС (локализация, см. модульный докстринг выше):
        # extract_muon_diagnostics теперь возвращает ПАРУ.
        muon_orth_resid, muon_worst_leaf_idx, muon_worst_leaf_grad_norm, muon_worst_leaf_grad_maxabs, muon_mean_orth_resid = extract_muon_diagnostics(new_s)

        was_clipped = jnp.any(jnp.stack([
            jnp.any(jnp.abs(leaf) >= 1e2) for leaf in jax.tree_util.tree_leaves(p)
        ]))

        zclip_diag = extract_zclip_diagnostics(new_s)

        zero_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return (new_p, new_s, zero_accum, is_finite,
                global_norm, clip_factor, group_nonfinite_flags, was_clipped,
                zclip_diag,
                layer_grad_norms, layer_grad_maxabs, layer_grad_nonfinite,
                layer_w_norms, layer_w_maxabs, layer_w_nonfinite,
                muon_orth_resid, muon_worst_leaf_idx,
                group_grad_norms, group_weight_norms,
                muon_worst_leaf_grad_norm, muon_worst_leaf_grad_maxabs, 
                muon_mean_orth_resid,
                layer_grad_raw_maxabs, layer_grad_nonfinite_count)   
    def distributed_val_step(p, b):
        return compute_loss(
            p, model_apply_wrapped, b, config,
            rngs=None,
            deterministic=True,
        )

    # ФИКС (этот пасс -- критический краш компиляции, см. модульный
    # докстринг выше): убраны ~30 ключей для kernel/activation-level
    # sow()-значений, которых compute_loss() БОЛЬШЕ НЕ возвращает (эти
    # self.sow(...) удалены из model.py). Оставлены только ключи, которые
    # compute_loss реально кладёт в aux_info при return_aux=True (см.
    # optimizer.py's compute_loss -- финальный `aux_info = {...}` словарь).
    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None)),
        "moe_dropped_ratio": NamedSharding(mesh, P(None)),
        "router_temp": NamedSharding(mesh, P(None)),
        "min_col_norm": NamedSharding(mesh, P(None)),
        "max_abs_logit_preclip": NamedSharding(mesh, P(None)),
        "norm_x_mean": NamedSharding(mesh, P(None)),
        "norm_x_max": NamedSharding(mesh, P(None)),
        "norm_x_min": NamedSharding(mesh, P(None)),
        "router_max_cos_per_layer": NamedSharding(mesh, P()),
        "router_max_cos": NamedSharding(mesh, P()),
        "assignment_frac": NamedSharding(mesh, P(None, None)),
        "moe_min_group_size": NamedSharding(mesh, P(None)),
        "moe_max_group_size": NamedSharding(mesh, P(None)),
    }
    compiled_train_micro = jax.jit(
        distributed_train_step_micro,
        donate_argnums=(0, 1, 4),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
            param_sharding,
            NamedSharding(mesh, P()),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),
            aux_info_sharding,
            NamedSharding(mesh, P()),
        ),
    )

    zclip_diag_sharding = {
        "ema_mean": NamedSharding(mesh, P()),
        "ema_var": NamedSharding(mesh, P()),
        "warm_count": NamedSharding(mesh, P()),
        "slow_ema_mean": NamedSharding(mesh, P()),
        "slow_warm_count": NamedSharding(mesh, P()),
    }

    compiled_apply = jax.jit(
        distributed_apply_step,
        donate_argnums=(0, 1, 2),
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),
            NamedSharding(mesh, P(None, None)),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            param_sharding,
            NamedSharding(mesh, P()),        # is_finite
            NamedSharding(mesh, P()),        # global_norm
            NamedSharding(mesh, P()),        # clip_factor
            NamedSharding(mesh, P(None)),    # group_nonfinite_flags
            NamedSharding(mesh, P()),        # was_clipped
            zclip_diag_sharding,
            NamedSharding(mesh, P(None)),    # layer_grad_norms
            NamedSharding(mesh, P(None)),    # layer_grad_maxabs
            NamedSharding(mesh, P(None)),    # layer_grad_nonfinite
            NamedSharding(mesh, P(None)),    # layer_w_norms
            NamedSharding(mesh, P(None)),    # layer_w_maxabs
            NamedSharding(mesh, P(None)),    # layer_w_nonfinite
            NamedSharding(mesh, P()),        # muon_orth_resid
            NamedSharding(mesh, P()),        # muon_worst_leaf_idx
            NamedSharding(mesh, P(None)),    # group_grad_norms    <-- НОВОЕ
            NamedSharding(mesh, P(None)),    # group_weight_norms  <-- НОВОЕ
            NamedSharding(mesh, P()),        # muon_worst_leaf_grad_norm    <-- НОВОЕ
            NamedSharding(mesh, P()),        # muon_worst_leaf_grad_maxabs  <-- НОВОЕ
            NamedSharding(mesh, P()),        # muon_mean_orth_resid   <-- НОВОЕ
            NamedSharding(mesh, P(None)),    # layer_grad_raw_maxabs      <-- НОВОЕ
            NamedSharding(mesh, P(None)),    # layer_grad_nonfinite_count <-- НОВОЕ

        ),
    )

    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )

    return (compiled_train_micro, compiled_apply, compiled_val, mesh, tx, model,
            param_sharding, opt_state_sharding, data_sharding, lr_schedule)


def resolve_source_files(output_dir, prefix):
    merged_ids = os.path.join(output_dir, f"{prefix}_input_ids.npy")
    merged_lbls = os.path.join(output_dir, f"{prefix}_labels.npy")
    if os.path.exists(merged_ids) and os.path.exists(merged_lbls):
        return [(merged_ids, merged_lbls)]

    shard_ids_paths = sorted(
        glob.glob(os.path.join(output_dir, f"{prefix}_shard_ids_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    pairs = []
    for ids_path in shard_ids_paths:
        lbls_path = ids_path.replace("_shard_ids_", "_shard_lbls_")
        if os.path.exists(lbls_path):
            pairs.append((ids_path, lbls_path))
    if not pairs:
        raise FileNotFoundError(
            f"Не найдены файлы для prefix={prefix!r} в {output_dir}."
        )
    return pairs


def build_manifest(file_pairs):
    manifest = []
    total = 0
    for entry in file_pairs:
        ids_path, lbls_path = entry[0], entry[1]
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, seq_len, val_split=0.05,
                             dataset_fraction=1.0, fraction_seed=777, skip_batches=0,
                             mode="mixed"):
    normalized_pairs = []
    for entry in file_pairs:
        if len(entry) == 3:
            ids_path, lbls_path, frac = entry
        else:
            ids_path, lbls_path = entry
            frac = 1.0
        normalized_pairs.append((ids_path, lbls_path, float(frac)))

    manifest = build_manifest([(p[0], p[1]) for p in normalized_pairs])
    sizes = np.array([n for _, _, n in manifest])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    context_length = np.load(manifest[0][0], mmap_mode="r").shape[1]
    if context_length > seq_len:
        context_length = seq_len

    mmap_cache = {}

    def _get_mmap(path):
        arr = mmap_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            mmap_cache[path] = arr
        return arr

    def _gather_batch(global_indices):
        shard_of = np.searchsorted(offsets, global_indices, side="right") - 1
        ids_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        lbls_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        for s in np.unique(shard_of):
            m = shard_of == s
            local_idx = global_indices[m] - offsets[s]
            ids_path, lbls_path, _ = manifest[s]
            ids_full = _get_mmap(ids_path)[local_idx]
            lbls_full = _get_mmap(lbls_path)[local_idx]
            ids_out[m] = ids_full[:, :seq_len]
            lbls_out[m] = lbls_full[:, :seq_len]
        return ids_out, lbls_out

    frac_rng_per_source = np.random.RandomState(fraction_seed)
    per_source_idx = []
    for s, (ids_path, _, n) in enumerate(manifest):
        local_idx = np.arange(offsets[s], offsets[s + 1])
        frac = normalized_pairs[s][2]
        if frac < 1.0:
            n_keep = max(1, int(len(local_idx) * frac))
            local_idx = frac_rng_per_source.choice(local_idx, size=n_keep, replace=False)
            local_idx.sort()
        if frac != 1.0:
            print(f"[DATA] Источник {os.path.basename(ids_path)}: "
                  f"{frac*100:.0f}% -> {len(local_idx):,} из {int(n):,} блоков")
        per_source_idx.append(local_idx)

    all_idx = np.concatenate(per_source_idx)

    if dataset_fraction < 1.0:
        frac_rng = np.random.RandomState(fraction_seed)
        n_keep = int(len(all_idx) * dataset_fraction)
        all_idx = frac_rng.choice(all_idx, size=n_keep, replace=False)
        all_idx.sort()
        print(f"[DATA] Общая подвыборка {dataset_fraction*100:.0f}%: {n_keep:,} блоков.")

    pool_size = len(all_idx)
    val_size = int(pool_size * val_split)
    train_size = pool_size - val_size

    split_rng = np.random.RandomState(42)
    shuffled = np.copy(all_idx)
    split_rng.shuffle(shuffled)
    train_idx_pool = shuffled[:train_size]
    val_idx_pool = shuffled[train_size:]

    train_pool_by_source = None
    if mode in ("sequential", "round_robin"):
        val_set = set(val_idx_pool.tolist())
        train_pool_by_source = []
        for s in range(len(manifest)):
            src_train_idx = np.array(
                [i for i in per_source_idx[s] if i not in val_set], dtype=np.int64
            )
            train_pool_by_source.append(src_train_idx)
            print(f"[DATA] (mode={mode}) источник #{s} ({os.path.basename(manifest[s][0])}): "
                  f"{len(src_train_idx):,} train-блоков")

    def _infinite_source_batches(src_idx, seed):
        local_rng = np.random.RandomState(seed)
        idx_local = np.copy(src_idx)
        while True:
            local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            for step in range(n_steps):
                batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
                yield _gather_batch(batch_idx)

    def _infinite_source_indices(src_idx, seed):
        local_rng = np.random.RandomState(seed)
        idx_local = np.copy(src_idx)
        while True:
            local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            for step in range(n_steps):
                yield idx_local[step * batch_size:(step + 1) * batch_size]

    def _round_robin_gen(pool_by_source, skip_first=0):
        src_gens = [_infinite_source_indices(pool_by_source[s], seed=1000 + s)
                    for s in range(len(pool_by_source))]
        n_sources = len(src_gens)
        step_i = 0
        if skip_first > 0:
            print(f"[DATA] (round_robin) Быстрый пропуск {skip_first} микрошагов "
                  f"(без чтения с диска)...")
        while True:
            s = step_i % n_sources
            batch_idx = next(src_gens[s])
            step_i += 1
            if step_i <= skip_first:
                continue
            ids_np, lbls_np = _gather_batch(batch_idx)
            yield {
                "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                "_source_idx": s,
            }

    def _sequential_gen(pool_by_source, skip_first=0):
        local_rng = np.random.RandomState(123)
        skip_remaining = skip_first
        first_pass = True
        while True:
            for s, src_idx in enumerate(pool_by_source):
                idx_local = np.copy(src_idx)
                local_rng.shuffle(idx_local)
                n_steps = len(idx_local) // batch_size
                start_step = 0
                if first_pass and skip_remaining > 0:
                    start_step = min(skip_remaining, n_steps)
                    skip_remaining -= start_step
                    if start_step > 0:
                        print(f"[DATA] Resume (sequential): пропускаем {start_step} "
                              f"микрошагов источника #{s}.")
                for step in range(start_step, n_steps):
                    batch_idx = idx_local[step * batch_size:(step + 1) * batch_size]
                    ids_np, lbls_np = _gather_batch(batch_idx)
                    yield {
                        "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                        "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                        "_source_idx": s,
                    }
            first_pass = False

    def _mixed_gen(pool, is_train=True, skip_first=0):
        idx_local = np.copy(pool)
        local_rng = np.random.RandomState(123)
        first_pass = True
        while True:
            if is_train:
                local_rng.shuffle(idx_local)
            n_steps = len(idx_local) // batch_size
            start_step = 0
            if first_pass and is_train:
                start_step = skip_first % max(n_steps, 1)
                if start_step > 0:
                    print(f"[DATA] Resume: пропускаем первые {start_step} микрошагов "
                          f"текущего прохода датасета.")
            first_pass = False
            for step in range(start_step, n_steps):
                batch_idx = idx_local[step * batch_size: (step + 1) * batch_size]
                ids_np, lbls_np = _gather_batch(batch_idx)
                yield {
                    "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                    "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                }
            if not is_train:
                break

    if mode == "round_robin":
        train_gen = _round_robin_gen(train_pool_by_source, skip_first=skip_batches)
    elif mode == "sequential":
        train_gen = _sequential_gen(train_pool_by_source, skip_first=skip_batches)
    elif mode == "mixed":
        train_gen = _mixed_gen(train_idx_pool, True, skip_first=skip_batches)
    else:
        raise ValueError(f"Неизвестный mode={mode!r}.")

    return (
        train_gen,
        lambda: _mixed_gen(val_idx_pool, False),
        train_size // batch_size,
        val_size // batch_size,
    )
