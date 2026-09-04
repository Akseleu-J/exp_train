"""
diagnostics.py -- ВСЕГДА включённая (без env-флага), дешёвая, ПО-СЛОЙНАЯ
диагностика обучения.

Зачем этот файл существует (см. чат): существующая диагностика в
train_setup.py (_classify_leaf_group/_DIAG_GROUPS) агрегирует ВСЕ 16+
слоёв gdn2 (и отдельно 2 mamba2, 3 mla) в ОДНУ группу -- "не-finite где-то
в gdn2" никогда не говорит, в КАКОМ ИМЕННО из 16 слоёв. Плюс единственная
метрика, которая реально льётся в W&B каждый шаг -- один агрегированный
global_grad_norm, который дёргается уже постфактум. Плюс вся
GDN2_FWD_DIAG-диагностика в kernel_d_pipeline.py/model.py идёт через
jax.debug.print внутри jax.lax.cond -- host-callback на каждый шаг на
каждый слой, и её саму пришлось чинить от eager/jit-несовместимости (см.
model.py's _is_traced).

Этот файл делает то же самое, что group_nonfinite_flags/was_clipped/
zclip_diag в train_setup.py уже делают правильно: ЧИСТЫЕ функции,
возвращающие обычные jnp-массивы как output jitted-шага, БЕЗ
debug.print/callback -- дёшево (несколько reduce_sum/reduce_max поверх
того, что и так уже вычислено), безопасно под jit, и с гранулярностью
"один тег на физический (block_idx, layer_idx) слой", а не "один тег на
архитектурный ТИП слоя".
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp

from utils import path_to_str


def classify_leaf_layer_tag(path_str: str) -> str:
    """Гранулярнее, чем train_setup.py's _classify_leaf_group:
    'b{block}_l{layer}' вместо 'gdn2'/'mamba2'/'mla' одной кучей на все
    слои этого типа сразу."""
    m = re.search(r"block_(\d+)/layer_(\d+)", path_str)
    if m:
        return f"b{m.group(1)}_l{m.group(2)}"
    if "experts_block" in path_str or "moe" in path_str or "router" in path_str:
        return "moe"
    if "embed" in path_str or "lm_head" in path_str:
        return "embed"
    return "other"


def make_leaf_layer_map(params):
    """pytree той же листовой структуры, что params -- каждый лист
    заменён на строковый тег его физического слоя."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _: classify_leaf_layer_tag(path_to_str(path)), params
    )


def param_layer_tags(leaf_layer_map):
    """Отсортированный список уникальных тегов, реально присутствующих в
    leaf_layer_map -- используется и как порядок build_leaf_stats_fn, и
    как подписи столбцов в train.py."""
    leaves = jax.tree_util.tree_leaves(leaf_layer_map)
    return sorted(set(leaves))


def layer_tags_in_sow_order(cfg):
    """Порядок тегов, СОВПАДАЮЩИЙ с порядком, в котором model.py's
    BlockDAR/BlockDARLayer реально sow'ят layer_delta_maxabs/
    layer_resid_maxabs (block-major, layer-minor -- см. BlockDAR's
    for-loop по layers_per_block внутри одного __call__, и порядок самих
    block_N в FullHybridMoEModel) -- нужен train.py, чтобы подписать
    столбцы sown-диагностики правильными именами слоёв."""
    tags = []
    n_blocks = cfg.num_layers // cfg.layers_per_block
    for block_idx in range(n_blocks):
        for i in range(cfg.layers_per_block):
            layer_idx = block_idx * cfg.layers_per_block + i
            tags.append(f"b{block_idx}_l{layer_idx}_{cfg.layer_types[layer_idx]}")
    return tags


def build_leaf_stats_fn(leaf_tag_map, tags):
    """Возвращает ЧИСТУЮ функцию (tree такой же листовой структуры, что
    leaf_tag_map) -> (norms, maxabs, any_nonfinite), каждый формы
    (len(tags),). Работает и для grads, и для params -- любой pytree с той
    же листовой структурой, что leaf_tag_map (используется дважды в
    train_setup.py: один раз на avg_grads, один раз на new_p)."""
    leaves_tag, _ = jax.tree_util.tree_flatten(leaf_tag_map)
    idx_by_tag = {t: [i for i, tt in enumerate(leaves_tag) if tt == t] for t in tags}

    def _stats(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        norms, maxabs, nonfinite = [], [], []
        for tag in tags:
            idxs = idx_by_tag[tag]
            if not idxs:
                norms.append(jnp.array(0.0, dtype=jnp.float32))
                maxabs.append(jnp.array(0.0, dtype=jnp.float32))
                nonfinite.append(jnp.array(False))
                continue
            raws = [leaves[i].astype(jnp.float32) for i in idxs]
            any_nf = jnp.any(jnp.stack([jnp.any(jnp.logical_not(jnp.isfinite(r))) for r in raws]))
            safe = [jnp.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0) for r in raws]
            sq = sum(jnp.sum(jnp.square(s)) for s in safe)
            mx = jnp.max(jnp.stack([jnp.max(jnp.abs(s)) for s in safe]))
            norms.append(jnp.sqrt(sq))
            maxabs.append(mx)
            nonfinite.append(any_nf)
        return jnp.stack(norms), jnp.stack(maxabs), jnp.stack(nonfinite)

    return _stats
def build_leaf_raw_stats_fn(leaf_tag_map, tags):
    """Как build_leaf_stats_fn, но maxabs/nonfinite_count считаются на
    СЫРЫХ значениях (без nan_to_num до подсчёта) -- существующий
    build_leaf_stats_fn санитизирует ДО вычисления normы/maxabs, что
    маскирует реальную величину NaN/inf-выброса (см. чат: "здоровые"
    normы 0.1-0.3 на шагах, где nonfinite-флаг сработал). Здесь maxabs
    считается ТОЛЬКО по конечной части (как kernel_d_pipeline._stage_diag),
    плюс отдельно -- сколько именно элементов non-finite, чтобы отличить
    "один залётный NaN" от "массового обвала"."""
    leaves_tag, _ = jax.tree_util.tree_flatten(leaf_tag_map)
    idx_by_tag = {t: [i for i, tt in enumerate(leaves_tag) if tt == t] for t in tags}

    def _stats(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        raw_maxabs, nonfinite_count = [], []
        for tag in tags:
            idxs = idx_by_tag[tag]
            if not idxs:
                raw_maxabs.append(jnp.array(0.0, dtype=jnp.float32))
                nonfinite_count.append(jnp.array(0, dtype=jnp.int32))
                continue
            raws = [leaves[i].astype(jnp.float32) for i in idxs]
            finite_masks = [jnp.isfinite(r) for r in raws]
            n_nf = sum(jnp.sum(jnp.logical_not(m)).astype(jnp.int32) for m in finite_masks)
            finite_only = [jnp.where(m, r, 0.0) for r, m in zip(raws, finite_masks)]
            mx = jnp.max(jnp.stack([jnp.max(jnp.abs(f)) for f in finite_only]))
            raw_maxabs.append(mx)
            nonfinite_count.append(n_nf)
        return jnp.stack(raw_maxabs), jnp.stack(nonfinite_count)

    return _stats


class HostLayerBurstTracker:
    """Host-side (Python, вне jit) EMA-трекер ПО КАЖДОМУ тегу отдельно --
    аналог train.py's существующего burst_streak (global_norm > 8.0 три
    шага подряд), но ЛОКАЛИЗОВАННЫЙ до ОДНОГО конкретного физического
    слоя вместо агрегата. Срабатывает, как только КОНКРЕТНЫЙ слой
    отклоняется от СВОЕЙ ЖЕ EMA сильнее, чем threshold_ratio, ДВА шага
    подряд (фильтр разового шумового выброса) -- это ловит начало
    локального разгона в одном месте модели РАНЬШЕ, чем это дойдёт до
    общего global_norm/burst_streak, и говорит, В КАКОМ ИМЕННО слое."""

    def __init__(self, tags, threshold_ratio=4.0, decay=0.98, min_ema=1e-6, min_streak=2):
        self.tags = list(tags)
        self.threshold_ratio = threshold_ratio
        self.decay = decay
        self.min_ema = min_ema
        self.min_streak = min_streak
        self._ema = {t: None for t in self.tags}
        self._streak = {t: 0 for t in self.tags}

    def update(self, values: dict):
        """values: dict tag -> float (уже device_get'нутое значение,
        обычно maxabs или norm за этот шаг). Возвращает список
        (tag, prev_ema, new_val, streak) для тегов, чей streak >=
        self.min_streak на этом вызове."""
        triggered = []
        for t in self.tags:
            v = float(values.get(t, 0.0))
            prev = self._ema[t]
            if prev is None:
                self._ema[t] = v
                continue
            ratio = v / max(prev, self.min_ema)
            if ratio > self.threshold_ratio:
                self._streak[t] += 1
            else:
                self._streak[t] = 0
            self._ema[t] = self.decay * prev + (1.0 - self.decay) * v
            if self._streak[t] >= self.min_streak:
                triggered.append((t, prev, v, self._streak[t]))
        return triggered

    def snapshot_state(self):
        return dict(self._ema)
