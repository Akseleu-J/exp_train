"""
muon_optax.py -- Muon (MomentUm Orthogonalized by Newton-schulz) как
optax.GradientTransformation, плюс гибрид Muon (для 2D-матриц) + AdamW
(для остального -- эмбеддинги, биасы, скаляры вроде decay_a, RMSNorm-scale).


ВАЖНО (numerical honesty, в духе вашего "honest results above all"):
Newton-Schulz(5) НЕ гарантированно сходится к точной ортогонализации на
few шагах для плохо обусловленных/близких к сингулярным матриц -- то же
самое ограничение, что вы уже диагностировали как "Muon NS(5)
non-convergence... высокий orth_resid на ранних шагах" в основной
архитектуре. Здесь НЕТ отдельного orth_resid-диагностики (loop_params
маленький, добавлять её пока не обязательно) -- если Muon-прогон
разойдётся, ПЕРВОЕ, что проверить -- не то же ли самое NS(5)
non-convergence, что вы уже видели, а не что-то новое.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from utils import path_to_str


def _newton_schulz5(G: jnp.ndarray, ns_steps: int = 5, eps: float = 1e-7) -> jnp.ndarray:
    """Квинтичная итерация Ньютона-Шульца (Keller Jordan's Muon).
    G: 2D матрица. Возвращает ортогонализованную (semi-orthogonal) матрицу
    того же размера. Работает в float32 независимо от входного dtype --
    NS расходится быстрее в bf16/fp16."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(jnp.float32)
    X = X / (jnp.linalg.norm(X) + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(ns_steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class MuonState(NamedTuple):
    momentum_buf: any


def muon(learning_rate: float, momentum: float = 0.95, ns_steps: int = 5,
         nesterov: bool = True, weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Muon для ПОЛНОСТЬЮ 2D-параметров. Не проверяет ndim сама -- вызывающий
    код (see make_hybrid_muon_adamw ниже) обязан маршрутизировать сюда
    ТОЛЬКО 2D-листья через optax.multi_transform; на не-2D листе упадёт с
    AssertionError внутри _newton_schulz5 (сделано намеренно, чтобы не
    молча деградировать до чего-то другого при неправильной маршрутизации)."""

    def init_fn(params):
        return MuonState(momentum_buf=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        def per_leaf(g, buf, p):
            new_buf = momentum * buf + g
            g_for_ns = g + momentum * new_buf if nesterov else new_buf
            ortho = _newton_schulz5(g_for_ns, ns_steps=ns_steps)
            # Стандартный Muon-скейлинг: sqrt(max(fan_in, fan_out)) --
            # держит эффективный update-масштаб независимым от формы матрицы.
            scale = jnp.sqrt(jnp.asarray(max(g.shape[-2], g.shape[-1]), dtype=jnp.float32))
            step = -learning_rate * ortho.astype(g.dtype) * scale
            if weight_decay > 0.0 and p is not None:
                step = step - learning_rate * weight_decay * p
            return step, new_buf

        if params is None:
            params_tree = jax.tree_util.tree_map(lambda g: None, updates)
        else:
            params_tree = params

        flat_g, treedef = jax.tree_util.tree_flatten(updates)
        flat_buf = treedef.flatten_up_to(state.momentum_buf)
        flat_p = treedef.flatten_up_to(params_tree)

        flat_out_step, flat_out_buf = [], []
        for g, buf, p in zip(flat_g, flat_buf, flat_p):
            s, b = per_leaf(g, buf, p)
            flat_out_step.append(s)
            flat_out_buf.append(b)

        new_updates = jax.tree_util.tree_unflatten(treedef, flat_out_step)
        new_buf = jax.tree_util.tree_unflatten(treedef, flat_out_buf)
        return new_updates, MuonState(momentum_buf=new_buf)

    return optax.GradientTransformation(init_fn, update_fn)


def _param_label(path, leaf) -> str:
    """Маршрутизация: 2D+ матрицы (кроме эмбеддингов/lm_head) -> 'muon',
    всё остальное (биасы, 1D-векторы вроде decay_a, RMSNorm-scale,
    эмбеддинги, lm_head) -> 'adam'. Эмбеддинги исключены из Muon
    намеренно -- стандартная практика (Muon разработан для hidden-to-hidden
    матриц с фиксированной семантикой строк/столбцов; эмбеддинг-таблица со
    строками-токенами имеет другую структуру, и ортогонализация по ней
    менее обоснована и хуже документирована)."""
    p_str = path_to_str(path)
    if "embed" in p_str or "lm_head" in p_str:
        return "adam"
    if hasattr(leaf, "ndim") and leaf.ndim == 2 and min(leaf.shape) >= 2:
        return "muon"
    return "adam"


def make_hybrid_muon_adamw(params_for_labels, muon_lr: float, adam_lr: float,
                            weight_decay: float = 0.0, muon_momentum: float = 0.95,
                            muon_ns_steps: int = 5, grad_clip_norm: float = 1.0):
    """Возвращает (tx, label_tree). label_tree нужен ТОЛЬКО для
    диагностики/логирования (сколько параметров ушло в какую группу) --
    сам tx уже содержит маршрутизацию через optax.multi_transform."""
    label_tree = jax.tree_util.tree_map_with_path(_param_label, params_for_labels)

    muon_tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        muon(learning_rate=muon_lr, momentum=muon_momentum, ns_steps=muon_ns_steps,
             weight_decay=weight_decay),
    )
    adam_tx = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=adam_lr, weight_decay=weight_decay),
    )

    tx = optax.multi_transform({"muon": muon_tx, "adam": adam_tx}, label_tree)

    n_muon = sum(1 for t in jax.tree_util.tree_leaves(label_tree) if t == "muon")
    n_adam = sum(1 for t in jax.tree_util.tree_leaves(label_tree) if t == "adam")
    print(f"[MUON] Маршрутизация параметров: muon={n_muon} листьев, adam={n_adam} листьев")

    return tx, label_tree
