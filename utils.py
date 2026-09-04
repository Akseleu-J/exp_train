def path_to_str(path) -> str:
    """Turn a jax.tree_util key-path into a lowercase string for substring matching.

    `jax.tree_util.tree_map_with_path` hands back a tuple of key objects whose type
    depends on the container at that point in the pytree:
      - DictKey            (dict / FrozenDict entries)   -> has `.key`
      - FlattenedIndexKey   (flattened containers)         -> has `.key`
      - GetAttrKey          (NamedTuple fields)             -> has `.name`
      - SequenceKey         (list / plain tuple entries)   -> has `.idx`

    A flax params pytree is dict-only, so `str(p.key)` alone happens to work there.
    An optax optimizer-state pytree (from `optax.chain`/`multi_transform`/NamedTuple
    states like `MuonState`) mixes in GetAttrKey and SequenceKey entries, and `p.key`
    raises `AttributeError: 'SequenceKey' object has no attribute 'key'` on those.
    This checks all the attribute names generically instead of assuming `.key`.
    """
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts).lower()
    
def collect_by_leaf_name(tree, target_name):
    import jax

    collected = []
    target = target_name.lower()

    def _mark(path, leaf):
        if not path:
            return leaf
        last = path[-1]
        # ФИКС: flax's sow() по умолчанию оборачивает значение в 1-tuple
        # (см. flax docs: state['intermediates'] == {'h': (...,)}) --
        # реальный скалярный лист лежит НА УРОВЕНЬ ГЛУБЖЕ, путь до него
        # заканчивается на SequenceKey/int-индекс кортежа, а не на имени
        # sown-переменной. Старая проверка `path[-1] == target_name`
        # поэтому НИКОГДА не совпадала для sown-значений aux_loss/z_loss/
        # expert_utilization/moe_dropped_ratio -- все они молча собирались
        # как пустые списки с самого первого dense-MoE прогона (виден в
        # логе как aux=0.0000/z=0.00000 на каждом шаге, включая шаг 1530).
        # Если последний сегмент -- числовой индекс (кортеж), берём ИМЯ
        # на уровень выше вместо него.
        if hasattr(last, "idx"):
            name_segment = path[-2] if len(path) >= 2 else None
        else:
            name_segment = last
        if name_segment is not None and path_to_str([name_segment]) == target:
            collected.append(leaf)
        return leaf

    jax.tree_util.tree_map_with_path(_mark, tree)
    return collected
