"""
model_gdn2_100m_tied.py -- вариант model_gdn2_100m.py с опцией ПОЛНОГО
weight-tying: вместо num_blocks РАЗНЫХ BlockDAR (свои веса на каждый
block_idx) -- ОДИН экземпляр BlockDAR, применённый R раз подряд к
current_x/history_blocks. Используется для проверки двух гипотез (см. чат):

  H2 (recurrent depth = дешёвая эффективная глубина):
      tied-модель с R применениями ОДНОГО блока против untied-модели с
      R разными блоками при ОДИНАКОВОМ compute (том же числе forward-
      применений GDN2-сублеера) -- сравниваем итоговый bpb.

  H1 (эмерджентная итеративная прогрессия без явного gold-step лосса):
      обучаем tied-модель ТОЛЬКО на финальном таргете (обычный LM loss на
      выходе после R-го применения) -- смотрим, убывает ли bpb(r), если
      "подсмотреть" через readout после КАЖДОГО r (см. eval_tied_probe.py).

НИЧЕГО в самой математике BlockDAR/BlockDARLayer/GatedDeltaNet2J не
меняется -- этот файл только меняет, СКОЛЬКО РАЗНЫХ nn.Module-экземпляров
создаётся в __call__ верхнего уровня (flax module reuse = tied weights
"бесплатно", т.к. flax разделяет параметры по имени модуля/vars scope, а
не по count вызовов).

ВАЖНО про historyблок при tying: BlockDAR уже накапливает history_blocks
(конкатенация block_delta с каждого прохода) и использует их через
HybridDARAttention -- это УЖЕ механизм доступа к "прошлым состояниям",
концептуально близкий к вашему отдельному FeedbackCompressorReal из
latent_reasoning_gdn2.py. При полном tying это ЕСТЬ ваш reasoning loop,
без отдельной надстройки -- поэтому это самый дешёвый первый эксперимент.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

# Переиспользуем ВСЁ как есть -- никаких переопределений математики.
from model_gdn2_100m import (
    ModelConfig as _BaseModelConfig,
    GatedDeltaNet2J, BlockDARLayer, BlockDAR, IntraBlockAttention, HybridDARAttention,
    make_grad_sanitizer, set_model_mesh, get_model_mesh, get_batch_axis,
    count_params, GDN2_PALLAS_BT,
)


@struct.dataclass
class TiedModelConfig(_BaseModelConfig):
    """Те же поля, что ModelConfig, плюс переключатели tying/R.

    tie_all_blocks=False  -> ведёт себя ИДЕНТИЧНО FullGDN2BlockDeltaModel
                              (num_layers//layers_per_block РАЗНЫХ блоков).
    tie_all_blocks=True   -> ОДИН блок, применённый n_reasoning_cycles раз.
                              num_layers в этом режиме используется ТОЛЬКО
                              для расчёта layers_per_block-внутренней
                              структуры одного блока (сколько GDN2-сублееров
                              внутри одного tied-блока), НЕ для подсчёта
                              числа блоков -- число повторов задаёт
                              n_reasoning_cycles отдельно, чтобы можно было
                              менять R без изменения layers_per_block.
    """
    tie_all_blocks: bool = False
    n_reasoning_cycles: int = 4


class FullGDN2BlockDeltaModelTied(nn.Module):
    """Как FullGDN2BlockDeltaModel, но с опцией tie_all_blocks.

    compute-эквивалентность для H2: если untied-baseline имеет
    num_blocks_baseline = num_layers // layers_per_block блоков, то для
    честного compute-match запускайте tied-версию с
    n_reasoning_cycles == num_blocks_baseline -- тогда число forward-
    применений GDN2 (и, следовательно, число Pallas-kernel вызовов на
    forward pass) СОВПАДАЕТ, различается только число УНИКАЛЬНЫХ
    параметров (в num_blocks_baseline раз меньше у tied-версии).
    """
    cfg: TiedModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None,
                 return_hidden: bool = False, return_all_r_states: bool = False):
        b, l = input_ids.shape
        embed_layer = nn.Embed(
            num_embeddings=self.cfg.vocab_size, features=self.cfg.d_model,
            name="embed", dtype=jnp.bfloat16,
        )
        x = embed_layer(input_ids)
        x = make_grad_sanitizer("embed_input_lookup", clip_val=1e3)(x)

        history_blocks = jnp.zeros((0, b, l, self.cfg.d_model), dtype=x.dtype)

        RematBlock = nn.remat(BlockDAR, static_argnums=())
        r_states = [] if return_all_r_states else None

        if self.cfg.tie_all_blocks:
            # ОДИН экземпляр -- flax разделяет параметры по name="shared_block"
            # на каждой итерации python-цикла ниже (тот же nn.Module vars
            # scope переиспользуется, а не создаётся заново).
            shared_block = RematBlock(cfg=self.cfg, block_idx=0, layer_idx_start=0, name="shared_block")
            for _r in range(self.cfg.n_reasoning_cycles):
                x, history_blocks = shared_block(x, history_blocks)
                if return_all_r_states:
                    r_states.append(x)
        else:
            num_blocks = self.cfg.num_layers // self.cfg.layers_per_block
            for block_idx in range(num_blocks):
                layer_idx_start = block_idx * self.cfg.layers_per_block
                x, history_blocks = RematBlock(
                    cfg=self.cfg, block_idx=block_idx, layer_idx_start=layer_idx_start,
                    name=f"block_{block_idx}",
                )(x, history_blocks)
                if return_all_r_states:
                    r_states.append(x)

        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x).astype(x.dtype)

        if return_hidden and not return_all_r_states:
            return final

        def _readout(h):
            h_normed = nn.RMSNorm(epsilon=1e-6, name="final_norm")(h).astype(h.dtype)
            if self.cfg.tie_embeddings:
                return embed_layer.attend(h_normed)
            return nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head",
                             dtype=jnp.bfloat16)(h_normed)

        if return_all_r_states:
            # ВНИМАНИЕ: используется ТОЛЬКО в eval/probe-скриптах на малых
            # batch/seq (см. eval_tied_probe.py's chunked readout) --
            # считать логиты для ВСЕХ r разом на полном (b,l,vocab) ведёт
            # к тому же OOM-паттерну, что описан в latent_reasoning_gdn2.py.
            # Здесь возвращаем states, а не logits -- readout снаружи.
            return r_states

        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head",
                               dtype=jnp.bfloat16)(final)
        return logits
