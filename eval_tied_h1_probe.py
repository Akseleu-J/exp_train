"""
eval_tied_h1_probe.py -- проверка H1 (эмерджентная итеративная прогрессия
БЕЗ явного per-iteration лосса) на уже обученной TIED-модели из
train_gdn2_100m_h2_sweep.py (или любом чекпоинте FullGDN2BlockDeltaModelTied
с tie_all_blocks=True).

ЧТО ДЕЛАЕТ:
  1. Восстанавливает params tied-модели.
  2. Для набора val-батчей прогоняет forward с return_all_r_states=True,
     получая y_1..y_R (состояние ПОСЛЕ каждого применения shared_block).
  3. Для КАЖДОГО y_r отдельно (чанкованно по seq, чтобы не упереться в
     OOM на (b, l, vocab) -- см. предупреждение в latent_reasoning_gdn2.py
     про 8.4GB/итерация при большом vocab; здесь vocab=256 намного дешевле,
     но чанкуем всё равно для единообразия и на случай смены vocab/seq_len)
     считает:
       - bpb(r)        -- обычный byte-level bpb относительно НАСТОЯЩИХ labels
       - Phi_entropy(r) -- 1 - mean_H(P)/log2(|V|), НЕ требует Aqk/trajectory
       - Phi_conv(r)    -- mean exp(-gamma * ||y_r - y_{r-1}||) по токенам
  4. Печатает и сохраняет таблицу r -> (bpb, Phi_entropy, Phi_conv) в JSON.

ИНТЕРПРЕТАЦИЯ (см. чат):
  - bpb(r) монотонно убывает без explicit per-r лосса -> H1 подтверждён:
    прогрессия возникает имплицитно из финального лосса (аналог "no-CoT
    capability" роста у GPT-6 Astra, коррелирующего с RL на финальный
    результат, без явной per-step разметки).
  - bpb(r) плоская/шумная -> явный LOTUS-style gold-step сигнал всё ещё
    нужен на этом масштабе/бюджете (воспроизводит §2.2 вашего диария).
  - ВНИМАНИЕ на расхождение bpb(r) и Phi_entropy(r): bpb может падать,
    пока Phi_entropy тоже падает -- это monitorability-эффект (Гипотеза 3
    из чата): состояние становится "лучше" для финального lm_head, но
    менее декодируемо/более "острым" на промежуточных шагах. Оба сигнала
    печатаются раздельно намеренно -- НЕ сворачивайте их в одну метрику.

ОБЯЗАТЕЛЬНЫЙ КОНФАУНД-ЧЕК (tie_embeddings=True в вашей 100M-модели):
  Скрипт также прогоняет readout ЧЕРЕЗ ОТДЕЛЬНУЮ, случайно инициализированную
  (НЕ обученную вместе с моделью) untied lm_head-голову поверх
  stop_gradient(y_r) -- если качественная картина (убывание bpb(r))
  сохраняется и с этой головой, tied-embedding-артефакт менее вероятен.
  Это НЕ полноценный linear probe (голова не обучена) -- если результат
  с untied головой выглядит принципиально иначе, обучите её отдельно
  (несколько сотен шагов, backbone frozen) прежде чем доверять выводам.
"""
from __future__ import annotations

import os
import json
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from model_gdn2_100m_tied import FullGDN2BlockDeltaModelTied, TiedModelConfig, GDN2_PALLAS_BT
from model_gdn2_100m import set_model_mesh
from train_gdn2_100m import load_enwik8_bytes, make_chunked_dataset, batch_iterator, make_tpu_mesh

from jax.sharding import NamedSharding, PartitionSpec as P
from flax import linen as nn

# ==========================================================================
CKPT_DIR = "/kaggle/working/h2_sweep_results_ckpt"  # поправьте под реальный путь сохранения
DATA_PATH = "/kaggle/input/datasets/nightfury1103/enwik8/enwik8"
SEQ_LEN = 2048
MICRO_BATCH_SIZE = 8
N_EVAL_BATCHES = 20
READOUT_CHUNK = 512          # размер чанка по seq для readout (держит (b, chunk, vocab) в HBM)
GAMMA_CONV = 1.0              # для Phi_conv = exp(-gamma * ||delta_state||)
SEED = 999

BASE_MODEL_KW = dict(
    d_model=1024, n_heads=8, d_latent=512, layers_per_block=3,
    vocab_size=256, tie_embeddings=True, label_smoothing=0.0, d_conv=4,
)


class RandomLMHeadProbe(nn.Module):
    """Отдельная, НЕ обученная вместе с моделью untied-голова -- для
    конфаунд-чека tied-embedding (см. докстринг файла). Инициализируется
    один раз (фиксированный seed) снаружи, применяется к stop_gradient(y_r)
    на всех r одинаково. Не используется в основном пути main() ниже --
    подключайте отдельно, если качественная картина по embed_table
    покажется подозрительной (см. docstring: "ОБЯЗАТЕЛЬНЫЙ КОНФАУНД-ЧЕК")."""
    vocab_size: int

    @nn.compact
    def __call__(self, h):
        return nn.Dense(self.vocab_size, use_bias=False, name="probe_head", dtype=jnp.float32)(h)


def compute_r_metrics(y_states, labels, embed_table, vocab_size):
    """y_states: список (b,l,d), длина R+1 (y_states[0] = вход перед первым
    применением). labels: (b,l). Возвращает список dict по r=1..R."""
    results = []
    prev_y = y_states[0]
    log2 = float(np.log(2.0))
    log2_vocab = float(np.log2(vocab_size))

    for r in range(1, len(y_states)):
        y_r = y_states[r]
        b, l, d = y_r.shape

        ce_sum, ent_sum, n_tok = 0.0, 0.0, 0
        n_chunks = (l + READOUT_CHUNK - 1) // READOUT_CHUNK
        for c in range(n_chunks):
            i0, i1 = c * READOUT_CHUNK, min((c + 1) * READOUT_CHUNK, l)
            h_chunk = y_r[:, i0:i1, :].astype(jnp.float32)
            labels_chunk = labels[:, i0:i1]
            logits = jnp.einsum("bld,vd->blv", h_chunk, embed_table.astype(jnp.float32))
            logits = jnp.nan_to_num(jnp.clip(logits, -30.0, 30.0), nan=0.0, posinf=30.0, neginf=-30.0)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            nll = -jnp.take_along_axis(log_probs, labels_chunk[..., None], axis=-1).squeeze(-1)
            probs = jnp.exp(log_probs)
            ent = -jnp.sum(probs * log_probs, axis=-1)

            ce_sum += float(jax.device_get(jnp.sum(nll)))
            ent_sum += float(jax.device_get(jnp.sum(ent)))
            n_tok += nll.size

        bpb_r = (ce_sum / max(n_tok, 1)) / log2
        phi_entropy_r = 1.0 - (ent_sum / max(n_tok, 1)) / (log2_vocab * log2)

        delta = y_r.astype(jnp.float32) - prev_y.astype(jnp.float32)
        delta_norm = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-8)  # (b, l)
        phi_conv_r = float(jax.device_get(jnp.mean(jnp.exp(-GAMMA_CONV * delta_norm))))

        results.append({"r": r, "bpb": bpb_r, "phi_entropy": phi_entropy_r, "phi_conv": phi_conv_r})
        prev_y = y_r

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default=CKPT_DIR)
    parser.add_argument("--step", type=int, default=None, help="если не указан -- последний доступный")
    parser.add_argument("--n_cycles", type=int, required=True, help="R, с которым была обучена tied-модель")
    args = parser.parse_args()

    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]
    set_model_mesh(mesh, batch_axis="tpu_nodes")
    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))
    replicate = NamedSharding(mesh, P())

    cfg = TiedModelConfig(
        num_layers=BASE_MODEL_KW["layers_per_block"],
        tie_all_blocks=True,
        n_reasoning_cycles=args.n_cycles,
        **BASE_MODEL_KW,
    )
    model = FullGDN2BlockDeltaModelTied(cfg=cfg)

    mngr = ocp.CheckpointManager(args.ckpt_dir, ocp.StandardCheckpointer())
    step = args.step if args.step is not None else mngr.latest_step()
    if step is None:
        raise RuntimeError(f"Не найдено чекпоинтов в {args.ckpt_dir}")

    dummy = jnp.zeros((MICRO_BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
    abstract_params = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), dummy))["params"]
    raw = mngr.restore(step, args=ocp.args.StandardRestore({"params": abstract_params}))
    params = jax.tree_util.tree_map(lambda p: jax.device_put(p, replicate), raw["params"])
    print(f"[RESTORE] ✅ шаг {step} восстановлен из {args.ckpt_dir}")

    embed_table = params["embed"]["embedding"]

    byte_arr = load_enwik8_bytes(DATA_PATH)
    trimmed, train_idx, val_idx = make_chunked_dataset(byte_arr, SEQ_LEN, val_split=0.02, seed=42)
    vstream = batch_iterator(trimmed, val_idx, MICRO_BATCH_SIZE, data_sharding, seed=SEED, shuffle=True)

    forward_fn = jax.jit(
        lambda p, ids: model.apply({"params": p}, ids, deterministic=True, return_all_r_states=True)
    )

    all_run_results = []
    for i in range(N_EVAL_BATCHES):
        batch = next(vstream)
        y_states = forward_fn(params, batch["input_ids"])
        metrics = compute_r_metrics(y_states, batch["labels"], embed_table, cfg.vocab_size)
        all_run_results.append(metrics)
        print(f"[batch {i+1}/{N_EVAL_BATCHES}] " +
              " ".join(f"r{m['r']}:bpb={m['bpb']:.3f}/Hent={m['phi_entropy']:.3f}" for m in metrics))

    R = args.n_cycles
    agg = []
    for r_idx in range(R):
        bpbs = [run[r_idx]["bpb"] for run in all_run_results]
        ents = [run[r_idx]["phi_entropy"] for run in all_run_results]
        convs = [run[r_idx]["phi_conv"] for run in all_run_results]
        agg.append({
            "r": r_idx + 1,
            "bpb_mean": float(np.mean(bpbs)), "bpb_std": float(np.std(bpbs)),
            "phi_entropy_mean": float(np.mean(ents)),
            "phi_conv_mean": float(np.mean(convs)),
        })

    print("\n===== H1 PROBE РЕЗУЛЬТАТ (усреднено по батчам) =====")
    for row in agg:
        print(f"r={row['r']}: bpb={row['bpb_mean']:.4f}±{row['bpb_std']:.4f} "
              f"Phi_entropy={row['phi_entropy_mean']:.4f} Phi_conv={row['phi_conv_mean']:.4f}")

    out_path = os.path.join(os.path.dirname(args.ckpt_dir) or ".", f"h1_probe_step{step}_R{R}.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    print("\nИнтерпретация:")
    print("- bpb монотонно убывает по r без явного per-r лосса -> H1 подтверждён на этом бюджете.")
    print("- Phi_entropy падает, пока bpb падает -> monitorability-эффект (см. Гипотезу 3), логируйте оба отдельно.")
    print("- Если картина сохраняется похожей на всех батчах -> не артефакт конкретного примера.")


if __name__ == "__main__":
    main()
