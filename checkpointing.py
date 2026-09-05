"""
checkpointing.py -- HF Hub relay + orbax чекпоинтинг (save/restore/async).

Вынесено из train.py (был >1400 строк, стал неудобно читать/находить нужный
кусок). Ничего в логике НЕ изменено -- это чистый перенос функций и их
докстрингов/ФИКС-комментариев как есть, чтобы вся история решений
(гонка donate_argnums на шаге 414, осиротевшие директории на шаге 379,
переход на async для 'latest') осталась на месте, рядом с кодом, который
её объясняет.
"""
from __future__ import annotations

import os
import re
import time
import json
import shutil

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

# ==================== HF HUB RELAY ====================
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    from huggingface_hub import HfApi, snapshot_download, upload_folder, create_repo, login
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    HF_REPO_ID = user_secrets.get_secret("HF_REPO_ID")
    _HAS_HF = bool(HF_TOKEN)
    login(HF_TOKEN)
    if _HAS_HF:
        print(f"[HF] ✅ Интеграция: {HF_REPO_ID}")
    else:
        raise ImportError("Hugging face worck's uncorrectly")
except ImportError:
    _HAS_HF = False
    print("[WARN] pip install -q huggingface_hub")
except Exception as e:
    _HAS_HF = False
    print(f"[WARN] HF-интеграция недоступна ({type(e).__name__}: {e}) -- "
          f"проверьте, что секреты HF_TOKEN/HF_REPO_ID добавлены в Kaggle notebook "
          f"и подключены к этому ноутбуку (Add-ons → Secrets). Продолжаю без HF.")
# ФИКС: раз в 25 минут. ВАЖНО: с синхронным чекпоинтингом (см. ниже) реальная
# длительность записи будет видна в логах как время выполнения save_all_slots() --
# следите за первыми 2-3 циклами и увеличьте интервал, если запись занимает
# больше половины интервала (иначе TPU будет простаивать в ожидании I/O больше,
# чем считать).
CHECKPOINT_EVERY_SECONDS = 50 * 60

# ФИКС: 4 именованных слота на HF вместо "последние N" -- защищает от того,
# что один плохой шаг (как на 414-м) перезатирает единственную сохранённую
# копию. Слоты:
#   latest      -- держит 2 последних чекпоинта (N и N-1), для обычного resume
#   best_train  -- лучший train_loss за всё время
#   best_val    -- лучший val_loss (по частичной или полной валидации)
HF_LATEST_KEEP_N = 1  # ФИКС: было 2 -- со слотами best_train/best_val локально одновременно
                       # хранилось до 4 полных копий (params+opt_state), это, похоже, и
                       # переполняло диск /kaggle/working (см. диагностику в save_slot).
                       # N-1 всё равно доступен на HF при необходимости отката.

# ПАТЧ: глобальный словарь для отслеживания асинхронных сохранений
_PENDING_SAVES = {}


def make_manager(local_dir, max_to_keep):
    os.makedirs(local_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
        enable_async_checkpointing=False,   # ПАТЧ: было False
    )
    return ocp.CheckpointManager(local_dir, ocp.StandardCheckpointer(), options)


def save_slot(mngr, local_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, train_loss=None):
    step_dir = os.path.join(local_dir, str(step))

    if os.path.exists(step_dir) and not os.listdir(step_dir):
        print(f"[CKPT] ⚠️ Найдена пустая осиротевшая директория {step_dir}, удаляю перед сейвом...")
        os.rmdir(step_dir)

    try:
        du = shutil.disk_usage(local_dir)
        print(f"[CKPT] Диск перед сейвом: свободно {du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ "
              f"({100 * du.free / du.total:.1f}% свободно)")
    except Exception as e_du:
        print(f"[CKPT] ⚠️ Не удалось проверить место на диске: {e_du}")

    t0 = time.perf_counter()
    mngr.save(step, args=ocp.args.StandardSave({"params": params, "opt_state": opt_state}))
    mngr.wait_until_finished()
    elapsed = time.perf_counter() - t0

    os.makedirs(step_dir, exist_ok=True)
    if not os.listdir(step_dir):
        du = shutil.disk_usage(local_dir)
        raise RuntimeError(
            f"orbax mngr.save() для шага {step} в {local_dir} не создал ожидаемых файлов "
            f"({step_dir} пуст после wait_until_finished()). Свободно на диске: "
            f"{du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ -- если свободного места мало, "
            f"это, скорее всего, и есть причина (см. HF_LATEST_KEEP_N и очистку старых локальных "
            f"чекпоинтов). Если места достаточно -- возможна рассинхронизация внутренней "
            f"бухгалтерии CheckpointManager, проверьте состояние {local_dir}."
        )

    meta = {
        "global_step": int(step),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_train_loss": float(best_train_loss),
        "timestamp": time.time(),
    }
    if train_loss is not None:
        meta["train_loss"] = float(jax.device_get(train_loss))
    meta_path = os.path.join(step_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[CKPT] Сохранён локально: {local_dir}/{step} (заняло {elapsed:.1f}с)")
    return elapsed


def _finalize_pending_save(mngr):
    """Дожидается фоновой записи (если есть), проверяет что файлы реально
    появились на диске, пишет metadata.json. No-op (возвращает None), если
    для этого mngr ничего не в процессе."""
    key = id(mngr)
    pending = _PENDING_SAVES.pop(key, None)
    if pending is None:
        return None

    mngr.wait_until_finished()
    elapsed = time.perf_counter() - pending["t0"]

    step_dir = os.path.join(pending["local_dir"], str(pending["step"]))
    os.makedirs(step_dir, exist_ok=True)
    if not os.listdir(step_dir):
        du = shutil.disk_usage(pending["local_dir"])
        raise RuntimeError(
            f"async mngr.save() для шага {pending['step']} в {pending['local_dir']} не создал "
            f"ожидаемых файлов после wait_until_finished(). Свободно на диске: "
            f"{du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ."
        )

    meta = {
        "global_step": int(pending["step"]),
        "epoch": int(pending["epoch"]),
        "best_val_loss": float(pending["best_val_loss"]),
        "best_train_loss": float(pending["best_train_loss"]),
        "timestamp": time.time(),
    }
    if pending["train_loss"] is not None:
        meta["train_loss"] = float(jax.device_get(pending["train_loss"]))
    with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)

    print(f"[CKPT] ✅ Async-сейв подтверждён: {pending['local_dir']}/{pending['step']} "
          f"(от запуска до подтверждения прошло {elapsed:.1f}с, включая параллельно шедшее обучение)")
    return pending


def save_slot_async(mngr, local_dir, step, params, opt_state, epoch, best_val_loss, best_train_loss, train_loss=None):
    """Запускает async-сейв. ФИКС (OOM на шаге 4527, RESOURCE_EXHAUSTED): раньше
    снапшот делался jnp.array(x, copy=True) -- это дубль НА УСТРОЙСТВЕ (HBM), то
    есть на пике держались одновременно живые params+opt_state И их device-side
    копия, поверх чего тут же запускался следующий train_step -- не хватило ~1.17G
    при 375M свободных. orbax при записи на диск всё равно требует данные на host,
    поэтому on-device дубль был лишним и просто дорогим риском OOM. Снимаем снапшот
    сразу на host (jax.device_get) -- orbax.StandardSave прекрасно принимает numpy
    массивы наравне с jax arrays, а HBM во время снапшота вообще не растёт вторым
    полным дублем."""
    _finalize_pending_save(mngr)

    params_snapshot = jax.tree_util.tree_map(jax.device_get, params)
    opt_state_snapshot = jax.tree_util.tree_map(jax.device_get, opt_state)

    try:
        du = shutil.disk_usage(local_dir)
        print(f"[CKPT] Диск перед async-сейвом: свободно {du.free / 1e9:.2f} ГБ из {du.total / 1e9:.2f} ГБ "
              f"({100 * du.free / du.total:.1f}% свободно)")
    except Exception as e_du:
        print(f"[CKPT] ⚠️ Не удалось проверить место на диске: {e_du}")

    t0 = time.perf_counter()
    mngr.save(step, args=ocp.args.StandardSave({"params": params_snapshot, "opt_state": opt_state_snapshot}))
    _PENDING_SAVES[id(mngr)] = dict(
        step=step, local_dir=local_dir, epoch=epoch,
        best_val_loss=best_val_loss, best_train_loss=best_train_loss,
        train_loss=train_loss, t0=t0,
    )
    print(f"[CKPT] 🚀 Async-сейв запущен для шага {step} -- TPU продолжает без ожидания.")

def upload_slot(local_dir, repo_subdir, step, msg="", keep_last_n=1):
    """Заливает {local_dir}/{step} -> HF под path_in_repo={repo_subdir}/{step},
    затем чистит старые шаги ИМЕННО внутри этого repo_subdir."""
    if not _HAS_HF:
        return
    step_dir = os.path.join(local_dir, str(step))
    if not os.path.exists(step_dir):
        print(f"[HF] ⚠️ {step_dir} не найден, пропускаю upload")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
        st_path = os.path.join(step_dir, "STATUS.txt")
        with open(st_path, "w") as f:
            f.write(f"IDLE: slot={repo_subdir} last_step={step} | t={time.time()}\n")
        upload_folder(
            folder_path=step_dir,
            repo_id=HF_REPO_ID,
            repo_type="model",
            path_in_repo=f"{repo_subdir}/{step}",
            commit_message=f"[{repo_subdir}] step {step} {msg}",
        )
        print(f"[HF] ✅ Uploaded: {repo_subdir}/{step}")

        try:
            all_files = api.list_repo_files(HF_REPO_ID, repo_type="model")
            prefix = f"{repo_subdir}/"
            found_steps = set()
            for f_path in all_files:
                if f_path.startswith(prefix):
                    rest = f_path[len(prefix):]
                    m = re.match(r"^(\d+)/", rest)
                    if m:
                        found_steps.add(int(m.group(1)))
            steps_sorted = sorted(found_steps, reverse=True)
            for old_step in steps_sorted[keep_last_n:]:
                try:
                    api.delete_folder(
                        path_in_repo=f"{repo_subdir}/{old_step}",
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                    )
                    print(f"[HF] 🗑️ [{repo_subdir}] удалён старый шаг: {old_step}")
                except Exception as e_del:
                    print(f"[HF] ⚠️ Не удалось удалить {repo_subdir}/{old_step}: {e_del}")
        except Exception as e_list:
            print(f"[HF] ⚠️ Не удалось получить список файлов для чистки {repo_subdir}: {e_list}")
    except Exception as e:
        print(f"[HF] ❌ Upload error ({repo_subdir}): {e}")


from huggingface_hub import (
    HfApi, snapshot_download, upload_folder, create_repo, login,
    sync_bucket,   # ФИКС: buckets -- отдельное объектное хранилище (hf://buckets/...),
)                  # НЕ repo (model/dataset/space) -- snapshot_download никогда не найдёт
                    # то, что лежит в бакете, независимо от repo_type. Нужен отдельный API.


def download_slot(local_dir, repo_subdir, repo_id=None, repo_type="model", bucket_id=None):
    """Скачивает слот. Если bucket_id задан -- игнорирует repo_id/repo_type
    полностью и качает из HF Bucket (hf://buckets/{bucket_id}/{repo_subdir})
    через sync_bucket, а не snapshot_download (buckets -- НЕ git-репо, у них
    нет repo_type=model/dataset/space -- это была причина, по которой
    любой repo_type молча ничего не находил для бакета)."""
    if not _HAS_HF:
        return None

    if bucket_id is not None:
        try:
            bucket_path = f"hf://buckets/{bucket_id}/{repo_subdir}"
            print(f"[HF] Downloading slot '{repo_subdir}' from BUCKET {bucket_path}...")
            os.makedirs(local_dir, exist_ok=True)
            sync_bucket(bucket_path, local_dir)
            items = [d for d in os.listdir(local_dir) if d.isdigit()]
            if not items:
                print(f"[HF] ⚠️ Слот '{repo_subdir}' не найден в bucket {bucket_id} -- "
                      f"проверьте, что путь внутри бакета действительно называется "
                      f"'{repo_subdir}/<step>/...'.")
                return None
            latest = max(int(d) for d in items)
            print(f"[HF] Bucket slot '{repo_subdir}': найден шаг {latest}")
            return latest
        except Exception as e:
            print(f"[HF] Bucket download failed для слота '{repo_subdir}' (bucket={bucket_id}): {e}")
            return None

    # ---- старый путь: обычный git-репо (model/dataset/space) ----
    target_repo = repo_id if repo_id is not None else HF_REPO_ID
    try:
        print(f"[HF] Downloading slot '{repo_subdir}' from {target_repo} (repo_type={repo_type})...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=target_repo,
            local_dir=local_dir,
            repo_type=repo_type,
            allow_patterns=[f"{repo_subdir}/**"],
        )
        src_root = os.path.join(local_dir, repo_subdir)
        if not os.path.isdir(src_root):
            print(f"[HF] ⚠️ Слот '{repo_subdir}' не найден в {target_repo} (repo_type={repo_type}).")
            return None
        for step_name in os.listdir(src_root):
            src = os.path.join(src_root, step_name)
            dst = os.path.join(local_dir, step_name)
            if os.path.isdir(src) and not os.path.exists(dst):
                os.rename(src, dst)
        items = [d for d in os.listdir(local_dir) if d.isdigit()]
        if not items:
            return None
        latest = max(int(d) for d in items)
        print(f"[HF] Slot '{repo_subdir}': найден шаг {latest}")
        return latest
    except Exception as e:
        print(f"[HF] Download failed для слота '{repo_subdir}' в {target_repo} (repo_type={repo_type}): {e}")
        return None
