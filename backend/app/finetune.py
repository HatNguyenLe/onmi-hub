import csv
import json
import queue
import re
import shutil
import threading
import traceback
import uuid
from pathlib import Path

from . import database as db
from .config import DATASETS_DIR, JOBS_DIR
from .hf_service import (
    HubError,
    dir_size,
    get_local_model,
    get_token,
    safe_model_path,
    validate_local_name,
    validate_repo_id,
)
from .inference import device_info, manager

FT_TASKS = {
    "causal-lm": "text-generation",
    "seq2seq": "text2text-generation",
    "classification": "text-classification",
}
ALLOWED_EXT = {".csv", ".json", ".jsonl", ".txt"}
DATASET_FILE_RE = re.compile(r"^[0-9a-f]{32}\.(csv|json|jsonl|txt)$")


class FineTuneError(Exception):
    pass


# ---------------- datasets ----------------


def dataset_path(file_id: str) -> Path:
    if not DATASET_FILE_RE.match(file_id or ""):
        raise FineTuneError("Dataset không hợp lệ")
    path = DATASETS_DIR / file_id
    if not path.exists():
        raise FineTuneError("Không tìm thấy dataset")
    return path


def save_upload(filename: str, chunks) -> dict:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise FineTuneError("Chỉ hỗ trợ file .csv, .json, .jsonl, .txt")
    file_id = f"{uuid.uuid4().hex}{ext}"
    path = DATASETS_DIR / file_id
    with open(path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)
    meta = {"original_name": Path(filename).name[:200]}
    (DATASETS_DIR / f"{file_id}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    try:
        return {"id": file_id, **meta, **preview_file(path)}
    except Exception as e:
        path.unlink(missing_ok=True)
        raise FineTuneError(f"Không đọc được file: {e}") from e


def preview_file(path: Path, n: int = 5) -> dict:
    ext = path.suffix.lower()
    rows: list[dict] = []
    total = 0
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        if ext == ".csv":
            for row in csv.DictReader(f):
                total += 1
                if len(rows) < n:
                    rows.append({k: v for k, v in row.items() if k is not None})
        elif ext == ".jsonl":
            for line in f:
                if line.strip():
                    total += 1
                    if len(rows) < n:
                        rows.append(json.loads(line))
        elif ext == ".json":
            data = json.load(f)
            if isinstance(data, dict):
                data = next((v for v in data.values() if isinstance(v, list)), [data])
            total = len(data)
            rows = data[:n]
        else:
            for line in f:
                if line.strip():
                    total += 1
                    if len(rows) < n:
                        rows.append({"text": line.rstrip("\n")})
    columns = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
    return {"columns": columns, "rows": rows, "num_rows": total, "size": path.stat().st_size}


def list_datasets() -> list[dict]:
    out = []
    for p in sorted(DATASETS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not DATASET_FILE_RE.match(p.name):
            continue
        meta_p = DATASETS_DIR / f"{p.name}.meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        try:
            info = preview_file(p, n=3)
        except Exception:  # noqa: BLE001
            info = {"columns": [], "rows": [], "num_rows": 0, "size": p.stat().st_size}
        out.append({"id": p.name, "original_name": meta.get("original_name", p.name), **info})
    return out


def delete_dataset(file_id: str) -> None:
    path = dataset_path(file_id)
    path.unlink(missing_ok=True)
    (DATASETS_DIR / f"{file_id}.meta.json").unlink(missing_ok=True)


def preview_hub_dataset(name: str, config: str | None, split: str) -> dict:
    from datasets import load_dataset

    name = validate_repo_id(name)
    try:
        ds = load_dataset(name, config or None, split=split, streaming=True, token=get_token())
        rows = [dict(r) for _, r in zip(range(5), ds)]
    except Exception as e:  # noqa: BLE001
        raise FineTuneError(f"Không tải được dataset: {e}") from e
    rows = [{k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in r.items()} for r in rows]
    return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}


def _load_dataset(spec: dict):
    from datasets import load_dataset

    if spec["source"] == "upload":
        path = dataset_path(spec["file_id"])
        ext = path.suffix.lower()
        builder = {".csv": "csv", ".json": "json", ".jsonl": "json", ".txt": "text"}[ext]
        return load_dataset(builder, data_files=str(path), split="train")
    name = validate_repo_id(spec["name"])
    return load_dataset(name, spec.get("config") or None, split=spec.get("split") or "train", token=get_token())


# ---------------- jobs ----------------


DEFAULT_PARAMS = {
    "epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "grad_accum": 1,
    "max_length": 512,
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "eval_ratio": 0.1,
    "max_samples": 0,
    "use_lora": False,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "seed": 42,
}

_queue: "queue.Queue[int]" = queue.Queue()
_cancel: dict[int, threading.Event] = {}
_worker_started = False
_worker_lock = threading.Lock()


def _ensure_worker():
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_worker_loop, daemon=True).start()
            _worker_started = True


def _clean_params(raw: dict) -> dict:
    p = dict(DEFAULT_PARAMS)
    for k, default in DEFAULT_PARAMS.items():
        if k in raw and raw[k] is not None:
            try:
                p[k] = type(default)(raw[k]) if not isinstance(default, bool) else bool(raw[k])
            except (TypeError, ValueError) as e:
                raise FineTuneError(f"Tham số {k} không hợp lệ") from e
    if not 0 < p["epochs"] <= 100:
        raise FineTuneError("Số epoch phải trong khoảng (0, 100]")
    if not 0 < p["learning_rate"] < 1:
        raise FineTuneError("Learning rate không hợp lệ")
    if not 1 <= p["batch_size"] <= 512 or not 1 <= p["grad_accum"] <= 512:
        raise FineTuneError("Batch size / gradient accumulation không hợp lệ")
    if not 16 <= p["max_length"] <= 32768:
        raise FineTuneError("Max length phải trong khoảng 16 - 32768")
    if not 0 <= p["eval_ratio"] < 0.5:
        raise FineTuneError("Tỉ lệ eval phải trong khoảng [0, 0.5)")
    if not 0 <= p["warmup_ratio"] < 1:
        raise FineTuneError("Warmup ratio phải trong khoảng [0, 1)")
    return p


def create_job(payload: dict) -> dict:
    base = get_local_model(payload.get("base_model", ""))
    if not base or base["status"] != "ready":
        raise FineTuneError("Model gốc chưa sẵn sàng")
    task = payload.get("task")
    if task not in FT_TASKS:
        raise FineTuneError("Loại tác vụ fine-tune không hợp lệ")
    try:
        output = validate_local_name(payload.get("output_name", ""))
    except HubError as e:
        raise FineTuneError(str(e)) from e
    if get_local_model(output) or db.query_one(
        "SELECT id FROM finetune_jobs WHERE output_model=? AND status IN ('queued','running')", (output,)
    ):
        raise FineTuneError("Tên model đầu ra đã tồn tại")

    ds = payload.get("dataset") or {}
    if ds.get("source") == "upload":
        dataset_path(ds.get("file_id", ""))
        dataset = {"source": "upload", "file_id": ds["file_id"]}
    elif ds.get("source") == "hub":
        dataset = {
            "source": "hub",
            "name": validate_repo_id(ds.get("name", "")),
            "config": (ds.get("config") or "").strip()[:100] or None,
            "split": (ds.get("split") or "train").strip()[:50],
        }
    else:
        raise FineTuneError("Chưa chọn dataset")

    cols = payload.get("columns") or {}
    columns = {k: str(v)[:200] for k, v in cols.items() if v}
    if task == "causal-lm" and not (columns.get("text") or (columns.get("prompt") and columns.get("response"))):
        raise FineTuneError("Cần chọn cột 'text' hoặc cặp cột 'prompt' + 'response'")
    if task == "seq2seq" and not (columns.get("input") and columns.get("target")):
        raise FineTuneError("Cần chọn cột 'input' và 'target'")
    if task == "classification" and not (columns.get("text") and columns.get("label")):
        raise FineTuneError("Cần chọn cột 'text' và 'label'")

    params = _clean_params(payload.get("params") or {})
    ts = db.now()
    job_id = db.execute(
        "INSERT INTO finetune_jobs(base_model, output_model, task, dataset, params, status, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (base["id"], output, task, json.dumps(dataset), json.dumps({**params, "columns": columns}), "queued", ts, ts),
    )
    _cancel[job_id] = threading.Event()
    _ensure_worker()
    _queue.put(job_id)
    return get_job(job_id)


def get_job(job_id: int) -> dict | None:
    row = db.query_one("SELECT * FROM finetune_jobs WHERE id=?", (job_id,))
    return _decode(row) if row else None


def list_jobs() -> list[dict]:
    return [_decode(r) for r in db.query("SELECT * FROM finetune_jobs ORDER BY id DESC")]


def _decode(row: dict) -> dict:
    row["dataset"] = json.loads(row["dataset"])
    row["params"] = json.loads(row["params"])
    row["logs"] = json.loads(row["logs"] or "[]")
    return row


def cancel_job(job_id: int) -> dict:
    job = get_job(job_id)
    if not job:
        raise FineTuneError("Không tìm thấy job")
    if job["status"] == "queued":
        _update(job_id, status="cancelled")
    elif job["status"] == "running":
        _cancel.setdefault(job_id, threading.Event()).set()
        _update(job_id, status="cancelling")
    return get_job(job_id)


def delete_job(job_id: int) -> None:
    job = get_job(job_id)
    if not job:
        raise FineTuneError("Không tìm thấy job")
    if job["status"] in ("running", "cancelling"):
        raise FineTuneError("Job đang chạy, hãy huỷ trước")
    db.execute("DELETE FROM finetune_jobs WHERE id=?", (job_id,))


def _update(job_id: int, **fields) -> None:
    fields["updated_at"] = db.now()
    cols = ", ".join(f"{k}=?" for k in fields)
    db.execute(f"UPDATE finetune_jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def _append_log(job_id: int, entry: dict) -> None:
    job = db.query_one("SELECT logs FROM finetune_jobs WHERE id=?", (job_id,))
    logs = json.loads(job["logs"] or "[]") if job else []
    logs.append(entry)
    _update(job_id, logs=json.dumps(logs[-500:]))


def _worker_loop():
    while True:
        job_id = _queue.get()
        job = get_job(job_id)
        if not job or job["status"] != "queued":
            continue
        _update(job_id, status="running", progress=0)
        try:
            _train(job, _cancel.setdefault(job_id, threading.Event()))
        except Exception as e:  # noqa: BLE001
            _append_log(job_id, {"type": "error", "message": traceback.format_exc()[-2000:]})
            _update(job_id, status="failed", error=str(e)[:1000])
        finally:
            _cancel.pop(job_id, None)
            shutil.rmtree(JOBS_DIR / str(job_id), ignore_errors=True)


def _train(job: dict, cancel: threading.Event) -> None:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    job_id = job["id"]
    task = job["task"]
    params = job["params"]
    cols = params.pop("columns")
    set_seed(params["seed"])
    base = get_local_model(job["base_model"])
    if not base:
        raise FineTuneError("Model gốc đã bị xoá")
    base_path = base["local_path"]
    _append_log(job_id, {"type": "info", "message": f"Đang tải dataset và model {base['id']}..."})

    # Free the chat model so training has the memory.
    manager.unload()

    ds = _load_dataset(job["dataset"])
    needed = [c for c in cols.values() if c]
    missing = [c for c in needed if c not in ds.column_names]
    if missing:
        raise FineTuneError(f"Dataset không có cột: {', '.join(missing)}")
    if params["max_samples"] > 0:
        ds = ds.shuffle(seed=params["seed"]).select(range(min(params["max_samples"], len(ds))))

    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    max_len = params["max_length"]
    extra_cfg = {}

    if task == "causal-lm":
        eos = tok.eos_token or ""

        def fmt(batch):
            if cols.get("text"):
                texts = [f"{t}{eos}" for t in batch[cols["text"]]]
            else:
                texts = []
                for p, r in zip(batch[cols["prompt"]], batch[cols["response"]]):
                    if tok.chat_template:
                        msgs = [{"role": "user", "content": str(p)}, {"role": "assistant", "content": str(r)}]
                        texts.append(tok.apply_chat_template(msgs, tokenize=False))
                    else:
                        texts.append(f"User: {p}\nAssistant: {r}{eos}")
            return tok(texts, truncation=True, max_length=max_len)

        model_cls = AutoModelForCausalLM
        collator = DataCollatorForLanguageModeling(tok, mlm=False)
    elif task == "seq2seq":

        def fmt(batch):
            enc = tok([str(x) for x in batch[cols["input"]]], truncation=True, max_length=max_len)
            enc["labels"] = tok(text_target=[str(x) for x in batch[cols["target"]]], truncation=True, max_length=max_len)["input_ids"]
            return enc

        model_cls = AutoModelForSeq2SeqLM
        collator = None  # needs the model, built below
    else:
        labels = sorted({str(x) for x in ds[cols["label"]]})
        if len(labels) < 2:
            raise FineTuneError("Cần ít nhất 2 nhãn để phân loại")
        label2id = {l: i for i, l in enumerate(labels)}
        extra_cfg = dict(
            num_labels=len(labels),
            id2label={i: l for l, i in label2id.items()},
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        def fmt(batch):
            enc = tok([str(x) for x in batch[cols["text"]]], truncation=True, max_length=max_len)
            enc["labels"] = [label2id[str(x)] for x in batch[cols["label"]]]
            return enc

        model_cls = AutoModelForSequenceClassification
        collator = DataCollatorWithPadding(tok)

    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    if len(ds) == 0:
        raise FineTuneError("Dataset rỗng sau khi xử lý")

    eval_ds = None
    if params["eval_ratio"] > 0 and len(ds) >= 10:
        split = ds.train_test_split(test_size=params["eval_ratio"], seed=params["seed"])
        ds, eval_ds = split["train"], split["test"]

    dev = device_info()
    model = model_cls.from_pretrained(base_path, **extra_cfg)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    if collator is None:
        collator = DataCollatorForSeq2Seq(tok, model=model)

    if params["use_lora"]:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise FineTuneError("Cần cài 'peft' để dùng LoRA: pip install peft") from e
        task_type = {"causal-lm": TaskType.CAUSAL_LM, "seq2seq": TaskType.SEQ_2_SEQ_LM, "classification": TaskType.SEQ_CLS}[task]
        model = get_peft_model(
            model,
            LoraConfig(task_type=task_type, r=params["lora_r"], lora_alpha=params["lora_alpha"], lora_dropout=params["lora_dropout"]),
        )
        trainable, total = model.get_nb_trainable_parameters()
        _append_log(job_id, {"type": "info", "message": f"LoRA: {trainable:,} / {total:,} tham số được huấn luyện"})

    _append_log(job_id, {"type": "info", "message": f"Train: {len(ds)} mẫu, Eval: {len(eval_ds) if eval_ds else 0} mẫu, thiết bị: {dev['device']}"})

    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.max_steps:
                _update(job_id, progress=round(state.global_step / state.max_steps, 4), epoch=state.epoch)
            if cancel.is_set():
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if not logs:
                return
            entry = {"type": "metric", "step": state.global_step, **logs}
            _append_log(job_id, entry)
            if "loss" in logs:
                _update(job_id, loss=logs["loss"])

    out_tmp = JOBS_DIR / str(job_id)
    steps_per_epoch = max(1, len(ds) // (params["batch_size"] * params["grad_accum"]))
    args = TrainingArguments(
        output_dir=str(out_tmp),
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        gradient_accumulation_steps=params["grad_accum"],
        learning_rate=params["learning_rate"],
        warmup_steps=params["warmup_ratio"],
        weight_decay=params["weight_decay"],
        logging_steps=max(1, min(50, steps_per_epoch // 5)),
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="no",
        report_to=[],
        fp16=dev["cuda"] and not torch.cuda.is_bf16_supported(),
        bf16=dev["cuda"] and torch.cuda.is_bf16_supported(),
        seed=params["seed"],
        disable_tqdm=True,
        dataloader_pin_memory=dev["cuda"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[ProgressCallback()],
    )
    trainer.train()

    if cancel.is_set():
        _update(job_id, status="cancelled")
        _append_log(job_id, {"type": "info", "message": "Đã huỷ huấn luyện"})
        return

    _append_log(job_id, {"type": "info", "message": "Đang lưu model..."})
    final = trainer.model
    if params["use_lora"]:
        final = final.merge_and_unload()
    out_path = safe_model_path(job["output_model"])
    final.save_pretrained(str(out_path), safe_serialization=True)
    tok.save_pretrained(str(out_path))
    db.execute(
        "INSERT INTO models(id, repo_id, task, source, base_model, local_path, status, progress, size_bytes, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (job["output_model"], job["output_model"], FT_TASKS[task], "finetuned", base["id"], str(out_path),
         "ready", 1, dir_size(out_path), db.now()),
    )
    _update(job_id, status="completed", progress=1)
    _append_log(job_id, {"type": "info", "message": f"Hoàn tất! Model đã lưu: {job['output_model']}"})
