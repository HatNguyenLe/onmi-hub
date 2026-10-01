import json
import os
import re
import shutil
import threading
from fnmatch import fnmatch
from pathlib import Path

from . import database as db
from .config import MODELS_DIR, NLP_TASKS

REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]{0,95}(/[A-Za-z0-9][\w.\-]{0,95})?$")
LOCAL_NAME_RE = re.compile(r"^[A-Za-z0-9][\w.\-]{0,190}$")

# Weight formats we never need for PyTorch inference/training.
IGNORE_PATTERNS = [
    "*.h5", "*.msgpack", "*.ot", "*.onnx", "*.onnx_data", "onnx/*", "*.tflite",
    "tf_model*", "flax_model*", "rust_model*", "*.gguf", "coreml/*", "openvino/*",
    "*.mlmodel", "*.mlpackage/*",
]


class HubError(Exception):
    pass


def validate_repo_id(repo_id: str) -> str:
    repo_id = (repo_id or "").strip()
    if not REPO_RE.match(repo_id) or ".." in repo_id:
        raise HubError("Repo ID không hợp lệ")
    return repo_id


def validate_local_name(name: str) -> str:
    name = (name or "").strip()
    if not LOCAL_NAME_RE.match(name) or ".." in name:
        raise HubError("Tên model không hợp lệ (chỉ dùng chữ, số, '.', '-', '_')")
    return name


def local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def safe_model_path(name: str) -> Path:
    path = (MODELS_DIR / name).resolve()
    if MODELS_DIR.resolve() not in path.parents:
        raise HubError("Đường dẫn không hợp lệ")
    return path


def get_token() -> str | None:
    return db.get_setting("hf_token") or os.getenv("HF_TOKEN") or None


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=get_token())


def _dt(value) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else value


def _serialize(m) -> dict:
    return {
        "id": m.id,
        "task": getattr(m, "pipeline_tag", None),
        "downloads": getattr(m, "downloads", 0) or 0,
        "likes": getattr(m, "likes", 0) or 0,
        "library": getattr(m, "library_name", None),
        "last_modified": _dt(getattr(m, "last_modified", None)),
        "tags": (getattr(m, "tags", None) or [])[:12],
        "gated": bool(getattr(m, "gated", False)),
    }


def search_models(search: str | None, task: str | None, sort: str, limit: int) -> list[dict]:
    if task and task not in NLP_TASKS:
        raise HubError("Task không thuộc nhóm NLP được hỗ trợ")
    if sort not in {"downloads", "likes", "last_modified", "trending_score"}:
        sort = "downloads"
    limit = max(1, min(limit, 100))
    api = _api()
    kwargs = dict(search=search or None, sort=sort, filter="transformers")
    if task:
        kwargs["pipeline_tag"] = task
        kwargs["limit"] = limit
    else:
        kwargs["limit"] = limit * 4
    try:
        models = list(api.list_models(**kwargs))
    except Exception as e:  # noqa: BLE001 - surface hub errors to the UI
        raise HubError(f"Không thể truy vấn Hugging Face Hub: {e}") from e
    results = [_serialize(m) for m in models if getattr(m, "pipeline_tag", None) in NLP_TASKS]
    return results[:limit]


def model_details(repo_id: str) -> dict:
    repo_id = validate_repo_id(repo_id)
    try:
        info = _api().model_info(repo_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        raise HubError(f"Không lấy được thông tin model: {e}") from e
    data = _serialize(info)
    files = [
        {"name": s.rfilename, "size": s.size or 0}
        for s in (info.siblings or [])
        if not any(fnmatch(s.rfilename, p) for p in IGNORE_PATTERNS)
    ]
    data["files"] = files
    data["download_size"] = _download_size(files)
    card = getattr(info, "card_data", None) or {}
    data["license"] = card.get("license") if hasattr(card, "get") else None
    data["languages"] = card.get("language") if hasattr(card, "get") else None
    st = getattr(info, "safetensors", None)
    data["parameters"] = getattr(st, "total", None) if st else None
    return data


def _download_size(files: list[dict]) -> int:
    names = {f["name"] for f in files}
    has_st = any(n.endswith(".safetensors") for n in names)
    total = 0
    for f in files:
        # snapshot_download pulls everything; when safetensors exist we skip .bin below.
        if has_st and f["name"].endswith(".bin") and "pytorch_model" in f["name"]:
            continue
        total += f["size"]
    return total


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def start_download(repo_id: str) -> dict:
    repo_id = validate_repo_id(repo_id)
    name = local_name(repo_id)
    existing = db.query_one("SELECT * FROM models WHERE id=?", (name,))
    if existing and existing["status"] in ("downloading", "ready"):
        return existing

    details = model_details(repo_id)
    if details["task"] not in NLP_TASKS:
        raise HubError("Model này không thuộc nhóm tác vụ NLP")

    path = safe_model_path(name)
    db.execute("DELETE FROM models WHERE id=?", (name,))
    db.execute(
        "INSERT INTO models(id, repo_id, task, source, local_path, status, total_bytes, created_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (name, repo_id, details["task"], "hub", str(path), "downloading",
         details["download_size"], db.now()),
    )
    has_st = any(f["name"].endswith(".safetensors") for f in details["files"])
    ignore = IGNORE_PATTERNS + (["pytorch_model*.bin"] if has_st else [])
    threading.Thread(target=_download_worker, args=(name, repo_id, path, ignore), daemon=True).start()
    return db.query_one("SELECT * FROM models WHERE id=?", (name,))


def _download_worker(name: str, repo_id: str, path: Path, ignore: list[str]) -> None:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=repo_id, local_dir=str(path), token=get_token(), ignore_patterns=ignore)
        shutil.rmtree(path / ".cache", ignore_errors=True)
        db.execute(
            "UPDATE models SET status='ready', progress=1, size_bytes=?, error=NULL WHERE id=?",
            (dir_size(path), name),
        )
    except Exception as e:  # noqa: BLE001
        db.execute("UPDATE models SET status='error', error=? WHERE id=?", (str(e)[:500], name))


def list_local_models() -> list[dict]:
    rows = db.query("SELECT * FROM models ORDER BY created_at DESC")
    for r in rows:
        if r["status"] == "downloading":
            current = dir_size(Path(r["local_path"]))
            total = r["total_bytes"] or 0
            r["size_bytes"] = current
            r["progress"] = min(current / total, 0.99) if total else 0
        r["architecture"] = _architecture(Path(r["local_path"])) if r["status"] == "ready" else None
    return rows


def _architecture(path: Path) -> str | None:
    cfg = path / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        archs = data.get("architectures") or []
        return archs[0] if archs else data.get("model_type")
    except (OSError, ValueError):
        return None


def get_local_model(name: str) -> dict | None:
    return db.query_one("SELECT * FROM models WHERE id=?", (name,))


def delete_local_model(name: str) -> None:
    row = get_local_model(name)
    if not row:
        raise HubError("Không tìm thấy model")
    if row["status"] == "downloading":
        raise HubError("Model đang được tải, không thể xoá lúc này")
    shutil.rmtree(safe_model_path(row["id"]), ignore_errors=True)
    db.execute("DELETE FROM models WHERE id=?", (name,))
