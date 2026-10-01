import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database as db
from . import finetune, hf_service
from .config import FRONTEND_DIR, MAX_UPLOAD_BYTES, NLP_TASKS
from .finetune import FineTuneError
from .hf_service import HubError
from .inference import InferenceError, device_info, manager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="OmniHub NLP", version="1.0.0", lifespan=lifespan)


@app.exception_handler(HubError)
@app.exception_handler(FineTuneError)
@app.exception_handler(InferenceError)
async def _domain_error(_req: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------- system / settings ----------------


class SettingsIn(BaseModel):
    hf_token: str | None = Field(default=None, max_length=200)


@app.get("/api/system")
def system_info():
    return {"device": device_info(), "tasks": NLP_TASKS, "loaded_model": manager.model_id}


@app.get("/api/settings")
def get_settings():
    token = db.get_setting("hf_token")
    return {"hf_token_set": bool(token), "hf_token_hint": f"hf_...{token[-4:]}" if token else None}


@app.put("/api/settings")
def update_settings(body: SettingsIn):
    if body.hf_token is not None:
        token = body.hf_token.strip()
        db.set_setting("hf_token", token or None)
    return get_settings()


# ---------------- hub ----------------


@app.get("/api/hub/models")
def hub_search(
    q: str | None = Query(default=None, max_length=100),
    task: str | None = None,
    sort: str = "downloads",
    limit: int = 30,
):
    return hf_service.search_models(q, task or None, sort, limit)


@app.get("/api/hub/models/{repo_id:path}")
def hub_model_details(repo_id: str):
    return hf_service.model_details(repo_id)


class DownloadIn(BaseModel):
    repo_id: str = Field(max_length=200)


@app.post("/api/models/download")
def download_model(body: DownloadIn):
    return hf_service.start_download(body.repo_id)


# ---------------- local models ----------------


@app.get("/api/models")
def list_models():
    return hf_service.list_local_models()


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    manager.unload(model_id)
    hf_service.delete_local_model(model_id)
    return {"ok": True}


@app.post("/api/models/unload")
def unload_model():
    manager.unload()
    return {"ok": True}


# ---------------- conversations ----------------


class ConversationIn(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    model_id: str | None = Field(default=None, max_length=200)
    system_prompt: str | None = Field(default=None, max_length=8000)


class GenParams(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=32000)
    params: GenParams = GenParams()
    regenerate: bool = False


def _get_conv(conv_id: int) -> dict:
    conv = db.query_one("SELECT * FROM conversations WHERE id=?", (conv_id,))
    if not conv:
        raise HTTPException(404, "Không tìm thấy cuộc hội thoại")
    return conv


@app.get("/api/conversations")
def list_conversations(q: str | None = Query(default=None, max_length=100)):
    sql = (
        "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) AS message_count "
        "FROM conversations c"
    )
    if q:
        like = f"%{q}%"
        return db.query(
            sql + " WHERE c.title LIKE ? OR EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=c.id "
            "AND m.content LIKE ?) ORDER BY c.updated_at DESC",
            (like, like),
        )
    return db.query(sql + " ORDER BY c.updated_at DESC")


@app.post("/api/conversations")
def create_conversation(body: ConversationIn):
    ts = db.now()
    conv_id = db.execute(
        "INSERT INTO conversations(title, model_id, system_prompt, created_at, updated_at) VALUES(?,?,?,?,?)",
        (body.title or "Cuộc trò chuyện mới", body.model_id, body.system_prompt, ts, ts),
    )
    return _get_conv(conv_id)


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: int):
    conv = _get_conv(conv_id)
    conv["messages"] = db.query("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,))
    return conv


@app.patch("/api/conversations/{conv_id}")
def update_conversation(conv_id: int, body: ConversationIn):
    _get_conv(conv_id)
    fields = body.model_dump(exclude_unset=True)
    if fields:
        cols = ", ".join(f"{k}=?" for k in fields)
        db.execute(
            f"UPDATE conversations SET {cols}, updated_at=? WHERE id=?",
            (*fields.values(), db.now(), conv_id),
        )
    return _get_conv(conv_id)


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int):
    _get_conv(conv_id)
    db.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    return {"ok": True}


@app.delete("/api/conversations")
def delete_all_conversations():
    db.execute("DELETE FROM messages")
    db.execute("DELETE FROM conversations")
    return {"ok": True}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/conversations/{conv_id}/messages")
def send_message(conv_id: int, body: MessageIn):
    conv = _get_conv(conv_id)
    model = hf_service.get_local_model(conv["model_id"] or "")
    if not model or model["status"] != "ready":
        raise HTTPException(400, "Hãy chọn một model đã tải xong cho cuộc hội thoại")

    if body.regenerate:
        # Drop the last assistant reply so it can be generated again.
        last = db.query_one(
            "SELECT id, role FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 1", (conv_id,)
        )
        if last and last["role"] == "assistant":
            db.execute("DELETE FROM messages WHERE id=?", (last["id"],))
    else:
        db.execute(
            "INSERT INTO messages(conversation_id, role, content, created_at) VALUES(?,?,?,?)",
            (conv_id, "user", body.content, db.now()),
        )
        if conv["title"] == "Cuộc trò chuyện mới":
            title = " ".join(body.content.split())[:60]
            db.execute("UPDATE conversations SET title=? WHERE id=?", (title, conv_id))

    history = db.query("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id", (conv_id,))
    messages = ([{"role": "system", "content": conv["system_prompt"]}] if conv["system_prompt"] else []) + history
    stop = threading.Event()
    params = body.params.model_dump()

    def stream():
        parts: list[str] = []
        try:
            yield _sse({"type": "start", "model": model["id"]})
            for chunk in manager.stream_chat(model, messages, params, stop):
                parts.append(chunk)
                yield _sse({"type": "token", "text": chunk})
        except InferenceError as e:
            yield _sse({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"Lỗi không xác định: {e}"})
        finally:
            stop.set()
            text = "".join(parts).strip()
            msg_id = None
            if text:
                msg_id = db.execute(
                    "INSERT INTO messages(conversation_id, role, content, model_id, created_at) VALUES(?,?,?,?,?)",
                    (conv_id, "assistant", text, model["id"], db.now()),
                )
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (db.now(), conv_id))
        yield _sse({"type": "done", "message_id": msg_id})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- datasets ----------------


@app.get("/api/datasets")
def list_datasets():
    return finetune.list_datasets()


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File quá lớn")
        chunks.append(chunk)
    return finetune.save_upload(file.filename or "", chunks)


@app.delete("/api/datasets/{file_id}")
def delete_dataset(file_id: str):
    finetune.delete_dataset(file_id)
    return {"ok": True}


class HubDatasetIn(BaseModel):
    name: str = Field(max_length=200)
    config: str | None = Field(default=None, max_length=100)
    split: str = Field(default="train", max_length=50)


@app.post("/api/datasets/hub-preview")
def hub_dataset_preview(body: HubDatasetIn):
    return finetune.preview_hub_dataset(body.name, body.config, body.split)


# ---------------- fine-tuning ----------------


@app.get("/api/finetune/jobs")
def list_jobs():
    return finetune.list_jobs()


@app.post("/api/finetune/jobs")
def create_job(payload: dict):
    return finetune.create_job(payload)


@app.get("/api/finetune/jobs/{job_id}")
def get_job(job_id: int):
    job = finetune.get_job(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    return job


@app.post("/api/finetune/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    return finetune.cancel_job(job_id)


@app.delete("/api/finetune/jobs/{job_id}")
def delete_job(job_id: int):
    finetune.delete_job(job_id)
    return {"ok": True}


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
