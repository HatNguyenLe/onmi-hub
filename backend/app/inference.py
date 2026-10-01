import threading
from pathlib import Path
from typing import Iterator

from .config import GENERATIVE_TASKS, SEQ2SEQ_TASKS

SEPARATOR = "\n---\n"


class InferenceError(Exception):
    pass


def device_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"torch": None, "device": "cpu", "cuda": False, "gpu": None}
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return {
        "torch": torch.__version__,
        "cuda": cuda,
        "mps": mps,
        "device": "cuda" if cuda else ("mps" if mps else "cpu"),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


class _StopOnEvent:
    """StoppingCriteria that aborts generation when the event is set."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class ModelManager:
    """Keeps a single model in memory at a time to bound RAM/VRAM usage."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self.model_id: str | None = None
        self.task: str | None = None
        self.model = None
        self.tokenizer = None
        self.pipe = None

    def unload(self, model_id: str | None = None) -> None:
        with self._lock:
            if model_id and model_id != self.model_id:
                return
            self.model = self.tokenizer = self.pipe = None
            self.model_id = self.task = None
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def load(self, model_row: dict) -> None:
        if self.model_id == model_row["id"]:
            return
        self.unload()
        with self._lock:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                pipeline,
            )

            path = str(Path(model_row["local_path"]))
            task = model_row["task"]
            device = device_info()["device"]
            dtype = torch.float16 if device == "cuda" else torch.float32
            try:
                if task in GENERATIVE_TASKS or task in SEQ2SEQ_TASKS:
                    cls = AutoModelForCausalLM if task in GENERATIVE_TASKS else AutoModelForSeq2SeqLM
                    self.tokenizer = AutoTokenizer.from_pretrained(path)
                    self.model = cls.from_pretrained(path, dtype=dtype).to(device)
                    self.model.eval()
                    if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # sentence-similarity is a Hub tag, not a transformers pipeline.
                    pipe_task = "feature-extraction" if task == "sentence-similarity" else task
                    self.pipe = pipeline(pipe_task, model=path, tokenizer=path, device=device)
            except Exception as e:  # noqa: BLE001
                self.model = self.tokenizer = self.pipe = None
                raise InferenceError(f"Không thể nạp model: {e}") from e
            self.model_id = model_row["id"]
            self.task = task

    # ---------- generation ----------

    def stream_chat(
        self, model_row: dict, messages: list[dict], params: dict, stop_event: threading.Event
    ) -> Iterator[str]:
        with self._gen_lock:
            self.load(model_row)
            if self.task in GENERATIVE_TASKS:
                yield from self._stream_causal(messages, params, stop_event)
            elif self.task in SEQ2SEQ_TASKS:
                yield from self._stream_seq2seq(messages, params, stop_event)
            else:
                yield self._run_pipeline(messages[-1]["content"], params)

    def _gen_kwargs(self, params: dict, stop_event: threading.Event) -> dict:
        from transformers import StoppingCriteriaList

        temperature = float(params.get("temperature", 0.7))
        kwargs = dict(
            max_new_tokens=int(params.get("max_new_tokens", 256)),
            repetition_penalty=float(params.get("repetition_penalty", 1.1)),
            stopping_criteria=StoppingCriteriaList([_StopOnEvent(stop_event)]),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=float(params.get("top_p", 0.9)))
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def _build_causal_inputs(self, messages: list[dict], max_new_tokens: int):
        tok = self.tokenizer
        ctx = getattr(self.model.config, "max_position_embeddings", None) or tok.model_max_length or 2048
        ctx = min(ctx, 32768)
        budget = max(ctx - max_new_tokens, 64)
        msgs = list(messages)
        while True:
            if tok.chat_template:
                try:
                    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                except Exception:  # noqa: BLE001 - e.g. template rejects 'system' role
                    msgs = _merge_system(msgs)
                    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
                if not hasattr(ids, "shape"):
                    ids = ids["input_ids"]
            else:
                ids = tok(_plain_prompt(msgs), return_tensors="pt").input_ids
            non_system = [m for m in msgs if m["role"] != "system"]
            if ids.shape[-1] <= budget or len(non_system) <= 1:
                return ids[:, -budget:]
            # Drop the oldest non-system turn and retry.
            idx = next(i for i, m in enumerate(msgs) if m["role"] != "system")
            msgs.pop(idx)

    def _stream_causal(self, messages, params, stop_event) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        kwargs = self._gen_kwargs(params, stop_event)
        input_ids = self._build_causal_inputs(messages, kwargs["max_new_tokens"]).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300)
        yield from self._run_streamer(
            dict(input_ids=input_ids, attention_mask=input_ids.new_ones(input_ids.shape), streamer=streamer, **kwargs),
            streamer,
            stop_event,
            stop_strings=None if self.tokenizer.chat_template else ["\nUser:", "\nNgười dùng:"],
        )

    def _stream_seq2seq(self, messages, params, stop_event) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        kwargs = self._gen_kwargs(params, stop_event)
        text = messages[-1]["content"]
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=300)
        yield from self._run_streamer(dict(**enc, streamer=streamer, **kwargs), streamer, stop_event)

    def _run_streamer(self, gen_kwargs, streamer, stop_event, stop_strings=None) -> Iterator[str]:
        import torch

        error: list[Exception] = []

        def target():
            try:
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)
            except Exception as e:  # noqa: BLE001
                error.append(e)
                streamer.end()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        produced = ""
        try:
            for chunk in streamer:
                if not chunk:
                    continue
                produced += chunk
                if stop_strings:
                    hit = min((produced.find(s) for s in stop_strings if s in produced), default=-1)
                    if hit >= 0:
                        tail = produced[:hit][len(produced) - len(chunk):]
                        if tail:
                            yield tail
                        break
                yield chunk
        finally:
            stop_event.set()
            thread.join(timeout=30)
        if error:
            raise InferenceError(f"Lỗi khi sinh văn bản: {error[0]}")

    def _run_pipeline(self, text: str, params: dict) -> str:
        task = self.task
        try:
            if task == "question-answering":
                if SEPARATOR not in text:
                    return "Định dạng: nhập **câu hỏi**, xuống dòng `---`, rồi nhập **ngữ cảnh**."
                q, c = text.split(SEPARATOR, 1)
                r = self.pipe(question=q.strip(), context=c.strip())
                return f"**Trả lời:** {r['answer']}\n\nĐộ tin cậy: {r['score']:.2%}"
            if task == "zero-shot-classification":
                if SEPARATOR not in text:
                    return "Định dạng: nhập **văn bản**, xuống dòng `---`, rồi nhập **các nhãn** cách nhau bởi dấu phẩy."
                t, labels = text.split(SEPARATOR, 1)
                r = self.pipe(t.strip(), candidate_labels=[x.strip() for x in labels.split(",") if x.strip()])
                return "\n".join(f"- **{l}**: {s:.2%}" for l, s in zip(r["labels"], r["scores"]))
            if task == "fill-mask":
                mask = self.pipe.tokenizer.mask_token
                text = text.replace("[MASK]", mask).replace("<mask>", mask)
                if mask not in text:
                    return f"Hãy chèn token `{mask}` (hoặc `[MASK]`) vào câu."
                r = self.pipe(text)
                r = r[0] if r and isinstance(r[0], list) else r
                return "\n".join(f"- {x['sequence']}  ({x['score']:.2%})" for x in r)
            if task == "text-classification":
                r = self.pipe(text, top_k=None)
                r = r[0] if r and isinstance(r[0], list) else r
                return "\n".join(f"- **{x['label']}**: {x['score']:.2%}" for x in r)
            if task == "token-classification":
                r = self.pipe(text, aggregation_strategy="simple")
                if not r:
                    return "Không tìm thấy thực thể nào."
                return "\n".join(f"- `{x['word']}` → **{x['entity_group']}** ({x['score']:.2%})" for x in r)
            if task in ("feature-extraction", "sentence-similarity"):
                if task == "sentence-similarity" and SEPARATOR in text:
                    return self._similarity(*text.split(SEPARATOR, 1))
                r = self.pipe(text)
                import numpy as np

                arr = np.array(r[0] if isinstance(r, list) else r).squeeze()
                vec = arr.mean(axis=0) if arr.ndim == 2 else arr
                preview = ", ".join(f"{v:.4f}" for v in vec[:8])
                return f"Vector đặc trưng ({vec.shape[0]} chiều): [{preview}, ...]"
            r = self.pipe(text)
            return f"```json\n{r}\n```"
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"Lỗi pipeline: {e}") from e

    def _similarity(self, a: str, b: str) -> str:
        import numpy as np

        def emb(t):
            arr = np.array(self.pipe(t.strip())).squeeze()
            return arr.mean(axis=0) if arr.ndim == 2 else arr

        va, vb = emb(a), emb(b)
        cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        return f"**Độ tương đồng cosine:** {cos:.4f}"


def _merge_system(messages: list[dict]) -> list[dict]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest:
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    return rest


def _plain_prompt(messages: list[dict]) -> str:
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines = [f"{names.get(m['role'], m['role'])}: {m['content']}" for m in messages]
    return "\n".join(lines) + "\nAssistant:"


manager = ModelManager()
