import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = Path(os.getenv("OMNI_DATA_DIR", ROOT_DIR / "data")).resolve()
MODELS_DIR = DATA_DIR / "models"
DATASETS_DIR = DATA_DIR / "datasets"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "omnihub.db"
FRONTEND_DIR = ROOT_DIR / "frontend"

MAX_UPLOAD_BYTES = int(os.getenv("OMNI_MAX_UPLOAD_MB", "200")) * 1024 * 1024

# Pipeline tags considered "NLP" when browsing the Hugging Face Hub.
NLP_TASKS = {
    "text-generation": "Sinh văn bản",
    "text2text-generation": "Text-to-Text",
    "summarization": "Tóm tắt",
    "translation": "Dịch máy",
    "text-classification": "Phân loại văn bản",
    "token-classification": "Nhận dạng thực thể (NER)",
    "question-answering": "Hỏi đáp",
    "fill-mask": "Điền từ (Fill-Mask)",
    "zero-shot-classification": "Phân loại Zero-shot",
    "sentence-similarity": "Độ tương đồng câu",
    "feature-extraction": "Trích xuất đặc trưng",
}

GENERATIVE_TASKS = {"text-generation"}
SEQ2SEQ_TASKS = {"text2text-generation", "summarization", "translation"}

for _d in (DATA_DIR, MODELS_DIR, DATASETS_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
