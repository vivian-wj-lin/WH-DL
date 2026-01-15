import csv
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from utils import BOARD_DISPLAY_NAMES, BOARDS, init_jieba, load_models, predict_title

BASE_DIR = Path(__file__).resolve().parent

models = {
    "doc2vec_model": None,
    "classifier": None,
    "device": None,
    "pseg": None,
}

FEEDBACK_FILE = "data/user-labeled-titles.csv"


class FeedbackRequest(BaseModel):
    input_title: str
    predicted_label: str
    predicted_confidence: float
    user_label: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nInitializing tokenizer...")
    models["pseg"] = init_jieba()

    print("\nLoading models...")
    doc2vec_model, classifier, device = load_models()
    models["doc2vec_model"] = doc2vec_model
    models["classifier"] = classifier
    models["device"] = device

    print("Application started successfully!")
    yield
    print("\nApplication shutting down...")


app = FastAPI(
    title="歡迎使用短文本分類預測",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "boards": BOARDS,
            "board_display_names": BOARD_DISPLAY_NAMES,
        },
    )


@app.get("/api/model/prediction")
def prediction_api(title: str = ""):
    if not title or not title.strip():
        raise HTTPException(
            status_code=400,
            detail="Please provide a title parameter.",
        )

    try:
        result = predict_title(
            title=title,
            doc2vec_model=models["doc2vec_model"],
            classifier=models["classifier"],
            device=models["device"],
            pseg_module=models["pseg"],
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/model/feedback")
def feedback_api(feedback: FeedbackRequest):
    if feedback.user_label not in BOARDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid board label. Must be one of: {', '.join(BOARDS)}",
        )

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp,
            feedback.input_title,
            feedback.predicted_label,
            feedback.predicted_confidence,
            feedback.user_label,
        ]

        file_exists = os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = [
                    "timestamp",
                    "input_title",
                    "predicted_label",
                    "predicted_confidence",
                    "user_label",
                ]
                writer.writerow(header)
            writer.writerow(row)
        return {"status": "success", "message": "感謝您的反饋"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save feedback: {str(e)}"
        )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
