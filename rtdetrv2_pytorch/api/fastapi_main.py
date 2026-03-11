import io
import os
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from src.core import YAMLConfig

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

# ---------------------------------------------------------------------------
# 모델 설정 (서버 시작 시 환경변수 또는 기본값으로 로드)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get(
    "RTDETR_CONFIG",
    os.path.join(os.path.dirname(__file__), "../configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"),
)
CHECKPOINT_PATH = os.environ.get(
    "RTDETR_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "../rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
)
DEVICE = os.environ.get("RTDETR_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_BASE_DIR = os.environ.get(
    "RTDETR_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "../output"),
)
Path(OUTPUT_BASE_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RT-DETRv2 Inference API")
app.mount("/output", StaticFiles(directory=OUTPUT_BASE_DIR), name="output")

# ---------------------------------------------------------------------------
# 모델 로드 (앱 시작 시 1회)
# ---------------------------------------------------------------------------
class RTDETRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


_model: RTDETRModel = None

def get_model() -> RTDETRModel:
    global _model
    if _model is None:
        cfg = YAMLConfig(CONFIG_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        _model = RTDETRModel(cfg).to(DEVICE)
        _model.eval()
        print(f"[INFO] Model loaded | config={CONFIG_PATH} | device={DEVICE}")
    return _model


@app.on_event("startup")
def startup():
    get_model()


# ---------------------------------------------------------------------------
# 전처리 / 추론 유틸
# ---------------------------------------------------------------------------
_transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

def preprocess(image: Image.Image, device: str):
    w, h = image.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = _transforms(image)[None].to(device)
    return im_data, orig_size


def run_inference(image: Image.Image, threshold: float):
    model = get_model()
    im_data, orig_size = preprocess(image, DEVICE)
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    results = []
    scr = scores[0]
    mask = scr > threshold
    for label, box, score in zip(labels[0][mask], boxes[0][mask], scr[mask]):
        results.append({
            "label": int(label.item()),
            "score": round(float(score.item()), 4),
            "box": [round(float(v), 2) for v in box],
        })
    return results


def draw_on_image(image: Image.Image, detections: list) -> Image.Image:
    d = ImageDraw.Draw(image)
    for det in detections:
        box = det["box"]
        d.rectangle(box, outline="red", width=2)
        d.text((box[0], box[1]), f"{det['label']} {det['score']}", fill="blue")
    return image


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@app.post("/detect/json", summary="이미지 업로드 → JSON 결과 반환")
async def detect_json(
    file: UploadFile = File(...),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="신뢰도 임계값"),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = run_inference(image, threshold)
    return JSONResponse({"detections": results, "count": len(results)})


@app.post("/detect/image", summary="이미지 업로드 → 바운딩박스 그린 이미지 반환")
async def detect_image(
    file: UploadFile = File(...),
    threshold: float = Query(0.6, ge=0.0, le=1.0, description="신뢰도 임계값"),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = run_inference(image, threshold)
    draw_on_image(image, results)

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# /detect/image_dir  — 디렉토리 or 단일 파일 경로 처리
# ---------------------------------------------------------------------------

class ImagePathRequest(BaseModel):
    image_path: str
    threshold: float = 0.6


@app.post("/detect/image_dir", summary="파일 또는 디렉토리 경로 → 처리 후 output 저장")
def detect_image_dir(body: ImagePathRequest):
    input_path = Path(body.image_path)
    threshold = body.threshold

    if not input_path.exists():
        return JSONResponse(status_code=400, content={"error": f"경로가 존재하지 않습니다: {input_path}"})

    # 단일 파일
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            return JSONResponse(status_code=400, content={"error": f"지원하지 않는 이미지 형식: {input_path.suffix}"})
        output_dir = Path(OUTPUT_BASE_DIR) / input_path.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            image = Image.open(input_path).convert("RGB")
            detections = run_inference(image, threshold)
            draw_on_image(image, detections)
            save_path = output_dir / (input_path.stem + ".jpg")
            image.save(save_path, format="JPEG")
            print(f"[INFO] {input_path.name} → {save_path} ({len(detections)} detections)")
            return JSONResponse({
                "output_dir": str(output_dir),
                "total": 1,
                "results": [{"file": input_path.name, "saved": str(save_path), "detections": detections, "count": len(detections)}],
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # 디렉토리
    if input_path.is_dir():
        image_files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_files:
            return JSONResponse(status_code=400, content={"error": "이미지 파일이 없습니다."})

        output_dir = Path(OUTPUT_BASE_DIR) / input_path.name
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = []
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
                detections = run_inference(image, threshold)
                draw_on_image(image, detections)
                save_path = output_dir / (img_path.stem + ".jpg")
                image.save(save_path, format="JPEG")
                print(f"[INFO] {img_path.name} → {save_path} ({len(detections)} detections)")
                summary.append({"file": img_path.name, "saved": str(save_path), "detections": detections, "count": len(detections)})
            except Exception as e:
                print(f"[ERROR] {img_path.name}: {e}")
                summary.append({"file": img_path.name, "error": str(e)})

        return JSONResponse({
            "output_dir": str(output_dir),
            "total": len(image_files),
            "results": summary,
        })

    return JSONResponse(status_code=400, content={"error": f"파일도 디렉토리도 아닙니다: {input_path}"})



@app.get("/health", summary="서버 상태 확인")
def health():
    return {"status": "ok", "device": DEVICE}
