import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
from src.core import YAMLConfig

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

app = FastAPI(title="RT-DETRv2 Inference API")

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
# 전처리 / 후처리 유틸
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

    draw = ImageDraw.Draw(image)
    for det in results:
        box = det["box"]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{det['label']} {det['score']}", fill="blue")

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/health", summary="서버 상태 확인")
def health():
    return {"status": "ok", "device": DEVICE}
