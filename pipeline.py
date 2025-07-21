# 첫 실행 또는 가상환경 재구성 시
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-U",
                "ultralytics",   # YOLO v11 core
                "sahi",          # Sliced Inference
                "wandb"])        # 선택·모니터링



from ultralytics.data.converter import convert_coco

# COCO json이 들어 있는 폴더 경로
convert_coco(labels_dir="agar/annotations",
             save_dir="agar/yolo_labels",
             use_segments=False)     # detection용이므로 세그먼트 불필요


from pathlib import Path
import yaml, textwrap

dataset_root = Path("dataset")             # images/, labels/ 2‑계층 폴더
yaml_dict = dict(
    path=str(dataset_root),
    train="images/train",
    val="images/val",
    test="images/test",
    nc=1,
    names=["colony"]
)
(Path("colony.yaml")).write_text(textwrap.dedent(yaml.dump(yaml_dict)))




from ultralytics import YOLO

model = YOLO("yolo11s.pt")   # COCO pre‑trained 가중치
results = model.train(
    task="detect",
    data="agar.yaml",
    epochs=120,
    imgsz=640,
    batch=32,
    lr0=0.01,
    warmup_epochs=3,
    optimizer="AdamW",
    cos_lr=True,
    project="colony_exp",
    name="agar_pretrain",
    exist_ok=False,
    tracker="wandb"
)
best_ag_weights = results.best  # best.pt 경로




finetune = YOLO(best_ag_weights)
results_ft = finetune.train(
    task="detect",
    data="colony.yaml",
    epochs=60,
    imgsz=640,
    batch=24,
    lr0=0.002,
    freeze=10,         # 백본 일부 동결
    project="colony_exp",
    name="finetune_run",
    exist_ok=False
)




stats = finetune.val(
    data="colony.yaml",
    plots=True,            # PR, F1 곡선 저장
    save_json=True         # COCO‑style 결과 json
)
print(stats.results_dict)  # mAP50, mAP50‑95 등



pred = finetune.predict(
    source="samples/plate.jpg",   # str 또는 Path / 폴더 / glob / url / webcam
    conf=0.25,
    iou=0.5,
    save=True,                    # 결과 이미지 저장
)




from sahi.models.yolov8 import Yolov8DetectionModel   # v11도 동일 인터페이스
from sahi.predict import get_sliced_prediction

detection_model = Yolov8DetectionModel(
    model_path=finetune.ckpt_path,   # 또는 'best.pt'
    confidence_threshold=0.25,
    device="cuda:0"                  # CPU 사용 시 'cpu'
)
result = get_sliced_prediction(
    image="samples/large_plate.jpg",
    detection_model=detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
result.export_visuals(export_dir="sahi_vis/")



engine_path = finetune.export(
    format="engine",           # ONNX → TensorRT
    device=0                   # GPU index
)
print(f"TensorRT engine saved to {engine_path}")




# (1) 데이터셋 통계만 빠르게 보고 싶을 때
from ultralytics.data.utils import yaml_check
yaml_check("colony.yaml")    # 클래스·라벨 통계 표 출력

# (2) 벤치마크 FPS
fps = finetune.benchmark(source="samples/webcam.mp4", imgsz=640, device=0)
print(f"Avg FPS: {fps:.2f}")

# (3) 시드 고정 (재현성)
import torch, random, numpy as np
seed = 42
torch.manual_seed(seed);  random.seed(seed);  np.random.seed(seed)

