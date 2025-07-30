# from ultralytics import YOLO

# # 모델 경로 지정
# model_path = "C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_2025/runs/exp_seg_colab3/weights/best.pt"

# # 모델 로드
# model = YOLO(model_path)

# print(model.task)
# #  ONNX로 export
# model.export(format='onnx')

from ultralytics import YOLO

model = YOLO(r"C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11_det/runs/train/gb4_exp2/weights/best.pt")

model.export(
    format="onnx",
    imgsz=1024,        # 학습과 동일
    opset=19,         # v8.3 로그 기준
    dynamic=True,
    simplify=False,   # ← 심플화 OFF
    optimize=False    # ← 슬림 OFF
)

import onnx
m = onnx.load(r"C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11_det/runs/train/gb4_exp2/weights/best.onnx")
print([o.name for o in m.graph.output])
# ['output0', 'output1'] → 반드시 두 개
