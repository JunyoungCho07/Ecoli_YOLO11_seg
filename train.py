from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # YOLOv11-Seg 모델 로드
    model = YOLO('yolov11n-seg.pt')
    # n s m l x 모델 크기별

    # 학습 실행
    model.train(
        data="C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11",
        imgsz=512,# 512 1024
        epochs=50,
        batch=8,
        lr0=1e-3,
        lrf=0.01,
        device=0,
        project='C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_2025/runs',
        name='exp_seg_v11',
        task='segment',
        augment=True,
        hsv_h=0.015,  # 색조 조정
        hsv_s=0.7,    # 채도 조정
        hsv_v=0.4,    # 명도 조정
        degrees=0.0,  # 회전 각도
        translate=0.1,  # 이동ㄴ
        scale=0.5,    # 스케일링
        shear=0.0,    # 기울기
        perspective=0.0,  # 투시 왜곡
        flipud=0.0,   # 위아래 뒤집기
        fliplr=0.5,   # 좌우 뒤집기 확률
        mosaic=1.0,   # Mosaic 비율
        mixup=0.0     # MixUp 비율
    )

