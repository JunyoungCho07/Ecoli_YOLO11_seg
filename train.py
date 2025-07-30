# from ultralytics import YOLO
# import multiprocessing

# if __name__ == '__main__':
#     multiprocessing.freeze_support()

#     # YOLOv11-Seg 모델 로드
#     model = YOLO('yolov11m.pt')
#     # n s m l x 모델 크기별

#     # 학습 실행
#     model.train(
#         data="C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_YOLO11",
#         imgsz=1024,# 512 1024
#         epochs=50,
#         batch=8,
#         lr0=1e-3,
#         lrf=0.01,
#         device=0,
#         project='C:/Users/cho-j/OneDrive/바탕 화면/Ecoli_2025/runs',
#         name='exp_det_v11',
#         task='detection',
#         augment=True,
#         hsv_h=0.015,  # 색조 조정
#         hsv_s=0.7,    # 채도 조정
#         hsv_v=0.4,    # 명도 조정
#         degrees=0.0,  # 회전 각도
#         translate=0.1,  # 이동ㄴ
#         scale=0.5,    # 스케일링
#         shear=0.0,    # 기울기
#         perspective=0.0,  # 투시 왜곡
#         flipud=0.0,   # 위아래 뒤집기
#         fliplr=0.5,   # 좌우 뒤집기 확률
#         mosaic=1.0,   # Mosaic 비율
#         mixup=0.0     # MixUp 비율
#     )



from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 1) 모델 로드
    model = YOLO("yolo11s.pt")



    # 3) 학습 실행
    model.train(
        data="C:/Users/cho-j/OneDrive/바탕 화면/data2nd/data.yaml",
        epochs=50,
        imgsz=1024,
        batch=-1,         # 60 % VRAM 자동
        device=0,
        close_mosaic=10,
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015, 
        hsv_s=0.7, 
        hsv_v=0.4,
        fliplr=0.5,
        translate=0.1, 
        scale=0.5,
        project="runs/train",
        name="gb4_exp2",
    )

# 기대 데이터 확대 배수

# E = 4 × (1 + mixup)
# → mixup 0.15일 때 E ≈ 4.6 배.
# 예) 원본 10 000장 ⇒ epoch당 약 46 000 장이 모델에 노출됩니다. (flip·색상 변형은 픽셀만 변형하므로 ‘장 수’에 추가 계산하지 않음.)




# 1. 증강을 “모두 새로운 장수”로 간주할 때의 산식 — 논리
# 증강기법	동작 방식	우리가 새 장수로 더칠 때의 가산 비율
# Mosaic (mosaic=1.0)	한 학습 샘플을 만들 때 원본 4 장을 붙여 1 장을 생성	4배
# MixUp (mixup=p)	p 확률로 두 번째 Mosaic 이미지를 블렌딩	1 + p
# Horizontal Flip (fliplr=q)	q 확률로 좌우 반전	1 + q
# Rotation
# (Ultralytics YOLO의 degrees≠0)	회전은 항상 적용되어 각 샘플이 임의 각도로 돌려짐	1 + 1 ≈ 2
# (회전有·無 를 별개 장수로 본다면)

# 따라서 “각 기법을 독립적으로 겹쳐 쓴다”는 가정 아래,
# 총 확대 배수 E 는
 
# 2. 예시 — 기본 권장 하이퍼파라미터
# 파라미터	값	설명
# mosaic	1.0	항상 사용
# mixup	0.15	15 % 확률
# fliplr	0.50	50 % 확률
# degrees	±10°	임의 회전(=100 % 확률)
  
# =4×(1+0.15)×(1+0.50)×2
# =4×1.15×1.5×2
# =13.8
# ​
 
# 즉, 원본 10 000장이 있다면 한 epoch 동안

# “이미지”
# 10000×13.8=138000“이미지”
# 가 서로 다른 픽셀 구성을 갖는 형태로 모델에 노출됩니다.