import cv2
import os
import logging
import numpy as np
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PDF 경로를 입력받아 분석 결과를 도출하는 함수
def analyze_pdf(pdf_path, output_dir="output"):
    logging.info("PDF 분석을 시작합니다.")
    
    # PDF 파일 이름 추출 및 하위 폴더 경로 생성
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_dir = os.path.join(output_dir, pdf_name)

    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(pdf_output_dir):
        os.makedirs(pdf_output_dir)
        logging.info(f"출력 디렉토리를 생성했습니다: {pdf_output_dir}")

    # PDF를 이미지로 변환
    logging.info("PDF를 이미지로 변환 중...")
    images = convert_from_path(pdf_path)
    logging.info(f"{len(images)}개의 페이지가 변환되었습니다.")
    
    # 모델 다운로드 및 로드
    logging.info("모델을 다운로드하고 로드 중입니다...")
    filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
    model = YOLOv10(filepath)
    logging.info("모델 로드 완료.")

    # 각 페이지를 분석하고 결과를 저장
    for i, image in enumerate(images):
        logging.info(f"페이지 {i + 1} 분석 중...")

        # PIL 이미지를 OpenCV 이미지로 변환
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 임시 이미지 경로에 저장
        temp_image_path = os.path.join(pdf_output_dir, f"page_{i + 1}.jpg")
        cv2.imwrite(temp_image_path, image_cv)
        logging.info(f"페이지 {i + 1} 이미지를 {temp_image_path}에 저장했습니다.")

        # 모델 예측
        logging.info(f"페이지 {i + 1} 예측 수행 중...")
        det_res = model.predict(temp_image_path, imgsz=1024, conf=0.3, device="cpu")
        logging.info(f"페이지 {i + 1} 예측 완료.")

        # 결과 시각화 및 저장
        annotated_frame = det_res[0].plot(pil=False, line_width=5, font_size=20)
        result_image_path = os.path.join(pdf_output_dir, f"result_page_{i + 1}.jpg")
        cv2.imwrite(result_image_path, annotated_frame)
        logging.info(f"페이지 {i + 1} 분석 결과를 {result_image_path}에 저장했습니다.")

    logging.info("PDF 분석이 완료되었습니다.")

# 사용 예시
pdf_path = "/Users/tachyon/견적서/크몽_폼마루_홈페이지_견적서.pdf"
analyze_pdf(pdf_path)
