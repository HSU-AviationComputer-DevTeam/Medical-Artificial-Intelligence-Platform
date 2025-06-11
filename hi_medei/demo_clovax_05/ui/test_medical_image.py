#!/usr/bin/env python3
"""
의료 영상 업로드 테스트를 위한 샘플 이미지 생성 스크립트
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_medical_images():
    """샘플 의료 영상들을 생성합니다."""
    
    # 샘플 이미지 저장 폴더 생성
    sample_dir = "sample_medical_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # 1. 흉부 X-Ray 시뮬레이션
    chest_xray = Image.new('L', (512, 512), color=30)  # 어두운 배경
    draw = ImageDraw.Draw(chest_xray)
    
    # 폐 영역 그리기 (밝은 회색)
    draw.ellipse([100, 150, 200, 350], fill=120)  # 왼쪽 폐
    draw.ellipse([300, 150, 400, 350], fill=120)  # 오른쪽 폐
    
    # 심장 영역 (더 밝은 회색)
    draw.ellipse([200, 200, 300, 320], fill=180)
    
    # 갈비뼈 시뮬레이션
    for i in range(8):
        y = 120 + i * 30
        draw.arc([80, y, 420, y+40], start=0, end=180, fill=200, width=2)
    
    # 텍스트 추가
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((10, 10), "Sample Chest X-Ray", fill=255, font=font)
    draw.text((10, 480), "Test Image for Medical AI", fill=255, font=font)
    
    chest_xray.save(os.path.join(sample_dir, "sample_chest_xray.png"))
    print(f"✅ 흉부 X-Ray 샘플 생성: {sample_dir}/sample_chest_xray.png")
    
    # 2. CT 스캔 시뮬레이션
    ct_scan = Image.new('L', (512, 512), color=50)
    draw = ImageDraw.Draw(ct_scan)
    
    # 뇌 외곽선
    draw.ellipse([100, 100, 400, 400], fill=100, outline=150, width=3)
    
    # 뇌 내부 구조
    draw.ellipse([150, 180, 350, 320], fill=120)  # 뇌실
    draw.ellipse([200, 200, 300, 280], fill=80)   # 중앙 구조
    
    # 결절 시뮬레이션 (밝은 점)
    draw.ellipse([280, 160, 300, 180], fill=200)
    
    draw.text((10, 10), "Sample Brain CT", fill=255, font=font)
    draw.text((10, 480), "Simulated Nodule at (290,170)", fill=255, font=font)
    
    ct_scan.save(os.path.join(sample_dir, "sample_brain_ct.png"))
    print(f"✅ 뇌 CT 샘플 생성: {sample_dir}/sample_brain_ct.png")
    
    # 3. MRI 시뮬레이션
    mri_scan = Image.new('L', (512, 512), color=20)
    draw = ImageDraw.Draw(mri_scan)
    
    # 뇌 구조 (더 복잡한 패턴)
    draw.ellipse([80, 80, 420, 420], fill=90, outline=140, width=2)
    
    # 회백질/백질 구분
    draw.ellipse([120, 120, 380, 380], fill=110)
    draw.ellipse([160, 160, 340, 340], fill=70)
    
    # 이상 신호
    draw.ellipse([250, 200, 280, 230], fill=180)  # 고신호 병변
    
    draw.text((10, 10), "Sample Brain MRI", fill=255, font=font)
    draw.text((10, 480), "High signal lesion detected", fill=255, font=font)
    
    mri_scan.save(os.path.join(sample_dir, "sample_brain_mri.png"))
    print(f"✅ 뇌 MRI 샘플 생성: {sample_dir}/sample_brain_mri.png")
    
    # 4. 유방촬영술 시뮬레이션
    mammography = Image.new('L', (400, 600), color=40)
    draw = ImageDraw.Draw(mammography)
    
    # 유방 윤곽
    draw.polygon([(50, 100), (350, 120), (380, 500), (20, 480)], fill=90, outline=120)
    
    # 유선 조직 패턴
    for i in range(20):
        x = 50 + i * 15
        y = 150 + (i % 3) * 50
        draw.ellipse([x, y, x+20, y+30], fill=110)
    
    # 미세석회화 시뮬레이션
    for i in range(5):
        x = 200 + i * 10
        y = 300 + i * 5
        draw.point([(x, y)], fill=200)
    
    draw.text((10, 10), "Sample Mammography", fill=255, font=font)
    draw.text((10, 570), "Microcalcifications visible", fill=255, font=font)
    
    mammography.save(os.path.join(sample_dir, "sample_mammography.png"))
    print(f"✅ 유방촬영술 샘플 생성: {sample_dir}/sample_mammography.png")
    
    # 5. 정상 흉부 X-Ray
    normal_chest = Image.new('L', (512, 512), color=30)
    draw = ImageDraw.Draw(normal_chest)
    
    # 정상 폐 구조
    draw.ellipse([90, 140, 190, 360], fill=100)   # 왼쪽 폐
    draw.ellipse([310, 140, 410, 360], fill=100)  # 오른쪽 폐
    
    # 정상 심장
    draw.ellipse([210, 220, 290, 340], fill=160)
    
    # 정상 갈비뼈
    for i in range(8):
        y = 120 + i * 30
        draw.arc([70, y, 430, y+40], start=0, end=180, fill=180, width=2)
    
    draw.text((10, 10), "Normal Chest X-Ray", fill=255, font=font)
    draw.text((10, 480), "No abnormal findings", fill=255, font=font)
    
    normal_chest.save(os.path.join(sample_dir, "normal_chest_xray.png"))
    print(f"✅ 정상 흉부 X-Ray 샘플 생성: {sample_dir}/normal_chest_xray.png")
    
    print(f"\n📁 모든 샘플 이미지가 '{sample_dir}' 폴더에 생성되었습니다!")
    print(f"💡 이제 demo UI에서 이 이미지들을 업로드하여 의료 영상 분석을 테스트해보세요.")
    
    return sample_dir

if __name__ == "__main__":
    create_sample_medical_images() 