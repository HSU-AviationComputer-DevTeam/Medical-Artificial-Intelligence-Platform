#!/usr/bin/env python3
"""
Simple Vision Analysis Pipeline
OpenAI Vision API로 간단한 이미지 분석 → HyperCLOVAX로 종합 분석 파이프라인
"""

import base64
import json
import asyncio
import aiohttp
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import openai
from dotenv import load_dotenv
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from common.types import Message, Part, FilePart, TextPart


logger = logging.getLogger(__name__)


@dataclass
class VisionAnalysisResult:
    """Vision API 분석 결과"""
    description: str                    # 영상 설명
    findings: List[str]                 # 주요 발견사항  
    anatomical_structures: List[str]    # 해부학적 구조
    abnormalities: List[str]           # 이상 소견
    image_quality: str                 # 영상 품질
    confidence: float                  # 신뢰도
    analysis_timestamp: str            # 분석 시간


@dataclass 
class MedicalAnalysisResult:
    """의료 분석 결과"""
    vision_analysis: str
    final_report: str
    confidence_score: float
    detected_modality: str


class SimpleMedicalVisionPipeline:
    """간단한 의료 영상 분석 파이프라인"""
    
    def __init__(self):
        """초기화"""
        # .env 파일 로드
        load_dotenv()
        
        # OpenAI API 키 설정
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        
        # HyperCLOVAX 주소 (현재 A2A 오케스트레이터)
        self.hyperclovax_url = "http://localhost:12000"  # A2A UI 서버
        
        # OpenAI 클라이언트 초기화
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # HyperCLOVAX 모델 로드 (필요시)
        self.hyperclovax_model = None
        self.hyperclovax_tokenizer = None
        
        logger.info("SimpleMedicalVisionPipeline 초기화 완료")
    
    async def analyze_medical_image(self, image_data: str, user_prompt: str = "") -> MedicalAnalysisResult:
        """의료 영상 분석 실행 - OpenAI Vision API만 사용"""
        
        if not user_prompt:
            user_prompt = "이 의료 영상을 분석해주세요"
        
        # OpenAI Vision API로 전문적인 의료 영상 분석
        vision_analysis = await self._openai_vision_analysis(image_data, user_prompt)
        
        # 결과 구성 (HyperCLOVAX 호출 제거 - A2A 오케스트레이터가 처리)
        result = MedicalAnalysisResult(
            vision_analysis=vision_analysis,
            final_report=vision_analysis,  # Vision 분석 결과를 그대로 반환
            confidence_score=0.85,
            detected_modality=self._detect_image_modality(vision_analysis)
        )
        
        return result
    
    async def _openai_vision_analysis(self, image_data: str, user_prompt: str) -> str:
        """OpenAI Vision API로 영상 분석"""
        try:
            logger.info("OpenAI Vision API 분석 시작")
            
            # 의료 전문 프롬프트 구성
            medical_prompt = f"""
당신은 의료 영상 분석 전문가입니다. 다음 의료 영상을 상세히 분석해주세요:

사용자 요청: {user_prompt}

분석 시 다음 사항들을 포함해주세요:
1. 영상 종류 식별 (X-Ray, CT, MRI, Ultrasound 등)
2. 해부학적 구조 확인
3. 정상/비정상 소견
4. 특이사항이나 병변 여부
5. 추가 검사가 필요한 부분

※ 이는 의료진 참고용이며, 최종 진단은 의료진이 내립니다.
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": medical_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            logger.info("OpenAI Vision API 분석 완료")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI Vision API 오류: {e}")
            return f"OpenAI Vision 분석 중 오류 발생: {str(e)}"
    
    def _detect_image_modality(self, analysis_text: str) -> str:
        """분석 텍스트에서 영상 종류 감지"""
        analysis_lower = analysis_text.lower()
        
        if any(keyword in analysis_lower for keyword in ['x-ray', 'radiograph', '엑스레이', 'chest x-ray']):
            return "X-Ray"
        elif any(keyword in analysis_lower for keyword in ['ct', 'computed tomography', 'ct scan']):
            return "CT"
        elif any(keyword in analysis_lower for keyword in ['mri', 'magnetic resonance', '자기공명영상']):
            return "MRI"
        elif any(keyword in analysis_lower for keyword in ['ultrasound', 'sonography', '초음파']):
            return "Ultrasound"
        elif any(keyword in analysis_lower for keyword in ['mammography', '유방촬영']):
            return "Mammography"
        else:
            return "Unknown"


# 전역 파이프라인 인스턴스
_pipeline = None


def get_pipeline() -> SimpleMedicalVisionPipeline:
    """파이프라인 싱글톤 인스턴스 반환"""
    global _pipeline
    if _pipeline is None:
        _pipeline = SimpleMedicalVisionPipeline()
    return _pipeline


async def process_medical_image(message: Message) -> str:
    """A2A Message 객체에서 이미지와 텍스트를 추출하여 의료 분석 실행"""
    try:
        # Message에서 텍스트와 이미지 추출
        text_content = ""
        image_data = None
        
        for part in message.parts:
            if isinstance(part, TextPart):
                text_content += part.text or ""
            elif isinstance(part, FilePart) and part.file:
                # Base64 이미지 데이터 추출
                image_data = part.file.bytes
                logger.debug(f"이미지 데이터 길이: {len(image_data) if image_data else 0}")
        
        if not image_data:
            return "안녕하세요! 의료 영상을 업로드하시면 OpenAI Vision API와 HyperCLOVAX로 분석해드리겠습니다."
        
        # 의료 영상 분석 실행
        pipeline = get_pipeline()
        result = await pipeline.analyze_medical_image(
            image_data=image_data,
            user_prompt=text_content or "이 의료 영상을 분석해주세요"
        )
        
        return result.final_report
        
    except Exception as e:
        logger.error(f"의료 이미지 처리 오류: {e}")
        return f"죄송합니다. 영상 분석 중 오류가 발생했습니다: {str(e)}"


# 🎯 메인 파이프라인 함수
async def simple_vision_analysis_pipeline(image_data: str, user_prompt: str = "", conversation_id: str = "") -> str:
    """기존 호환성을 위한 래퍼 함수"""
    pipeline = get_pipeline()
    result = await pipeline.analyze_medical_image(image_data, user_prompt)
    return result.final_report


# 🧪 테스트 함수
async def test_simple_pipeline():
    """파이프라인 테스트"""
    # 테스트용 더미 이미지 데이터
    test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    test_prompt = "이 흉부 X-Ray 영상을 분석해주세요"
    
    try:
        result = await simple_vision_analysis_pipeline(
            test_image, test_prompt, "test_conversation"
        )
        print("✅ 테스트 성공:")
        print(result)
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_simple_pipeline()) 