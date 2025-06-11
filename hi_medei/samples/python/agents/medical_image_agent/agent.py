#!/usr/bin/env python3
"""
Simplified Medical Image Agent
OpenAI Vision API 기반의 간단한 의료 영상 분석 에이전트
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from models import (
    BodyPartType,
    ImageAnalysisRequest,
    ImageAnalysisResult,
    ImageModalityType,
    UrgencyLevel,
)


class SimplifiedMedicalImageAgent:
    """OpenAI Vision 기반 간단한 의료 영상 분석 에이전트"""
    
    def __init__(self, agent_id: str = None):
        """에이전트 초기화"""
        self.agent_id = agent_id or f"medical_image_agent_{uuid.uuid4().hex[:8]}"
        self.name = "Simplified Medical Image Agent"
        self.description = "OpenAI Vision API를 활용한 간단한 의료 영상 분석 에이전트"
        
        # 로깅 설정
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Simplified Medical Image Agent 초기화 완료: {self.agent_id}")
    
    async def analyze_image(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        """
        간소화된 이미지 분석 - OpenAI Vision Pipeline로 위임
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"이미지 분석 시작: {request.patient_id}")
            
            # 간단한 결과 생성 (실제 분석은 Simple Vision Pipeline에서)
            result = ImageAnalysisResult(
                request_id=f"simple_{request.patient_id}_{int(time.time())}",
                patient_id=request.patient_id,
                modality=request.modality,
                body_part=request.body_part,
                overall_impression="OpenAI Vision Pipeline으로 상세 분석 진행",
                findings=["OpenAI Vision API 기반 분석"],
                recommendations=["전문의 판독 권장"],
                urgency_level=request.priority,
                follow_up_required=False,
                image_quality="OpenAI에서 자동 평가",
                ai_confidence=0.8,
                model_version="Simplified-Agent-v1.0",
                processing_time=time.time() - start_time
            )
            
            self.logger.info(f"이미지 분석 완료: {request.patient_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"이미지 분석 실패: {e}")
            return self._create_error_result(request, str(e))
    
    def _create_error_result(self, request: ImageAnalysisRequest, error_msg: str) -> ImageAnalysisResult:
        """오류 결과 생성"""
        return ImageAnalysisResult(
            request_id=f"error_{request.patient_id}_{int(time.time())}",
            patient_id=request.patient_id,
            modality=request.modality,
            body_part=request.body_part,
            overall_impression=f"분석 실패: {error_msg}",
            findings=["분석 오류 발생"],
            recommendations=["기술적 문제로 인한 분석 실패. 다시 시도하거나 전문의 상담 권장"],
            urgency_level=UrgencyLevel.ROUTINE,
            follow_up_required=True,
            image_quality="unknown",
            ai_confidence=0.0,
            model_version="Simplified-Agent-v1.0",
            processing_time=0.0
        )
    
    async def process_query(self, query: str) -> str:
        """사용자 쿼리 처리"""
        return f"""
안녕하세요! 간소화된 의료 영상 분석 에이전트입니다.

**질문:** {query}

현재 OpenAI Vision API 기반으로 의료 영상을 분석합니다.
- 실시간 이미지 분석
- HyperCLOVAX와 연동한 종합 의료 소견
- A2A 프로토콜 지원

상세한 의료 영상 분석은 OpenAI Vision Pipeline을 통해 제공됩니다.
의료진의 판단을 보조하는 목적이며, 최종 진단은 전문의가 수행해야 합니다.
"""
    
    def get_agent_info(self) -> Dict[str, Any]:
        """에이전트 정보 반환"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": "1.0.0",
            "capabilities": [
                "openai_vision_analysis",
                "hyperclovax_integration", 
                "a2a_protocol_support"
            ],
            "supported_modalities": [
                "X-Ray", "CT", "MRI", "Ultrasound"
            ],
            "ai_backend": "OpenAI Vision API + HyperCLOVAX",
            "status": "active"
        }
    
    async def shutdown(self):
        """에이전트 종료"""
        self.logger.info(f"Simplified Medical Image Agent 종료: {self.agent_id}")


# 편의를 위한 팩토리 함수
def create_simplified_medical_image_agent(agent_id: str = None) -> SimplifiedMedicalImageAgent:
    """간소화된 의료 영상 분석 에이전트 생성"""
    return SimplifiedMedicalImageAgent(agent_id=agent_id) 