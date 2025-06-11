"""Simplified Data models for OpenAI Vision + HyperCLOVAX pipeline."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ImageModalityType(str, Enum):
    """의료 영상 검사 유형"""
    XRAY = "X-Ray"
    CT = "CT"
    MRI = "MRI"
    ULTRASOUND = "Ultrasound"


class BodyPartType(str, Enum):
    """촬영 부위"""
    CHEST = "Chest"
    ABDOMEN = "Abdomen"
    HEAD = "Head"
    SPINE = "Spine"
    EXTREMITIES = "Extremities"


class UrgencyLevel(str, Enum):
    """응급도 수준"""
    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    STAT = "STAT"
    EMERGENCY = "EMERGENCY"


class ImageAnalysisRequest(BaseModel):
    """간소화된 의료 영상 분석 요청"""
    
    patient_id: str = Field(..., description="환자 ID")
    image_data: str = Field(..., description="Base64 인코딩된 이미지 데이터")
    modality: ImageModalityType = Field(..., description="영상 검사 유형")
    body_part: BodyPartType = Field(..., description="촬영 부위")
    priority: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE, description="우선순위")


class ImageAnalysisResult(BaseModel):
    """간소화된 의료 영상 분석 결과"""
    
    request_id: str = Field(..., description="요청 ID")
    patient_id: str = Field(..., description="환자 ID")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="분석 시간")
    modality: ImageModalityType = Field(..., description="영상 검사 유형")
    body_part: BodyPartType = Field(..., description="촬영 부위")
    
    # OpenAI Vision 결과
    overall_impression: str = Field(..., description="전체적인 인상")
    findings: List[str] = Field(default_factory=list, description="주요 발견사항")
    recommendations: List[str] = Field(default_factory=list, description="권고사항")
    
    # 메타 정보
    urgency_level: UrgencyLevel = Field(..., description="응급도")
    follow_up_required: bool = Field(default=False, description="추적검사 필요 여부")
    image_quality: str = Field(..., description="영상 품질")
    ai_confidence: float = Field(..., ge=0.0, le=1.0, description="AI 분석 신뢰도")
    model_version: str = Field(..., description="사용된 모델 버전")
    processing_time: float = Field(..., description="처리 시간 (초)")


# 이전의 복잡한 모델들 제거:
# - FindingResult (OpenAI Vision이 직접 처리)
# - ComparisonAnalysisRequest (불필요)
# - ComparisonResult (불필요) 
# - ReportTemplate (불필요)
# - QualityAssessment (OpenAI가 자동 처리)
# - PACSIntegration (현재 미사용)
# - SeverityLevel (간소화) 