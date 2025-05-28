"""Medical data models using Pydantic."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class PatientGender(str, Enum):
    """환자 성별"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class DepartmentType(str, Enum):
    """진료과 타입"""
    INTERNAL_MEDICINE = "내과"
    SURGERY = "외과"
    SAME_DAY = "당일진료"
    EMERGENCY = "응급실"


class UrgencyLevel(str, Enum):
    """응급도 레벨"""
    LOW = "낮음"
    MEDIUM = "보통"
    HIGH = "높음"
    CRITICAL = "위급"


class Patient(BaseModel):
    """환자 정보 모델"""
    patient_id: str = Field(..., description="환자 ID")
    name: str = Field(..., description="환자 이름")
    age: int = Field(..., description="나이")
    gender: PatientGender = Field(..., description="성별")
    phone: Optional[str] = Field(None, description="전화번호")
    address: Optional[str] = Field(None, description="주소")
    emergency_contact: Optional[str] = Field(None, description="응급 연락처")
    allergies: List[str] = Field(default_factory=list, description="알레르기 정보")
    medical_history: List[str] = Field(default_factory=list, description="과거 병력")
    created_at: datetime = Field(default_factory=datetime.now, description="등록일시")


class Symptom(BaseModel):
    """증상 정보 모델"""
    name: str = Field(..., description="증상명")
    severity: int = Field(..., ge=1, le=10, description="심각도 (1-10)")
    duration: str = Field(..., description="지속 기간")
    description: Optional[str] = Field(None, description="증상 설명")


class Diagnosis(BaseModel):
    """진단 정보 모델"""
    code: str = Field(..., description="진단 코드")
    name: str = Field(..., description="진단명")
    confidence: float = Field(..., ge=0.0, le=1.0, description="확신도")
    description: Optional[str] = Field(None, description="진단 설명")


class Medication(BaseModel):
    """약물 정보 모델"""
    name: str = Field(..., description="약물명")
    dosage: str = Field(..., description="용량")
    frequency: str = Field(..., description="복용 빈도")
    duration: str = Field(..., description="복용 기간")
    instructions: Optional[str] = Field(None, description="복용 지시사항")
    side_effects: List[str] = Field(default_factory=list, description="부작용")


class MedicalRecord(BaseModel):
    """진료 기록 모델"""
    record_id: str = Field(..., description="진료 기록 ID")
    patient_id: str = Field(..., description="환자 ID")
    department: DepartmentType = Field(..., description="진료과")
    doctor_name: str = Field(..., description="담당의")
    visit_date: datetime = Field(..., description="진료일시")
    chief_complaint: str = Field(..., description="주 호소")
    symptoms: List[Symptom] = Field(default_factory=list, description="증상 목록")
    vital_signs: Dict[str, Any] = Field(default_factory=dict, description="활력징후")
    physical_exam: Optional[str] = Field(None, description="신체 검사")
    diagnosis: List[Diagnosis] = Field(default_factory=list, description="진단 목록")
    treatment_plan: Optional[str] = Field(None, description="치료 계획")
    medications: List[Medication] = Field(default_factory=list, description="처방 약물")
    follow_up: Optional[str] = Field(None, description="추후 관리")
    urgency_level: UrgencyLevel = Field(default=UrgencyLevel.LOW, description="응급도")


class SOAPNote(BaseModel):
    """SOAP 노트 모델"""
    record_id: str = Field(..., description="진료 기록 ID")
    subjective: str = Field(..., description="주관적 정보 (S)")
    objective: str = Field(..., description="객관적 정보 (O)")
    assessment: str = Field(..., description="평가 (A)")
    plan: str = Field(..., description="계획 (P)")
    created_by: str = Field(..., description="작성자")
    created_at: datetime = Field(default_factory=datetime.now, description="작성일시")


class PatientSearchQuery(BaseModel):
    """환자 검색 쿼리 모델"""
    query_type: str = Field(..., description="검색 타입 (name, id, symptom, department)")
    query_text: str = Field(..., description="검색어")
    department: Optional[DepartmentType] = Field(None, description="진료과 필터")
    max_results: int = Field(default=10, description="최대 결과 수")
    similarity_threshold: float = Field(default=0.7, description="유사도 임계값")


class PatientSearchResult(BaseModel):
    """환자 검색 결과 모델"""
    patients: List[Patient] = Field(default_factory=list, description="검색된 환자 목록")
    total_count: int = Field(..., description="총 검색 결과 수")
    search_time: float = Field(..., description="검색 소요 시간")
    query: PatientSearchQuery = Field(..., description="검색 쿼리")


class DrugInteraction(BaseModel):
    """약물 상호작용 모델"""
    drug1: str = Field(..., description="약물 1")
    drug2: str = Field(..., description="약물 2")
    interaction_type: str = Field(..., description="상호작용 타입")
    severity: str = Field(..., description="심각도")
    description: str = Field(..., description="상호작용 설명")
    recommendation: str = Field(..., description="권장사항")


class MedicalAnalysis(BaseModel):
    """의료 분석 결과 모델"""
    patient_id: str = Field(..., description="환자 ID")
    analysis_type: str = Field(..., description="분석 타입")
    results: Dict[str, Any] = Field(..., description="분석 결과")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도")
    recommendations: List[str] = Field(default_factory=list, description="권장사항")
    created_at: datetime = Field(default_factory=datetime.now, description="분석일시") 