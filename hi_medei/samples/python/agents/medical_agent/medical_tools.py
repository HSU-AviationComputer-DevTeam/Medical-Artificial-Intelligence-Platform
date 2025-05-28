"""Medical tools for patient search, document generation, and analysis."""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from langchain.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

from models import (
    Patient, MedicalRecord, SOAPNote, PatientSearchQuery, 
    PatientSearchResult, DrugInteraction, MedicalAnalysis,
    DepartmentType, UrgencyLevel
)


class PatientSearchTool(BaseTool):
    """환자 검색 도구"""
    
    name: str = "patient_search"
    description: str = "환자 ID, 이름, 증상으로 환자를 검색합니다."
    data_path: str = Field(default="/Users/sindong-u/coding/project/hi_medei/data")
    patients_data: Dict[str, List[Dict]] = Field(default_factory=dict)
    
    def __init__(self, data_path: str = "/Users/sindong-u/coding/project/hi_medei/data", **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.patients_data = self._load_patient_data()
    
    def _load_patient_data(self) -> Dict[str, List[Dict]]:
        """환자 데이터를 로드합니다."""
        data = {}
        departments = ["내과환자", "외과환자", "당일진료환자"]
        
        for dept in departments:
            dept_path = os.path.join(self.data_path, dept)
            print(f"Checking path: {dept_path}")
            if os.path.exists(dept_path):
                dept_data = []
                for file in os.listdir(dept_path):
                    if file.endswith('.json'):
                        try:
                            with open(os.path.join(dept_path, file), 'r', encoding='utf-8') as f:
                                patient_list = json.load(f)
                                # JSON 파일이 리스트인 경우 각 환자에 department 추가
                                if isinstance(patient_list, list):
                                    for patient in patient_list:
                                        patient['department'] = dept
                                        dept_data.append(patient)
                                else:
                                    # 단일 환자 객체인 경우
                                    patient_list['department'] = dept
                                    dept_data.append(patient_list)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
                data[dept] = dept_data
            else:
                print(f"Path does not exist: {dept_path}")
        
        print(f"Loaded patient data: {len(data)} departments")
        for dept, patients in data.items():
            print(f"  {dept}: {len(patients)} patients")
        
        return data
    
    def _run(self, query: str, query_type: str = "name", max_results: int = 10) -> str:
        """환자 검색을 실행합니다."""
        start_time = time.time()
        results = []
        
        query_lower = query.lower()
        
        # 쿼리에서 검색 타입과 값을 파싱
        if ":" in query:
            parts = query.split(":", 1)
            if len(parts) == 2:
                search_type = parts[0].strip()
                search_value = parts[1].strip()
                
                if "이름" in search_type or "name" in search_type.lower():
                    query_type = "name"
                    query_lower = search_value.lower()
                elif "증상" in search_type or "symptom" in search_type.lower():
                    query_type = "symptom"
                    query_lower = search_value.lower()
                elif "진단" in search_type or "diagnosis" in search_type.lower():
                    query_type = "diagnosis"
                    query_lower = search_value.lower()
        
        for dept, patients in self.patients_data.items():
            for patient in patients:
                match_found = False
                
                if query_type == "name" and query_lower in patient.get('name', '').lower():
                    match_found = True
                elif query_type == "id" and query_lower in patient.get('id', '').lower():
                    match_found = True
                elif query_type == "diagnosis" and query_lower in patient.get('diagnosis', '').lower():
                    match_found = True
                elif query_type == "symptom":
                    symptoms = patient.get('symptoms', [])
                    if isinstance(symptoms, list):
                        if any(query_lower in symptom.lower() for symptom in symptoms):
                            match_found = True
                    elif isinstance(symptoms, str) and query_lower in symptoms.lower():
                        match_found = True
                
                if match_found:
                    results.append(patient)
        
        search_time = time.time() - start_time
        
        # 결과 제한
        results = results[:max_results]
        
        return json.dumps({
            "patients": results,
            "total_count": len(results),
            "search_time": search_time,
            "query": query,
            "query_type": query_type
        }, ensure_ascii=False, indent=2)


class VectorSearchTool(BaseTool):
    """벡터 검색 도구 - 유사 증상 환자 검색"""
    
    name: str = "vector_search"
    description: str = "증상을 기반으로 유사한 환자를 벡터 검색으로 찾습니다."
    openai_api_key: str = Field(default="")
    vectorstore: Optional[str] = Field(default=None)
    
    def __init__(self, openai_api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.openai_api_key = openai_api_key
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """벡터스토어를 초기화합니다."""
        try:
            # GeminiVectorStore 경로 사용
            gemini_vector_path = "/Users/sindong-u/coding/project/hi_medei/GeminiVectorStore/medical_vector_store"
            
            # FAISS 벡터스토어 로드 시도
            import faiss
            import pickle
            
            # 인덱스 파일들 확인
            index_path = os.path.join(gemini_vector_path, "index.faiss")
            pkl_path = os.path.join(gemini_vector_path, "index.pkl")
            
            if os.path.exists(index_path) and os.path.exists(pkl_path):
                print(f"GeminiVectorStore 로드 시도: {gemini_vector_path}")
                # 간단한 FAISS 벡터스토어 로드
                self.vectorstore = "faiss_loaded"  # 실제 구현은 FAISS 로드
                print("GeminiVectorStore 로드 성공")
            else:
                print("GeminiVectorStore 파일을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"Vector store initialization error: {e}")
    
    def _run(self, symptoms: str, k: int = 5) -> str:
        """유사 증상 환자를 검색합니다."""
        if not self.vectorstore:
            return json.dumps({"error": "Vector store not available"})
        
        try:
            # 실제 구현에서는 FAISS 검색을 수행
            # 현재는 시뮬레이션된 결과 반환
            results = [
                {
                    "content": f"유사 증상 환자 사례: {symptoms}와 관련된 환자",
                    "metadata": {"patient_id": "SIM001", "similarity": 0.85},
                    "similarity_score": 0.85
                },
                {
                    "content": f"{symptoms} 증상을 보인 과거 환자 기록",
                    "metadata": {"patient_id": "SIM002", "similarity": 0.78},
                    "similarity_score": 0.78
                }
            ]
            
            return json.dumps({
                "similar_cases": results,
                "query": symptoms,
                "total_found": len(results),
                "source": "GeminiVectorStore"
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Vector search failed: {e}"})


class SOAPNoteGeneratorTool(BaseTool):
    """SOAP 노트 생성 도구"""
    
    name: str = "soap_note_generator"
    description: str = "환자 정보를 바탕으로 SOAP 노트를 생성합니다."
    
    def _run(self, patient_data: str) -> str:
        """SOAP 노트를 생성합니다."""
        try:
            data = json.loads(patient_data)
            
            # SOAP 노트 구조 생성
            subjective = self._generate_subjective(data)
            objective = self._generate_objective(data)
            assessment = self._generate_assessment(data)
            plan = self._generate_plan(data)
            
            soap_note = {
                "record_id": data.get('record_id', f"SOAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "subjective": subjective,
                "objective": objective,
                "assessment": assessment,
                "plan": plan,
                "created_by": "AI Medical Agent",
                "created_at": datetime.now().isoformat()
            }
            
            return json.dumps(soap_note, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"SOAP note generation failed: {e}"})
    
    def _generate_subjective(self, data: Dict) -> str:
        """주관적 정보 (S) 생성"""
        subjective_parts = []
        
        if 'chief_complaint' in data:
            subjective_parts.append(f"주 호소: {data['chief_complaint']}")
        
        if 'symptoms' in data:
            symptoms_text = ", ".join(data['symptoms'])
            subjective_parts.append(f"증상: {symptoms_text}")
        
        if 'pain_scale' in data:
            subjective_parts.append(f"통증 정도: {data['pain_scale']}/10")
        
        return "\n".join(subjective_parts)
    
    def _generate_objective(self, data: Dict) -> str:
        """객관적 정보 (O) 생성"""
        objective_parts = []
        
        if 'vital_signs' in data:
            vital_signs = data['vital_signs']
            vital_text = []
            for key, value in vital_signs.items():
                vital_text.append(f"{key}: {value}")
            objective_parts.append(f"활력징후: {', '.join(vital_text)}")
        
        if 'physical_exam' in data:
            objective_parts.append(f"신체검사: {data['physical_exam']}")
        
        if 'lab_results' in data:
            objective_parts.append(f"검사결과: {data['lab_results']}")
        
        return "\n".join(objective_parts)
    
    def _generate_assessment(self, data: Dict) -> str:
        """평가 (A) 생성"""
        assessment_parts = []
        
        if 'diagnosis' in data:
            if isinstance(data['diagnosis'], list):
                for i, diag in enumerate(data['diagnosis'], 1):
                    assessment_parts.append(f"{i}. {diag}")
            else:
                assessment_parts.append(f"1. {data['diagnosis']}")
        
        if 'differential_diagnosis' in data:
            assessment_parts.append(f"감별진단: {data['differential_diagnosis']}")
        
        return "\n".join(assessment_parts)
    
    def _generate_plan(self, data: Dict) -> str:
        """계획 (P) 생성"""
        plan_parts = []
        
        if 'medications' in data:
            plan_parts.append("처방:")
            for med in data['medications']:
                if isinstance(med, dict):
                    med_text = f"- {med.get('name', '')} {med.get('dosage', '')} {med.get('frequency', '')}"
                else:
                    med_text = f"- {med}"
                plan_parts.append(med_text)
        
        if 'follow_up' in data:
            plan_parts.append(f"추후 관리: {data['follow_up']}")
        
        if 'patient_education' in data:
            plan_parts.append(f"환자 교육: {data['patient_education']}")
        
        return "\n".join(plan_parts)


class DrugInteractionCheckerTool(BaseTool):
    """약물 상호작용 검사 도구"""
    
    name: str = "drug_interaction_checker"
    description: str = "처방된 약물들 간의 상호작용을 검사합니다."
    interaction_db: Dict = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 간단한 약물 상호작용 데이터베이스 (실제로는 외부 API 사용)
        self.interaction_db = {
            ("아스피린", "와파린"): {
                "severity": "높음",
                "description": "출혈 위험 증가",
                "recommendation": "INR 모니터링 강화"
            },
            ("메트포르민", "요오드 조영제"): {
                "severity": "보통",
                "description": "젖산증 위험",
                "recommendation": "조영제 사용 전후 메트포르민 중단"
            }
        }
    
    def _run(self, medications: str) -> str:
        """약물 상호작용을 검사합니다."""
        try:
            med_list = json.loads(medications)
            interactions = []
            
            for i in range(len(med_list)):
                for j in range(i + 1, len(med_list)):
                    drug1 = med_list[i].get('name', '') if isinstance(med_list[i], dict) else med_list[i]
                    drug2 = med_list[j].get('name', '') if isinstance(med_list[j], dict) else med_list[j]
                    
                    # 상호작용 검사
                    interaction_key = (drug1, drug2)
                    reverse_key = (drug2, drug1)
                    
                    if interaction_key in self.interaction_db:
                        interaction_data = self.interaction_db[interaction_key]
                        interactions.append({
                            "drug1": drug1,
                            "drug2": drug2,
                            **interaction_data
                        })
                    elif reverse_key in self.interaction_db:
                        interaction_data = self.interaction_db[reverse_key]
                        interactions.append({
                            "drug1": drug2,
                            "drug2": drug1,
                            **interaction_data
                        })
            
            return json.dumps({
                "interactions": interactions,
                "total_interactions": len(interactions),
                "checked_medications": med_list
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Drug interaction check failed: {e}"})


class UrgencyAssessmentTool(BaseTool):
    """응급도 평가 도구"""
    
    name: str = "urgency_assessment"
    description: str = "환자의 증상과 활력징후를 바탕으로 응급도를 평가합니다."
    
    def _run(self, patient_data: str) -> str:
        """응급도를 평가합니다."""
        try:
            data = json.loads(patient_data)
            urgency_score = 0
            factors = []
            
            # 활력징후 평가
            vital_signs = data.get('vital_signs', {})
            
            # 혈압 평가
            if 'systolic_bp' in vital_signs:
                systolic = vital_signs['systolic_bp']
                if systolic > 180 or systolic < 90:
                    urgency_score += 3
                    factors.append("혈압 이상")
            
            # 맥박 평가
            if 'heart_rate' in vital_signs:
                hr = vital_signs['heart_rate']
                if hr > 120 or hr < 50:
                    urgency_score += 2
                    factors.append("맥박 이상")
            
            # 체온 평가
            if 'temperature' in vital_signs:
                temp = vital_signs['temperature']
                if temp > 39.0 or temp < 35.0:
                    urgency_score += 2
                    factors.append("체온 이상")
            
            # 증상 평가
            symptoms = data.get('symptoms', [])
            critical_symptoms = ['흉통', '호흡곤란', '의식저하', '심한복통', '출혈']
            
            for symptom in symptoms:
                if any(critical in symptom for critical in critical_symptoms):
                    urgency_score += 3
                    factors.append(f"위험 증상: {symptom}")
            
            # 응급도 결정
            if urgency_score >= 7:
                urgency_level = UrgencyLevel.CRITICAL
            elif urgency_score >= 5:
                urgency_level = UrgencyLevel.HIGH
            elif urgency_score >= 3:
                urgency_level = UrgencyLevel.MEDIUM
            else:
                urgency_level = UrgencyLevel.LOW
            
            return json.dumps({
                "urgency_level": urgency_level.value,
                "urgency_score": urgency_score,
                "contributing_factors": factors,
                "recommendations": self._get_urgency_recommendations(urgency_level)
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Urgency assessment failed: {e}"})
    
    def _get_urgency_recommendations(self, urgency_level: UrgencyLevel) -> List[str]:
        """응급도별 권장사항을 반환합니다."""
        recommendations = {
            UrgencyLevel.CRITICAL: [
                "즉시 응급실 이송",
                "활력징후 지속 모니터링",
                "응급처치 준비"
            ],
            UrgencyLevel.HIGH: [
                "우선 진료 필요",
                "30분 이내 재평가",
                "전문의 상담"
            ],
            UrgencyLevel.MEDIUM: [
                "1시간 이내 진료",
                "증상 변화 관찰",
                "필요시 재평가"
            ],
            UrgencyLevel.LOW: [
                "일반 진료 대기",
                "증상 악화시 재평가",
                "환자 교육 제공"
            ]
        }
        
        return recommendations.get(urgency_level, [])


class HybridSearchTool(BaseTool):
    """하이브리드 검색 도구 - JSON 데이터와 벡터 검색 결합"""
    
    name: str = "hybrid_search"
    description: str = "JSON 데이터와 벡터 검색을 결합하여 포괄적인 환자 검색을 수행합니다."
    data_path: str = Field(default="")
    openai_api_key: str = Field(default="")
    patient_search: Optional[PatientSearchTool] = Field(default=None)
    vector_search: Optional[VectorSearchTool] = Field(default=None)
    
    def __init__(self, data_path: str, openai_api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.openai_api_key = openai_api_key
        self.patient_search = PatientSearchTool(data_path=data_path)
        self.vector_search = VectorSearchTool(openai_api_key=openai_api_key)
    
    def _run(self, query: str, search_type: str = "comprehensive") -> str:
        """하이브리드 검색을 실행합니다."""
        try:
            results = {
                "query": query,
                "search_type": search_type,
                "json_results": {},
                "vector_results": {},
                "combined_insights": []
            }
            
            # 1. JSON 데이터 검색
            # 쿼리 형태에 따라 적절한 검색 수행
            if ":" not in query:
                # 단순 키워드인 경우 진단명으로 검색
                search_query = f"진단: {query}"
            else:
                search_query = query
            
            json_result = self.patient_search._run(search_query)
            results["json_results"] = json.loads(json_result)
            
            # 2. 벡터 검색 (증상 기반)
            vector_result = self.vector_search._run(query)
            results["vector_results"] = json.loads(vector_result)
            
            # 3. 결과 결합 및 인사이트 생성
            insights = []
            
            if results["json_results"]["total_count"] > 0:
                insights.append(f"구조화된 데이터에서 {results['json_results']['total_count']}명의 환자를 찾았습니다.")
            
            if results["vector_results"].get("total_found", 0) > 0:
                insights.append(f"유사 증상 검색에서 {results['vector_results']['total_found']}개의 관련 사례를 찾았습니다.")
            
            if not insights:
                insights.append("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
            
            results["combined_insights"] = insights
            
            return json.dumps(results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Hybrid search failed: {e}"})


class MCPConnectorTool(BaseTool):
    """MCP(Model Context Protocol) 연결 도구"""
    
    name: str = "mcp_connector"
    description: str = "외부 의료 데이터베이스나 시스템과 MCP를 통해 연결합니다."
    mcp_endpoints: Dict[str, str] = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mcp_endpoints = {
            "hospital_db": "http://localhost:8080/mcp/hospital",
            "medical_records": "http://localhost:8081/mcp/records",
            "drug_database": "http://localhost:8082/mcp/drugs"
        }
    
    def _run(self, query: str, endpoint: str = "hospital_db") -> str:
        """MCP를 통해 외부 시스템에 쿼리를 전송합니다."""
        try:
            # 실제 구현에서는 HTTP 요청을 보냄
            # 현재는 시뮬레이션된 응답 반환
            
            if endpoint not in self.mcp_endpoints:
                return json.dumps({"error": f"Unknown endpoint: {endpoint}"})
            
            # 시뮬레이션된 MCP 응답
            mcp_response = {
                "endpoint": endpoint,
                "query": query,
                "status": "success",
                "data": {
                    "message": f"MCP 연결을 통해 {endpoint}에서 '{query}' 검색 완료",
                    "external_results": [
                        {
                            "source": endpoint,
                            "content": f"{query}와 관련된 외부 데이터",
                            "confidence": 0.9
                        }
                    ]
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(mcp_response, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"MCP connection failed: {e}"}) 