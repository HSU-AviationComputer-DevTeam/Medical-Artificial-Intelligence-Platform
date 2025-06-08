"""Medical tools for patient search, document generation, and analysis."""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from models import (
    DepartmentType,
    DrugInteraction,
    MedicalAnalysis,
    MedicalRecord,
    Patient,
    PatientSearchQuery,
    PatientSearchResult,
    SOAPNote,
    UrgencyLevel,
)


class PatientSearchTool(BaseTool):
    """환자 검색 도구"""
    
    name: str = "patient_search"
    description: str = "환자 ID, 이름, 증상으로 환자를 검색합니다."
    data_path: str = Field(default="../../../../../VectorStore2/medical_data")
    patients_data: Dict[str, List[Dict]] = Field(default_factory=dict)
    
    def __init__(self, data_path: str = "../../../../../VectorStore2/medical_data", **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.patients_data = self._load_patient_data()
    
    def _load_patient_data(self) -> Dict[str, List[Dict]]:
        """환자 데이터를 로드합니다."""
        data = {}
        
        # VectorStore2/medical_data의 JSON 파일 매핑
        file_mappings = {
            "cardiology_patients.json": "심장내과",
            "emergency_patients.json": "응급의학과", 
            "internal_medicine_patients.json": "내과",
            "neurology_patients.json": "신경과",
            "surgery_patients.json": "외과"
        }
        
        print(f"Loading patient data from: {self.data_path}")
        
        for filename, department in file_mappings.items():
            file_path = os.path.join(self.data_path, filename)
            print(f"Checking file: {file_path}")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        patient_list = json.load(f)
                        dept_data = []
                        
                        # JSON 파일이 리스트인 경우 각 환자에 department 추가
                        if isinstance(patient_list, list):
                            for patient in patient_list:
                                patient['department'] = department
                                dept_data.append(patient)
                        else:
                            # 단일 환자 객체인 경우
                            patient_list['department'] = department
                            dept_data.append(patient_list)
                        
                        data[department] = dept_data
                        print(f"✅ Loaded {filename}: {len(dept_data)} patients")
                        
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
            else:
                print(f"❌ File not found: {file_path}")
        
        print(f"\n📊 총 로드된 데이터: {len(data)} 부서")
        total_patients = 0
        for dept, patients in data.items():
            patient_count = len(patients)
            total_patients += patient_count
            print(f"  🏥 {dept}: {patient_count}명")
        print(f"  👥 전체 환자 수: {total_patients}명\n")
        
        return data
    
    def _run(self, query: str, query_type: str = "auto", max_results: int = 10) -> str:
        """환자 검색을 실행합니다."""
        start_time = time.time()
        results = []
        
        query_lower = query.lower()
        
        # 자동 쿼리 타입 감지
        if query_type == "auto":
            query_type = self._detect_query_type(query_lower)
        
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
        
        print(f"🔍 검색 실행: '{query}' (타입: {query_type})")
        
        # 복합 검색: 여러 필드에서 동시 검색
        for dept, patients in self.patients_data.items():
            for patient in patients:
                match_found = False
                match_fields = []
                
                # 이름 검색
                if query_type in ["name", "all"] and query_lower in patient.get('name', '').lower():
                    match_found = True
                    match_fields.append("이름")
                
                # ID 검색
                if query_type in ["id", "all"] and query_lower in patient.get('id', '').lower():
                    match_found = True
                    match_fields.append("ID")
                
                # 진단 검색 (의료 용어 매핑 포함)
                if query_type in ["diagnosis", "all"]:
                    # diagnoses 배열에서 검색
                    diagnoses = patient.get('diagnoses', [])
                    if isinstance(diagnoses, list):
                        for diag in diagnoses:
                            if isinstance(diag, dict):
                                diag_name = diag.get('name', '').lower()
                                if self._matches_medical_term(query_lower, diag_name):
                                    match_found = True
                                    match_fields.append("진단")
                                    break
                    
                    # 단일 diagnosis 필드도 확인 (하위 호환성)
                    diagnosis = patient.get('diagnosis', '').lower()
                    if diagnosis and self._matches_medical_term(query_lower, diagnosis):
                        match_found = True
                        match_fields.append("진단")
                
                # 증상 검색
                if query_type in ["symptom", "all"]:
                    symptoms = patient.get('symptoms', [])
                    if isinstance(symptoms, list):
                        for symptom in symptoms:
                            if self._matches_medical_term(query_lower, symptom.lower()):
                                match_found = True
                                match_fields.append("증상")
                                break
                    elif isinstance(symptoms, str) and self._matches_medical_term(query_lower, symptoms.lower()):
                        match_found = True
                        match_fields.append("증상")
                
                if match_found:
                    patient_result = patient.copy()
                    patient_result['match_fields'] = match_fields
                    results.append(patient_result)
        
        search_time = time.time() - start_time
        
        # 결과 제한
        results = results[:max_results]
        
        print(f"✅ 검색 완료: {len(results)}명 발견 ({search_time:.3f}초)")
        
        return json.dumps({
            "patients": results,
            "total_count": len(results),
            "search_time": search_time,
            "query": query,
            "query_type": query_type,
            "search_performed": True
        }, ensure_ascii=False, indent=2)
    
    def _detect_query_type(self, query: str) -> str:
        """쿼리 타입을 자동으로 감지합니다."""
        # 의료 용어나 증상 관련 키워드 감지
        medical_terms = ['환자', '병', '질환', '증상', '통증', '열', '기침', '두통', '복통', '당뇨', '고혈압', '암', '심장', '폐', '간', '신장']
        
        if any(term in query for term in medical_terms):
            if '환자' in query:
                return "diagnosis"  # "당뇨병환자" -> 진단명으로 검색
            else:
                return "symptom"
        
        # ID 패턴 감지 (P001, H123 등)
        if len(query) <= 10 and any(char.isdigit() for char in query):
            return "id"
        
        # 기본값은 전체 검색
        return "all"
    
    def _matches_medical_term(self, search_term: str, target_text: str) -> bool:
        """의료 용어 매칭 (유사어 및 부분 매칭 포함)"""
        # 직접 매칭
        if search_term in target_text:
            return True
        
        # 의료 용어 매핑
        medical_mappings = {
            '당뇨': ['diabetes', '당뇨병', 'dm', '혈당'],
            '고혈압': ['hypertension', '혈압', 'htn', '고혈압증'],
            '심장': ['cardiac', 'heart', '심근', '관상동맥'],
            '암': ['cancer', '종양', 'tumor', 'carcinoma'],
            '열': ['fever', '발열', '체온'],
            '통증': ['pain', '아픔', 'ache'],
            '기침': ['cough', '해수'],
            '호흡': ['breathing', 'respiratory', '숨', '호흡곤란']
        }
        
        # 매핑된 용어로 검색
        for key, synonyms in medical_mappings.items():
            if key in search_term:
                for synonym in synonyms:
                    if synonym in target_text:
                        return True
        
        return False


class GeminiEmbeddings:
    """Gemini 임베딩 클래스 - LangChain 호환"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.api_key = api_key
    
    def __call__(self, text):
        """LangChain 호환을 위한 callable 메서드"""
        return self.embed_query(text)
    
    def embed_documents(self, texts):
        """문서 임베딩 (사용하지 않음)"""
        raise NotImplementedError("이 메서드는 사용하지 않습니다.")
    
    def embed_query(self, text):
        """쿼리 임베딩"""
        try:
            # 원래 노트북에서 사용한 정확한 모델 사용 (3072차원)
            print(f"🔍 Gemini 임베딩 실행: gemini-embedding-exp-03-07")
            result = genai.embed_content(
                model="models/gemini-embedding-exp-03-07",
                content=text,
                task_type="retrieval_query"  # 노트북에서는 retrieval_document였지만 쿼리에는 retrieval_query
            )
            embedding = result['embedding']
            print(f"✅ 임베딩 성공! 차원: {len(embedding)}")
            return embedding
        except Exception as e:
            print(f"❌ Gemini 임베딩 오류 (gemini-embedding-exp-03-07): {e}")
            # text-embedding-004로 fallback 시도 
            try:
                print("🔄 text-embedding-004 모델로 fallback...")
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_query"
                )
                embedding = result['embedding']
                print(f"⚠️ Fallback 성공! 차원: {len(embedding)} (차원 불일치 가능)")
                return embedding
            except Exception as e2:
                print(f"❌ 모든 Gemini 모델 실패: {e2}")
                raise


class VectorSearchTool(BaseTool):
    """벡터 검색 도구 - 유사 증상 환자 검색"""
    
    name: str = "vector_search"
    description: str = "증상을 기반으로 유사한 환자를 벡터 검색으로 찾습니다."
    openai_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    vectorstore: Optional[str] = Field(default=None)
    
    def __init__(self, openai_api_key: str = "", gemini_api_key: str = "", **kwargs):
        super().__init__(**kwargs)
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """벡터스토어를 초기화합니다."""
        try:
            # 실제 벡터스토어 경로 사용
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "../../../../..")
            vector_path = os.path.join(project_root, "VectorStore2", "vector_stores", "medical_vector_store")
            vector_path = os.path.abspath(vector_path)
            
            print(f"🔍 벡터스토어 경로: {vector_path}")
            
            # FAISS 벡터스토어 로드 시도
            try:
                from langchain_community.vectorstores import FAISS

                # 강제로 Gemini 우선 시도 (3072차원 벡터스토어용)
                if self.gemini_api_key:
                    print("🌟 Gemini 임베딩으로 벡터스토어 로드 시도 (3072차원)")
                    try:
                        embeddings = GeminiEmbeddings(api_key=self.gemini_api_key)
                        # FAISS 벡터스토어 로드
                        self.vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
                        print("✅ FAISS 벡터스토어 로드 성공! (Gemini 3072차원)")
                        return
                    except Exception as gemini_error:
                        print(f"❌ Gemini 로드 실패: {gemini_error}")
                        print("🔄 OpenAI로 fallback 시도...")
                
                # OpenAI API 키가 있으면 OpenAI 임베딩 사용 (1536차원) - Fallback
                if self.openai_api_key:
                    print("🔑 OpenAI 임베딩으로 벡터스토어 로드 시도")
                    from langchain_openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                    # FAISS 벡터스토어 로드
                    self.vectorstore = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization=True)
                    print("✅ FAISS 벡터스토어 로드 성공! (OpenAI 1536차원)")
                    return
                else:
                    print("⚠️ API 키가 없어 벡터스토어를 로드할 수 없습니다.")
            except Exception as e:
                print(f"❌ FAISS 로드 실패: {e}")
            
            # 대안: ChromaDB 시도
            try:
                # ChromaDB 벡터스토어 로드 시도
                chroma_path = os.path.join(vector_path, "chroma_db")
                if os.path.exists(chroma_path):
                    print(f"🔍 ChromaDB 시도: {chroma_path}")
                    self.vectorstore = "chroma_available"
                    print("✅ ChromaDB 벡터스토어 발견!")
                    return
            except Exception as e:
                print(f"❌ ChromaDB 로드 실패: {e}")
            
            # 파일 확인
            index_path = os.path.join(vector_path, "index.faiss")
            pkl_path = os.path.join(vector_path, "index.pkl")
            
            print(f"📁 FAISS 파일 확인:")
            print(f"  - index.faiss: {os.path.exists(index_path)}")
            print(f"  - index.pkl: {os.path.exists(pkl_path)}")
            
            if os.path.exists(index_path) and os.path.exists(pkl_path):
                self.vectorstore = "faiss_files_found"
                print("✅ 벡터스토어 파일들을 발견했습니다!")
            else:
                print("❌ 벡터스토어 파일을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ Vector store initialization error: {e}")
    
    def _run(self, symptoms: str, k: int = 5) -> str:
        """유사 증상 환자를 검색합니다."""
        if not self.vectorstore:
            return json.dumps({"error": "Vector store not available"}, ensure_ascii=False)
        
        try:
            # 실제 FAISS 벡터스토어가 로드된 경우
            if hasattr(self.vectorstore, 'similarity_search_with_score'):
                print(f"🔍 벡터 검색 실행: '{symptoms}'")
                
                # 벡터 검색 수행
                docs_with_scores = self.vectorstore.similarity_search_with_score(symptoms, k=k)
                
                results = []
                for doc, score in docs_with_scores:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(1 - score)  # 거리를 유사도로 변환
                    })
                
                print(f"✅ 벡터 검색 완료: {len(results)}개 결과 발견")
                
                return json.dumps({
                    "similar_cases": results,
                    "query": symptoms,
                    "total_found": len(results),
                    "source": "FAISS VectorStore",
                    "search_type": "semantic_vector_search"
                }, ensure_ascii=False, indent=2)
            
            # 벡터스토어 파일이 있지만 차원 불일치 등으로 로드되지 않은 경우
            elif self.vectorstore in ["faiss_files_found", "chroma_available"] or not hasattr(self.vectorstore, 'similarity_search_with_score'):
                print(f"🧠 벡터 검색 대신 향상된 의미적 키워드 검색 사용")
                
                # 향상된 의미적 검색 - 다양한 키워드로 확장
                expanded_queries = self._expand_medical_query(symptoms)
                print(f"🔍 확장된 검색어: {expanded_queries}")
                
                all_results = []
                for query in expanded_queries:
                    # 증상 기반 검색
                    symptom_results = self._search_json_data(query, "symptom")
                    # 진단 기반 검색  
                    diagnosis_results = self._search_json_data(query, "diagnosis")
                    all_results.extend(symptom_results + diagnosis_results)
                
                # 중복 제거 및 관련성 스코어 계산
                unique_results = self._deduplicate_and_score(all_results, symptoms)
                
                return json.dumps({
                    "similar_cases": unique_results[:k],
                    "query": symptoms,
                    "total_found": len(unique_results),
                    "source": "향상된 의미적 키워드 검색",
                    "search_type": "enhanced_semantic_keyword_search",
                    "expanded_queries": expanded_queries
                }, ensure_ascii=False, indent=2)
            
            else:
                return json.dumps({
                    "error": "Vector store not properly initialized",
                    "vectorstore_status": str(self.vectorstore)
                }, ensure_ascii=False)
            
        except Exception as e:
            print(f"❌ 벡터 검색 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return json.dumps({"error": f"Vector search failed: {str(e)}"}, ensure_ascii=False)
    
    def _expand_medical_query(self, query: str) -> List[str]:
        """의료 쿼리를 의미적으로 확장합니다."""
        expanded = [query]  # 원본 포함
        
        # 의료 용어 확장 매핑
        medical_expansions = {
            '당뇨병': ['diabetes', '혈당', '인슐린', '당뇨', 'dm', '제2형 당뇨병', '제1형 당뇨병'],
            '고혈압': ['hypertension', '혈압', 'htn', '고혈압증', '수축기', '이완기'],
            '심장': ['cardiac', 'heart', '심근', '관상동맥', '심혈관', '부정맥'],
            '당뇨': ['diabetes', '혈당', '인슐린', '당뇨병', 'dm'],
            '통증': ['pain', '아픔', 'ache', '불편감', '압박감'],
            '호흡': ['respiratory', 'breathing', '숨', '호흡곤란', '기침', '폐'],
            '혈당': ['glucose', 'sugar', '당뇨', 'diabetes', 'blood sugar']
        }
        
        # 증상 관련 확장
        symptom_expansions = {
            '다뇨': ['빈뇨', '소변', '화장실', '야간뇨'],
            '다갈': ['갈증', '목마름', '수분'],
            '시야': ['시력', '눈', '망막', '시야장애'],
            '피로': ['무력감', '기력', '체력', '무기력'],
            '체중': ['몸무게', '살', '비만', '체중감소', '체중증가']
        }
        
        # 쿼리에서 키워드 찾아서 확장
        query_lower = query.lower()
        for keyword, expansions in {**medical_expansions, **symptom_expansions}.items():
            if keyword in query_lower:
                expanded.extend(expansions)
        
        return list(set(expanded))  # 중복 제거
    
    def _search_json_data(self, query: str, search_type: str) -> List[Dict]:
        """JSON 데이터에서 검색을 수행합니다."""
        from medical_tools import PatientSearchTool
        patient_search = PatientSearchTool()
        
        try:
            result = patient_search._run(query, query_type=search_type)
            result_data = json.loads(result)
            return result_data.get("patients", [])
        except:
            return []
    
    def _deduplicate_and_score(self, results: List[Dict], original_query: str) -> List[Dict]:
        """결과 중복 제거 및 관련성 스코어 계산"""
        seen_ids = set()
        unique_results = []
        
        for patient in results:
            patient_id = patient.get('id', '')
            if patient_id and patient_id not in seen_ids:
                seen_ids.add(patient_id)
                
                # 관련성 스코어 계산 (간단한 키워드 매칭)
                score = self._calculate_relevance_score(patient, original_query)
                
                # 벡터 검색 형태로 변환
                unique_results.append({
                    "content": self._patient_to_content(patient),
                    "metadata": {
                        "patient_id": patient_id,
                        "name": patient.get('name', ''),
                        "department": patient.get('department', ''),
                        "relevance_type": "keyword_match"
                    },
                    "similarity_score": score
                })
        
        # 스코어 기준으로 정렬
        unique_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return unique_results
    
    def _calculate_relevance_score(self, patient: Dict, query: str) -> float:
        """환자와 쿼리의 관련성 스코어를 계산합니다."""
        score = 0.0
        query_lower = query.lower()
        
        # 진단명 매칭 (높은 가중치)
        diagnoses = patient.get('diagnoses', [])
        for diag in diagnoses:
            if isinstance(diag, dict):
                diag_name = diag.get('name', '').lower()
                if query_lower in diag_name or any(word in diag_name for word in query_lower.split()):
                    score += 0.9
        
        # 증상 매칭
        symptoms = patient.get('symptoms', [])
        if isinstance(symptoms, list):
            for symptom in symptoms:
                if query_lower in symptom.lower():
                    score += 0.7
        
        # 약물 매칭
        medications = patient.get('medications', [])
        for med in medications:
            if isinstance(med, dict):
                med_name = med.get('medication', '').lower()
                if query_lower in med_name:
                    score += 0.5
        
        # 기본 스코어 (0.1-1.0 범위로 정규화)
        return min(max(score, 0.1), 1.0)
    
    def _patient_to_content(self, patient: Dict) -> str:
        """환자 정보를 검색 가능한 텍스트로 변환합니다."""
        content_parts = []
        
        # 기본 정보
        content_parts.append(f"환자: {patient.get('name', '')} ({patient.get('age', '')}세)")
        
        # 진단명
        diagnoses = patient.get('diagnoses', [])
        if diagnoses:
            diag_names = [d.get('name', '') for d in diagnoses if isinstance(d, dict)]
            content_parts.append(f"진단: {', '.join(diag_names)}")
        
        # 증상
        symptoms = patient.get('symptoms', [])
        if symptoms:
            symptoms_text = ', '.join(symptoms) if isinstance(symptoms, list) else str(symptoms)
            content_parts.append(f"증상: {symptoms_text}")
        
        # 부서
        if patient.get('department'):
            content_parts.append(f"부서: {patient['department']}")
        
        return ' | '.join(content_parts)


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
    gemini_api_key: str = Field(default="")
    patient_search: Optional[PatientSearchTool] = Field(default=None)
    vector_search: Optional[VectorSearchTool] = Field(default=None)
    
    def __init__(self, data_path: str, openai_api_key: str = "", gemini_api_key: str = "", **kwargs):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.patient_search = PatientSearchTool(data_path=data_path)
        self.vector_search = VectorSearchTool(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key
        )
    
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
        # 실제 MCP 서버 엔드포인트로 수정
        self.mcp_endpoints = {
            "pubmed": "http://localhost:8080",
            "memory": "http://localhost:8081", 
            "hospital_db": "http://localhost:8080",
            "medical_records": "http://localhost:8081"
        }
    
    async def _call_mcp_async(self, endpoint_url: str, query: str) -> Dict[str, Any]:
        """비동기 MCP 호출"""
        try:
            # MCP JSON-RPC 요청 포맷
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "search",
                "params": {
                    "query": query,
                    "context": "medical_data"
                },
                "id": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint_url,
                    json=mcp_request,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        return {
                            "error": f"HTTP {response.status}: {await response.text()}"
                        }
        except Exception as e:
            return {"error": f"MCP 연결 실패: {str(e)}"}
    
    def _call_mcp_sync(self, endpoint_url: str, query: str) -> Dict[str, Any]:
        """동기 MCP 호출 - 실제 MCP 서버 API 형식"""
        try:
            # PubMed 서버인 경우
            if "8080" in endpoint_url:
                mcp_request = {
                    "parameters": {
                        "query": query,
                        "max_results": 5
                    }
                }
                response = requests.post(
                    f"{endpoint_url}/tools/search_pubmed",
                    json=mcp_request,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
            # Memory 서버인 경우  
            elif "8081" in endpoint_url:
                mcp_request = {
                    "parameters": {
                        "session_id": "agent_session",
                        "content": f"Agent searched for: {query}",
                        "entry_type": "agent_search"
                    }
                }
                response = requests.post(
                    f"{endpoint_url}/tools/save_memory",
                    json=mcp_request,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
            else:
                return {"error": f"Unknown MCP server: {endpoint_url}"}
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "mcp_result": response.json(),
                    "endpoint": endpoint_url
                }
            else:
                return {
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {"error": f"MCP 연결 실패: {str(e)}"}
    
    def _run(self, query: str, endpoint: str = "hospital_db") -> str:
        """MCP를 통해 외부 시스템에 쿼리를 전송합니다."""
        try:
            if endpoint not in self.mcp_endpoints:
                return json.dumps({"error": f"Unknown endpoint: {endpoint}"})
            
            endpoint_url = self.mcp_endpoints[endpoint]
            
            # 실제 MCP 서버가 동작중인지 확인하고 호출
            try:
                # 먼저 ping을 통해 서버 상태 확인
                ping_response = requests.get(f"{endpoint_url}/health", timeout=5)
                if ping_response.status_code == 200:
                    # 실제 MCP 서버가 있으면 호출
                    mcp_response = self._call_mcp_sync(endpoint_url, query)
                else:
                    # 서버가 없으면 시뮬레이션 응답
                    mcp_response = self._create_simulation_response(endpoint, query)
            except (requests.ConnectionError, requests.Timeout):
                # 연결 실패시 시뮬레이션 응답
                mcp_response = self._create_simulation_response(endpoint, query)
            
            return json.dumps(mcp_response, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"MCP connection failed: {e}"})
    
    def _create_simulation_response(self, endpoint: str, query: str) -> Dict[str, Any]:
        """MCP 서버가 없을 때 시뮬레이션 응답 생성"""
        return {
            "jsonrpc": "2.0",
            "result": {
                "endpoint": endpoint,
                "query": query,
                "status": "simulated",
                "data": {
                    "message": f"MCP 시뮬레이션: {endpoint}에서 '{query}' 검색 완료",
                    "external_results": [
                        {
                            "source": endpoint,
                            "content": f"{query}와 관련된 외부 의료 데이터 (시뮬레이션)",
                            "confidence": 0.8,
                            "type": "simulation"
                        }
                    ]
                },
                "timestamp": datetime.now().isoformat(),
                "simulation": True
            },
            "id": 1
        } 