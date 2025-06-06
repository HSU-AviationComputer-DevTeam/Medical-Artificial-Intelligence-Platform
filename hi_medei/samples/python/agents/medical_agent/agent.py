"""Patient Data Manager Agent - Main agent implementation."""

import json
import os
from datetime import datetime
from typing import Any, AsyncIterable, Dict, List, Optional

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from medical_tools import (
    DrugInteractionCheckerTool,
    HybridSearchTool,
    MCPConnectorTool,
    PatientSearchTool,
    SOAPNoteGeneratorTool,
    UrgencyAssessmentTool,
    VectorSearchTool,
)
from models import MedicalRecord, PatientSearchQuery


class PatientDataManagerAgent:
    """환자 데이터 관리 및 진료문서 작성을 위한 의료 AI 에이전트"""
    
    SYSTEM_INSTRUCTION = """
    당신은 병원의 환자 데이터 관리 및 진료문서 작성을 담당하는 의료 AI 에이전트입니다.
    
    주요 역할:
    1. 환자 검색 및 정보 조회
    2. 유사 증상 환자 검색 (벡터 검색)
    3. SOAP 노트 자동 생성
    4. 약물 상호작용 검사
    5. 응급도 평가
    6. 진료 기록 분석 및 요약
    
    사용 가능한 도구들:
    - patient_search: 환자 ID, 이름, 증상으로 환자 검색
    - vector_search: 증상 기반 유사 환자 벡터 검색
    - soap_note_generator: SOAP 노트 자동 생성
    - drug_interaction_checker: 약물 상호작용 검사
    - urgency_assessment: 응급도 평가
    
    항상 의료 윤리를 준수하고, 정확하고 신뢰할 수 있는 정보를 제공하세요.
    불확실한 경우에는 전문의 상담을 권하세요.
    """
    
    def __init__(self, openai_api_key: str, data_path: str = "/Users/sindong-u/coding/project/hi_medei/data"):
        """에이전트를 초기화합니다."""
        self.openai_api_key = openai_api_key
        self.data_path = data_path
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # 도구들 초기화
        self.tools = self._initialize_tools()
        
        # 메모리 초기화
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 에이전트 생성 - 강제로 도구 사용하도록 하는 프롬프트
        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate.from_template("""
        당신은 병원의 환자 데이터 관리 및 진료문서 작성을 담당하는 의료 AI 에이전트입니다.
        
        **절대 규칙**: 
        - 환자나 의료 관련 질문에는 반드시 도구를 사용해야 합니다
        - 도구 없이 직접 답변하는 것은 금지됩니다
        - 추측하거나 가정하지 마세요
        
        **도구 사용 매핑**:
        - 환자 이름 (예: "홍길1") → patient_search 도구 필수
        - 질병명 (예: "당뇨병") → patient_search 도구 필수
        - 증상 관련 → hybrid_search 도구 필수
        
        사용 가능한 도구들:
        {tools}

        **형식을 정확히 따르세요**:

        Question: {input}
        Thought: I need to use a tool to answer this question about medical data.
        Action: [도구명을 {tool_names} 중에서 선택]
        Action Input: [도구에 전달할 입력]
        Observation: [도구 실행 결과]
        Thought: Now I have the information I need.
        Final Answer: [한국어로 최종 답변]

        Question: {input}
        Thought:{agent_scratchpad}""")
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="force"
        )
    
    def _initialize_tools(self) -> List:
        """도구들을 초기화합니다."""
        tools = []
        
        try:
            # 환자 검색 도구
            patient_search_tool = PatientSearchTool(data_path=self.data_path)
            tools.append(patient_search_tool)
            print(f"[DEBUG] PatientSearchTool 초기화 완료: {self.data_path}")
            
            # 벡터 검색 도구
            vector_search_tool = VectorSearchTool(openai_api_key=self.openai_api_key)
            tools.append(vector_search_tool)
            print(f"[DEBUG] VectorSearchTool 초기화 완료")
            
            # 하이브리드 검색 도구 (JSON + Vector)
            hybrid_search_tool = HybridSearchTool(
                data_path=self.data_path, 
                openai_api_key=self.openai_api_key
            )
            tools.append(hybrid_search_tool)
            print(f"[DEBUG] HybridSearchTool 초기화 완료")
            
            # MCP 연결 도구
            mcp_connector_tool = MCPConnectorTool()
            tools.append(mcp_connector_tool)
            print(f"[DEBUG] MCPConnectorTool 초기화 완료")
            
            # SOAP 노트 생성 도구
            soap_generator_tool = SOAPNoteGeneratorTool()
            tools.append(soap_generator_tool)
            print(f"[DEBUG] SOAPNoteGeneratorTool 초기화 완료")
            
            # 약물 상호작용 검사 도구
            drug_interaction_tool = DrugInteractionCheckerTool()
            tools.append(drug_interaction_tool)
            print(f"[DEBUG] DrugInteractionCheckerTool 초기화 완료")
            
            # 응급도 평가 도구
            urgency_assessment_tool = UrgencyAssessmentTool()
            tools.append(urgency_assessment_tool)
            print(f"[DEBUG] UrgencyAssessmentTool 초기화 완료")
            
        except Exception as e:
            print(f"도구 초기화 중 오류 발생: {e}")
        
        print(f"[DEBUG] 총 {len(tools)}개 도구 초기화 완료")
        return tools
    
    def invoke(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """동기 방식으로 쿼리를 처리합니다."""
        try:
            print(f"[DEBUG] 쿼리 수신: {query}")
            print(f"[DEBUG] 세션 ID: {session_id}")
            print(f"[DEBUG] 사용 가능한 도구 수: {len(self.tools)}")
            
            # 쿼리 분석해서 적절한 도구 직접 호출
            response_content = self._analyze_and_execute_query(query)
            print(f"[DEBUG] 응답 생성 완료: {response_content[:100]}...")
            
            # 메모리에 대화 추가
            from langchain.schema import AIMessage, HumanMessage
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response_content)
            
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": response_content,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_message = f"에이전트 실행 중 오류가 발생했습니다: {str(e)}"
            print(f"[ERROR] {error_message}")
            import traceback
            traceback.print_exc()
            
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": error_message,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_and_execute_query(self, query: str) -> str:
        """쿼리를 분석하고 적절한 도구를 실행합니다."""
        query_lower = query.lower()
        
        # 환자 이름 패턴 검색
        import re
        name_patterns = [r'홍길\d+', r'김철\d+', r'박민\d+', r'이영\d+', r'최수\d+']
        
        for pattern in name_patterns:
            if re.search(pattern, query):
                print(f"[DEBUG] 환자 이름 패턴 발견: {pattern}")
                # patient_search 도구 사용
                for tool in self.tools:
                    if tool.name == "patient_search":
                        result = tool._run(f"이름: {re.search(pattern, query).group()}")
                        result_data = json.loads(result)
                        if result_data["total_count"] > 0:
                            patient = result_data["patients"][0]
                            return f"환자 정보를 찾았습니다:\n이름: {patient['name']}\n나이: {patient['age']}세\n성별: {patient['gender']}\n진단: {patient['diagnosis']}\n처방: {patient.get('prescription', 'N/A')}\n혈압: {patient.get('blood_pressure', 'N/A')}\n방문일: {patient.get('visit_date', 'N/A')}"
                        else:
                            return f"'{re.search(pattern, query).group()}' 환자의 정보를 찾을 수 없습니다."
        
        # 질병명 검색 - 벡터 검색 우선!
        diseases = ['당뇨병', '고혈압', '담낭염', '위염', '감기', '독감', '환자']
        for disease in diseases:
            if disease in query:
                print(f"[DEBUG] 질병명 발견: {disease} - 벡터 검색 우선 실행")
                # hybrid_search 도구 사용 (벡터 + JSON 결합)
                for tool in self.tools:
                    if tool.name == "hybrid_search":
                        result = tool._run(query)
                        result_data = json.loads(result)
                        
                        # 벡터 검색 결과 확인
                        vector_results = result_data.get("vector_results", {})
                        json_results = result_data.get("json_results", {})
                        
                        response_parts = []
                        
                        # 벡터 검색 결과 (의미적 유사도)
                        if vector_results.get("total_found", 0) > 0:
                            response_parts.append("🧠 **벡터 검색 결과 (의미적 유사도)**:")
                            similar_cases = vector_results.get("similar_cases", [])[:3]
                            for i, case in enumerate(similar_cases, 1):
                                similarity = case.get("similarity_score", 0)
                                content = case.get("content", "")[:100]
                                response_parts.append(f"{i}. 유사도 {similarity:.2f}: {content}...")
                        
                        # JSON 구조화 검색 결과
                        if json_results.get("total_count", 0) > 0:
                            response_parts.append(f"\n📊 **구조화된 데이터 검색**: {json_results['total_count']}명 발견")
                            patients = json_results["patients"][:3]
                            for patient in patients:
                                match_fields = patient.get('match_fields', [])
                                response_parts.append(f"- {patient['name']} ({patient['age']}세, {patient['department']}) - 매칭: {', '.join(match_fields)}")
                        
                        if response_parts:
                            return "\n".join(response_parts)
                        else:
                            return f"'{disease}' 관련 환자나 사례를 찾을 수 없습니다."
        
        # 일반적인 환자 검색
        if any(keyword in query_lower for keyword in ['환자', '검색', '찾', '정보']):
            print(f"[DEBUG] 일반 검색 키워드 발견")
            # hybrid_search 도구 사용
            for tool in self.tools:
                if tool.name == "hybrid_search":
                    result = tool._run(query)
                    result_data = json.loads(result)
                    
                    insights = result_data.get("combined_insights", [])
                    json_count = result_data.get("json_results", {}).get("total_count", 0)
                    vector_count = result_data.get("vector_results", {}).get("total_found", 0)
                    
                    response = f"검색 결과:\n"
                    if json_count > 0:
                        response += f"- 구조화된 데이터에서 {json_count}명의 환자를 찾았습니다.\n"
                        patients = result_data["json_results"]["patients"][:3]  # 최대 3명
                        for patient in patients:
                            response += f"  • {patient['name']} - {patient.get('diagnosis', 'N/A')}\n"
                    
                    if vector_count > 0:
                        response += f"- 유사 증상 검색에서 {vector_count}개의 관련 사례를 찾았습니다.\n"
                    
                    return response if json_count > 0 or vector_count > 0 else "검색 결과가 없습니다."
        
        # 기본 응답
        return "의료 데이터 검색을 위해 환자 이름이나 질병명을 포함해서 질문해주세요. 예: '홍길1 환자 정보', '당뇨병 환자들'"
    
    async def stream(self, query: str, session_id: Optional[str] = None) -> AsyncIterable[Dict[str, Any]]:
        """스트리밍 방식으로 쿼리를 처리합니다."""
        try:
            # 처리 시작 알림
            yield {
                "type": "status",
                "content": "환자 데이터를 분석하고 있습니다...",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # 에이전트 실행
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # 최종 결과 반환
            yield {
                "type": "final_result",
                "content": result.get("output", "처리 완료"),
                "is_task_complete": True,
                "require_user_input": False,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"스트리밍 처리 중 오류: {str(e)}",
                "is_task_complete": True,
                "require_user_input": False,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def search_patients(self, query: str, query_type: str = "name", max_results: int = 10) -> Dict[str, Any]:
        """환자를 검색합니다."""
        try:
            patient_search_tool = PatientSearchTool(data_path=self.data_path)
            result = patient_search_tool._run(
                query=query,
                query_type=query_type,
                max_results=max_results
            )
            return json.loads(result)
        except Exception as e:
            return {"error": f"환자 검색 실패: {str(e)}"}
    
    def generate_soap_note(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """SOAP 노트를 생성합니다."""
        try:
            soap_generator = SOAPNoteGeneratorTool()
            result = soap_generator._run(json.dumps(patient_data, ensure_ascii=False))
            return json.loads(result)
        except Exception as e:
            return {"error": f"SOAP 노트 생성 실패: {str(e)}"}
    
    def check_drug_interactions(self, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """약물 상호작용을 검사합니다."""
        try:
            drug_checker = DrugInteractionCheckerTool()
            result = drug_checker._run(json.dumps(medications, ensure_ascii=False))
            return json.loads(result)
        except Exception as e:
            return {"error": f"약물 상호작용 검사 실패: {str(e)}"}
    
    def assess_urgency(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """응급도를 평가합니다."""
        try:
            urgency_tool = UrgencyAssessmentTool()
            result = urgency_tool._run(json.dumps(patient_data, ensure_ascii=False))
            return json.loads(result)
        except Exception as e:
            return {"error": f"응급도 평가 실패: {str(e)}"}
    
    def get_similar_cases(self, symptoms: str, k: int = 5) -> Dict[str, Any]:
        """유사한 증상의 환자 사례를 검색합니다."""
        try:
            vector_search_tool = VectorSearchTool(openai_api_key=self.openai_api_key)
            result = vector_search_tool._run(symptoms=symptoms, k=k)
            return json.loads(result)
        except Exception as e:
            return {"error": f"유사 사례 검색 실패: {str(e)}"}
    
    def collaborate_with_langgraph_agent(self, patient_data: Dict[str, Any], medical_question: str) -> str:
        """LangGraph 에이전트와 협업하기 위한 데이터를 준비합니다."""
        try:
            # 환자 데이터 요약
            patient_summary = self._create_patient_summary(patient_data)
            
            # LangGraph 에이전트에게 전달할 질문 구성
            collaboration_query = f"""
            환자 정보:
            {patient_summary}
            
            의료진 질문:
            {medical_question}
            
            위 환자 사례와 관련된 최신 의료 문헌이나 가이드라인을 검색하여 
            진단 및 치료에 도움이 되는 정보를 제공해주세요.
            """
            
            return collaboration_query
            
        except Exception as e:
            return f"협업 데이터 준비 실패: {str(e)}"
    
    def _create_patient_summary(self, patient_data: Dict[str, Any]) -> str:
        """환자 데이터 요약을 생성합니다."""
        summary_parts = []
        
        if 'name' in patient_data:
            summary_parts.append(f"환자명: {patient_data['name']}")
        
        if 'age' in patient_data:
            summary_parts.append(f"나이: {patient_data['age']}세")
        
        if 'gender' in patient_data:
            summary_parts.append(f"성별: {patient_data['gender']}")
        
        if 'chief_complaint' in patient_data:
            summary_parts.append(f"주 호소: {patient_data['chief_complaint']}")
        
        if 'symptoms' in patient_data:
            symptoms_text = ", ".join(patient_data['symptoms'])
            summary_parts.append(f"증상: {symptoms_text}")
        
        if 'vital_signs' in patient_data:
            vital_signs = patient_data['vital_signs']
            vital_text = []
            for key, value in vital_signs.items():
                vital_text.append(f"{key}: {value}")
            summary_parts.append(f"활력징후: {', '.join(vital_text)}")
        
        if 'medical_history' in patient_data:
            history_text = ", ".join(patient_data['medical_history'])
            summary_parts.append(f"과거 병력: {history_text}")
        
        return "\n".join(summary_parts)
    
    # A2A 프로토콜 호환성을 위한 속성들
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/json'] 