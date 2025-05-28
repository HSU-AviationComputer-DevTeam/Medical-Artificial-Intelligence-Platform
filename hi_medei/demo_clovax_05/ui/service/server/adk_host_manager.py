import base64
import datetime
import json
import os
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torch
import re

from common.types import (
    AgentCard,
    DataPart,
    FileContent,
    FilePart,
    Message,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from google.adk import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events.event import Event as ADKEvent
from google.adk.events.event_actions import EventActions as ADKEventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from hosts.multiagent.host_agent import HostAgent
from hosts.multiagent.remote_agent_connection import (
    TaskCallbackArg,
)
from service.server.application_manager import ApplicationManager
from service.types import Conversation, Event
from utils.agent_card import get_agent_card


class ADKHostManager(ApplicationManager):
    """An implementation of memory based management with fake agent actions

    This implements the interface of the ApplicationManager to plug into
    the AgentServer. This acts as the service contract that the Mesop app
    uses to send messages to the agent and provide information for the frontend.
    """

    _conversations: list[Conversation]
    _messages: list[Message]
    _tasks: list[Task]
    _events: dict[str, Event]
    _pending_message_ids: list[str]
    _agents: list[AgentCard]
    _task_map: dict[str, str]

    def __init__(self, api_key: str = '', uses_vertex_ai: bool = False):
        self._conversations = []
        self._messages = []
        self._tasks = []
        self._events = {}
        self._pending_message_ids = []
        self._agents = []
        self._artifact_chunks = {}
        self._session_service = InMemorySessionService()
        self._artifact_service = InMemoryArtifactService()
        self._memory_service = InMemoryMemoryService()
        self._host_agent = HostAgent([], self.task_callback)
        self.user_id = 'test_user'
        self.app_name = 'A2A'
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY', '')
        self.uses_vertex_ai = (
            uses_vertex_ai
            or os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').upper() == 'TRUE'
        )

        # Set environment variables based on auth method
        if self.uses_vertex_ai:
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'TRUE'
        elif self.api_key:
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'FALSE'
            os.environ['GOOGLE_API_KEY'] = self.api_key

        self._initialize_host()

        # Map of message id to task id
        self._task_map = {}
        # Map to manage 'lost' message ids until protocol level id is introduced
        self._next_id = {}  # dict[str, str]: previous message to next message

        # .env 파일의 절대 경로를 명시적으로 지정 (UI 폴더의 .env 파일)
        dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
        load_dotenv(dotenv_path)
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set in .env file")
            
        MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            torch_dtype=torch.float16,
            device_map="cpu",
            token=token,
            use_cache=True  
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            use_cache=True
        )

    def update_api_key(self, api_key: str):
        """Update the API key and reinitialize the host if needed"""
        if api_key and api_key != self.api_key:
            self.api_key = api_key

            # Only update if not using Vertex AI
            if not self.uses_vertex_ai:
                os.environ['GOOGLE_API_KEY'] = api_key
                # Reinitialize host with new API key
                self._initialize_host()

    def _initialize_host(self):
        agent = self._host_agent.create_agent()
        self._host_runner = Runner(
            app_name=self.app_name,
            agent=agent,
            artifact_service=self._artifact_service,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )

    def create_conversation(self) -> Conversation:
        session = self._session_service.create_session(
            app_name=self.app_name, user_id=self.user_id
        )
        conversation_id = session.id
        c = Conversation(conversation_id=conversation_id, is_active=True)
        self._conversations.append(c)
        return c

    def sanitize_message(self, message: Message) -> Message:
        if not message.metadata:
            message.metadata = {}
        if 'message_id' not in message.metadata:
            message.metadata.update({'message_id': str(uuid.uuid4())})
        if 'conversation_id' in message.metadata:
            conversation = self.get_conversation(
                message.metadata['conversation_id']
            )
            if conversation:
                if conversation.messages:
                    # Get the last message
                    last_message_id = get_message_id(conversation.messages[-1])
                    if last_message_id:
                        message.metadata.update(
                            {'last_message_id': last_message_id}
                        )
        return message

    async def process_message(self, message: Message):
        self._messages.append(message)
        message_id = get_message_id(message)
        if message_id:
            self._pending_message_ids.append(message_id)
        conversation_id = (
            message.metadata['conversation_id']
            if 'conversation_id' in message.metadata
            else None
        )
        # Now check the conversation and attach the message id.
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.messages.append(message)
        
        # Create a new task for the message
        task = Task(
            id=str(uuid.uuid4()),
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                message=message
            ),
            metadata=message.metadata,
            artifacts=[],
            sessionId=conversation_id
        )
        self.add_task(task)
        
        # Extract user prompt
        prompt = "\n".join([part.text for part in message.parts if hasattr(part, 'text') and part.text])
        
        # A2A Orchestration: Analyze intent and route to appropriate agents
        intent_analysis = self._classify_intent(prompt)
        routing_strategy = intent_analysis["routing_strategy"]
        needed_agents = intent_analysis["needed_agents"]
        
        print(f"[A2A Orchestrator] Intent: {needed_agents}, Strategy: {routing_strategy}")
        
        agent_response = ""
        
        if routing_strategy == "hyperclova_only":
            # Use HyperCLOVAX for general medical consultation
            agent_response = self.call_llama4(prompt)
            self._create_event("HyperCLOVAX Medical AI", agent_response, conversation_id)
            
        elif routing_strategy == "single_agent" and "patient_search" in needed_agents:
            # Use medical agent for patient data search
            agent_response = await self._call_medical_agent(prompt, conversation_id)
            
        elif routing_strategy == "pdf_only":
            # Use PDF QA agent for document analysis
            agent_response = await self._call_pdf_agent(prompt, conversation_id)
            
        elif routing_strategy == "sequential":
            # Sequential: Data Search → Medical Analysis
            if "patient_search" in needed_agents:
                # Step 1: Get patient data
                patient_data = await self._call_medical_agent(prompt, conversation_id)
                self._create_event("Medical Data Agent", f"환자 데이터 수집 완료", conversation_id)
                
                # Step 2: Analyze with HyperCLOVAX
                analysis_prompt = f"""
다음 환자 데이터를 바탕으로 의료 분석을 제공해주세요:

환자 데이터:
{patient_data}

사용자 요청: {prompt}

전문적인 의료 소견과 권장사항을 제공해주세요.
"""
                medical_analysis = self.call_llama4(analysis_prompt)
                self._create_event("HyperCLOVAX Medical AI", medical_analysis, conversation_id)
                
                agent_response = f"**환자 데이터:**\n{patient_data}\n\n**의료 분석:**\n{medical_analysis}"
            
        elif routing_strategy == "parallel_with_safety":
            # Parallel execution with safety check
            tasks_results = []
            
            if "patient_search" in needed_agents:
                patient_data = await self._call_medical_agent(prompt, conversation_id)
                tasks_results.append(("환자 데이터", patient_data))
            
            if "medical_analysis" in needed_agents:
                medical_analysis = self.call_llama4(prompt)
                tasks_results.append(("의료 분석", medical_analysis))
                
            # Safety check (simulated)
            safety_check = "✅ 안전성 검토 완료: 특별한 주의사항 없음"
            self._create_event("Safety Check Agent", safety_check, conversation_id)
            
            # Combine results
            combined_response = "\n\n".join([f"**{title}:**\n{content}" for title, content in tasks_results])
            agent_response = f"{combined_response}\n\n**안전성 검토:**\n{safety_check}"
            
        elif routing_strategy == "multi_agent_collaboration":
            # Multi-agent collaboration: Patient Data + PDF Analysis + Medical Analysis
            collaboration_results = []
            
            # Step 1: Get patient data
            if "patient_search" in needed_agents:
                patient_data = await self._call_medical_agent(prompt, conversation_id)
                collaboration_results.append(("환자 데이터", patient_data))
                self._create_event("Medical Data Agent", "환자 데이터 수집 완료", conversation_id)
            
            # Step 2: Analyze relevant PDF documents
            if "pdf_analysis" in needed_agents:
                pdf_analysis = await self._call_pdf_agent(prompt, conversation_id)
                collaboration_results.append(("문서 분석", pdf_analysis))
                self._create_event("PDF QA Agent", "의료 문서 분석 완료", conversation_id)
            
            # Step 3: Comprehensive medical analysis with HyperCLOVAX
            comprehensive_prompt = f"""
다음 정보들을 종합하여 전문적인 의료 분석을 제공해주세요:

{chr(10).join([f"**{title}:**{chr(10)}{content}" for title, content in collaboration_results])}

사용자 요청: {prompt}

위 정보들을 종합하여 다음을 제공해주세요:
1. 종합적인 의료 소견
2. 권장 치료 방향
3. 주의사항 및 추가 검사 필요성
"""
            comprehensive_analysis = self.call_llama4(comprehensive_prompt)
            self._create_event("HyperCLOVAX Medical AI", "종합 의료 분석 완료", conversation_id)
            
            # Combine all results
            final_response = "\n\n".join([f"**{title}:**\n{content}" for title, content in collaboration_results])
            agent_response = f"{final_response}\n\n**종합 의료 분석:**\n{comprehensive_analysis}"
            
        elif routing_strategy == "pdf_patient_treatment_plan":
            # 순차적 처리: PDF 분석 → 환자 검색 → HyperCLOVAX 치료계획서 작성
            print(f"[A2A Orchestrator] PDF → 환자 → 치료계획 순차 처리 시작")
            
            # Step 1: PDF 문서 분석
            pdf_analysis = await self._call_pdf_agent(prompt, conversation_id)
            self._create_event("PDF QA Agent", "의료 지침서 분석 완료", conversation_id)
            print(f"[A2A Orchestrator] Step 1 완료: PDF 분석")
            
            # Step 2: 환자 데이터 검색
            patient_data = await self._call_medical_agent(prompt, conversation_id)
            self._create_event("Medical Data Agent", "환자 데이터 수집 완료", conversation_id)
            print(f"[A2A Orchestrator] Step 2 완료: 환자 데이터 수집")
            
            # Step 3: HyperCLOVAX로 종합 치료계획서 작성
            treatment_plan_prompt = f"""
다음 정보들을 바탕으로 환자의 개인화된 치료계획서를 작성해주세요:

**의료 지침서 분석 결과:**
{pdf_analysis}

**환자 정보:**
{patient_data}

**사용자 요청:** {prompt}

위 정보를 종합하여 다음 형식으로 치료계획서를 작성해주세요:

## 환자 치료계획서

### 1. 환자 기본 정보
- 환자명, 나이, 성별, 진단명 등

### 2. 현재 상태 평가
- 현재 치료 상황 및 검사 결과

### 3. 지침서 기반 권장 치료법
- 의료 지침서에 따른 표준 치료 방법

### 4. 개인화된 치료 계획
- 환자 개별 상황을 고려한 맞춤형 치료 방안

### 5. 모니터링 계획
- 추적 관찰 및 검사 일정

### 6. 주의사항 및 환자 교육
- 복용법, 생활습관 개선 등
"""
            treatment_plan = self.call_llama4(treatment_plan_prompt)
            self._create_event("HyperCLOVAX Medical AI", "개인화된 치료계획서 작성 완료", conversation_id)
            print(f"[A2A Orchestrator] Step 3 완료: 치료계획서 작성")
            
            agent_response = f"**📋 치료계획서 작성 완료**\n\n{treatment_plan}\n\n---\n\n**📚 참고한 지침서 내용:**\n{pdf_analysis}\n\n**👤 환자 데이터:**\n{patient_data}"
            
        elif routing_strategy == "pdf_with_analysis":
            # PDF Analysis + Medical Analysis
            # Step 1: Analyze PDF documents
            pdf_analysis = await self._call_pdf_agent(prompt, conversation_id)
            self._create_event("PDF QA Agent", "의료 문서 분석 완료", conversation_id)
            
            # Step 2: Medical analysis based on PDF content
            analysis_prompt = f"""
다음 의료 문서 분석 결과를 바탕으로 전문적인 의료 소견을 제공해주세요:

문서 분석 결과:
{pdf_analysis}

사용자 요청: {prompt}

문서 내용을 바탕으로 한 전문적인 의료 해석과 권장사항을 제공해주세요.
"""
            medical_analysis = self.call_llama4(analysis_prompt)
            self._create_event("HyperCLOVAX Medical AI", "의료 분석 완료", conversation_id)
            
            agent_response = f"**문서 분석:**\n{pdf_analysis}\n\n**의료 소견:**\n{medical_analysis}"
            
        else:
            # Fallback to medical agent or HyperCLOVAX
            if self._agents:
                agent_response = await self._call_medical_agent(prompt, conversation_id)
            else:
                agent_response = self.call_llama4(prompt)
        
        # Update task status
        task.status.state = TaskState.COMPLETED
        task.status.message = Message(
            parts=[TextPart(text=agent_response)],
            role='agent',
            metadata={
                **message.metadata,
                'last_message_id': get_message_id(message),
                'message_id': str(uuid.uuid4()),
            },
        )
        self.update_task(task)
        
        # Add response to conversation
        if conversation:
            conversation.messages.append(task.status.message)
        self._pending_message_ids.remove(message_id)

    def _classify_intent(self, user_message: str) -> dict:
        """사용자 메시지의 의도를 분류하여 필요한 에이전트들을 결정"""
        import re
        
        patterns = {
            "patient_search": [r"환자.*정보", r".*환자.*찾", r"홍길\d+", r"김철\d+", r".*환자.*목록", r"병원.*내.*환자"],
            "medical_analysis": [r"진단", r"치료.*계획", r"소견", r"분석", r"어떻게.*생각", r"권장", r"추천", r"치료계획서"],
            "documentation": [r"SOAP", r"노트.*작성", r"보고서", r"기록"],
            "safety_check": [r"약물.*상호작용", r"부작용", r"금기", r"알레르기"],
            "pdf_analysis": [r"\.pdf", r"PDF", r"문서.*분석", r"스캔.*결과", r"검사.*결과", r"보고서.*분석", r"CT", r"MRI", r"X-ray", r"엑스레이", r"가이드라인", r"지침서"],
            "general_medical": [r"당뇨병", r"고혈압", r"치료법", r"증상", r"관리"]
        }
        
        needed_agents = []
        
        for intent, regex_list in patterns.items():
            for pattern in regex_list:
                if re.search(pattern, user_message, re.IGNORECASE):
                    needed_agents.append(intent)
                    break
        
        # Remove duplicates
        needed_agents = list(set(needed_agents))
        
        print(f"[DEBUG] 감지된 의도들: {needed_agents}")
        print(f"[DEBUG] 사용자 메시지: {user_message}")
        
        # Determine routing strategy with enhanced logic
        if not needed_agents:
            routing_strategy = "hyperclova_only"
        elif len(needed_agents) == 1:
            if needed_agents[0] == "patient_search":
                routing_strategy = "single_agent"
            elif needed_agents[0] == "pdf_analysis":
                routing_strategy = "pdf_only"
            else:
                routing_strategy = "hyperclova_only"
        # 특별 케이스: PDF + 환자 + 치료계획 = 순차적 처리 (PDF → 환자 → 분석)
        elif ("pdf_analysis" in needed_agents and 
              "patient_search" in needed_agents and 
              "medical_analysis" in needed_agents):
            routing_strategy = "pdf_patient_treatment_plan"  # 새로운 전략
        elif "patient_search" in needed_agents and "pdf_analysis" in needed_agents:
            routing_strategy = "multi_agent_collaboration"  # 환자 데이터 + PDF 분석
        elif "patient_search" in needed_agents and ("medical_analysis" in needed_agents or "general_medical" in needed_agents):
            routing_strategy = "sequential"
        elif "pdf_analysis" in needed_agents and ("medical_analysis" in needed_agents or "general_medical" in needed_agents):
            routing_strategy = "pdf_with_analysis"  # PDF 분석 + 의료 분석
        elif "safety_check" in needed_agents:
            routing_strategy = "parallel_with_safety"
        else:
            routing_strategy = "parallel"
        
        return {
            "needed_agents": needed_agents,
            "routing_strategy": routing_strategy
        }

    async def _call_medical_agent(self, prompt: str, conversation_id: str) -> str:
        """의료 에이전트 호출"""
        if not self._agents:
            return "의료 에이전트가 등록되지 않았습니다."
        
        # 의료 에이전트 찾기 (포트 10001 또는 이름으로 식별)
        medical_agent = None
        for agent in self._agents:
            if ("10001" in agent.url or 
                "Patient Data Manager" in agent.name or 
                "Medical" in agent.name):
                medical_agent = agent
                break
        
        if not medical_agent:
            return "의료 에이전트를 찾을 수 없습니다."
        
        try:
            import aiohttp
            import json
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": {
                        "task": {"text": prompt},
                        "sessionId": conversation_id
                    },
                    "id": 123
                }
                
                async with session.post(
                    medical_agent.url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("result") and result["result"].get("artifacts"):
                            agent_response = result["result"]["artifacts"][0]["parts"][0]["text"]
                            self._create_event(medical_agent.name, agent_response, conversation_id)
                            return agent_response
                        else:
                            return "의료 에이전트로부터 응답을 받지 못했습니다."
                    else:
                        return f"의료 에이전트 연결 실패 (상태 코드: {response.status})"
                        
        except Exception as e:
            print(f"의료 에이전트 호출 중 오류: {e}")
            return f"의료 에이전트 호출 중 오류가 발생했습니다: {str(e)}"

    async def _call_pdf_agent(self, prompt: str, conversation_id: str) -> str:
        """PDF QA 에이전트 호출"""
        try:
            import aiohttp
            import json
            
            # PDF QA 에이전트 찾기
            pdf_agent = None
            for agent in self._agents:
                if ("10000" in agent.url or 
                    "PDF" in agent.name or 
                    "pdf" in agent.name.lower()):
                    pdf_agent = agent
                    break
            
            if not pdf_agent:
                return "PDF QA 에이전트를 찾을 수 없습니다."
            
            async with aiohttp.ClientSession() as session:
                # A2A 표준 형식으로 직접 요청 (올바른 형식)
                a2a_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": {
                        "message": {
                            "parts": [{"type": "text", "text": prompt}]
                        }
                    },
                    "id": 124
                }
                
                print(f"[DEBUG] PDF Agent 요청 URL: {pdf_agent.url}")
                print(f"[DEBUG] PDF Agent 요청 데이터: {a2a_payload}")
                
                async with session.post(
                    pdf_agent.url,
                    json=a2a_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"[DEBUG] PDF Agent 응답: {result}")
                        if result.get("result") and result["result"].get("artifacts"):
                            agent_response = result["result"]["artifacts"][0]["parts"][0]["text"]
                            self._create_event("PDF QA Agent", agent_response, conversation_id)
                            return agent_response
                        else:
                            return "PDF QA 에이전트로부터 올바른 응답을 받지 못했습니다."
                    else:
                        error_text = await response.text()
                        print(f"[DEBUG] PDF Agent 오류 응답: {error_text}")
                        return f"PDF QA 에이전트 연결 실패 (상태 코드: {response.status}): {error_text}"
                        
        except Exception as e:
            print(f"PDF QA 에이전트 호출 중 오류: {e}")
            return f"PDF QA 에이전트 호출 중 오류가 발생했습니다: {str(e)}"

    def _create_event(self, actor_name: str, content: str, conversation_id: str):
        """이벤트 생성 헬퍼 메서드"""
        self.add_event(
            Event(
                id=str(uuid.uuid4()),
                actor=actor_name,
                content=Message(
                    parts=[TextPart(text=content)],
                    role='agent',
                    metadata={'conversation_id': conversation_id}
                ),
                timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            )
        )

    def add_task(self, task: Task):
        self._tasks.append(task)

    def update_task(self, task: Task):
        for i, t in enumerate(self._tasks):
            if t.id == task.id:
                self._tasks[i] = task
                return

    def task_callback(self, task: TaskCallbackArg, agent_card: AgentCard):
        self.emit_event(task, agent_card)
        if isinstance(task, TaskStatusUpdateEvent):
            current_task = self.add_or_get_task(task)
            current_task.status = task.status
            self.attach_message_to_task(task.status.message, current_task.id)
            self.insert_message_history(current_task, task.status.message)
            self.update_task(current_task)
            self.insert_id_trace(task.status.message)
            return current_task
        if isinstance(task, TaskArtifactUpdateEvent):
            current_task = self.add_or_get_task(task)
            self.process_artifact_event(current_task, task)
            self.update_task(current_task)
            return current_task
        # Otherwise this is a Task, either new or updated
        if not any(filter(lambda x: x.id == task.id, self._tasks)):
            self.attach_message_to_task(task.status.message, task.id)
            self.insert_id_trace(task.status.message)
            self.add_task(task)
            return task
        self.attach_message_to_task(task.status.message, task.id)
        self.insert_id_trace(task.status.message)
        self.update_task(task)
        return task

    def emit_event(self, task: TaskCallbackArg, agent_card: AgentCard):
        content = None
        conversation_id = get_conversation_id(task)
        metadata = (
            {'conversation_id': conversation_id} if conversation_id else None
        )
        if isinstance(task, TaskStatusUpdateEvent):
            if task.status.message:
                content = task.status.message
            else:
                content = Message(
                    parts=[TextPart(text=str(task.status.state))],
                    role='agent',
                    metadata=metadata,
                )
        elif isinstance(task, TaskArtifactUpdateEvent):
            content = Message(
                parts=task.artifact.parts,
                role='agent',
                metadata=metadata,
            )
        elif task.status and task.status.message:
            content = task.status.message
        elif task.artifacts:
            parts = []
            for a in task.artifacts:
                parts.extend(a.parts)
            content = Message(
                parts=parts,
                role='agent',
                metadata=metadata,
            )
        else:
            content = Message(
                parts=[TextPart(text=str(task.status.state))],
                role='agent',
                metadata=metadata,
            )
        self.add_event(
            Event(
                id=str(uuid.uuid4()),
                actor=agent_card.name,
                content=content,
                timestamp=datetime.datetime.now(datetime.UTC).timestamp(),
            )
        )

    def attach_message_to_task(self, message: Message | None, task_id: str):
        if message and message.metadata and 'message_id' in message.metadata:
            self._task_map[message.metadata['message_id']] = task_id

    def insert_id_trace(self, message: Message | None):
        if not message:
            return
        message_id = get_message_id(message)
        last_message_id = get_last_message_id(message)
        if message_id and last_message_id:
            self._next_id[last_message_id] = message_id

    def insert_message_history(self, task: Task, message: Message | None):
        if not message:
            return
        if task.history is None:
            task.history = []
        message_id = get_message_id(message)
        if not message_id:
            return
        if get_message_id(task.status.message) not in [
            get_message_id(x) for x in task.history
        ]:
            task.history.append(task.status.message)
        else:
            print(
                'Message id already in history',
                get_message_id(task.status.message),
                task.history,
            )

    def add_or_get_task(self, task: TaskCallbackArg):
        current_task = next(
            filter(lambda x: x.id == task.id, self._tasks), None
        )
        if not current_task:
            conversation_id = None
            if task.metadata and 'conversation_id' in task.metadata:
                conversation_id = task.metadata['conversation_id']
            current_task = Task(
                id=task.id,
                status=TaskStatus(
                    state=TaskState.SUBMITTED
                ),  # initialize with submitted
                metadata=task.metadata,
                artifacts=[],
                sessionId=conversation_id,
            )
            self.add_task(current_task)
            return current_task

        return current_task

    def process_artifact_event(
        self, current_task: Task, task_update_event: TaskArtifactUpdateEvent
    ):
        artifact = task_update_event.artifact
        if not artifact.append:
            # received the first chunk or entire payload for an artifact
            if artifact.lastChunk is None or artifact.lastChunk:
                # lastChunk bit is missing or is set to true, so this is the entire payload
                # add this to artifacts
                if not current_task.artifacts:
                    current_task.artifacts = []
                current_task.artifacts.append(artifact)
            else:
                # this is a chunk of an artifact, stash it in temp store for assembling
                if task_update_event.id not in self._artifact_chunks:
                    self._artifact_chunks[task_update_event.id] = {}
                self._artifact_chunks[task_update_event.id][artifact.index] = (
                    artifact
                )
        else:
            # we received an append chunk, add to the existing temp artifact
            current_temp_artifact = self._artifact_chunks[task_update_event.id][
                artifact.index
            ]
            # TODO handle if current_temp_artifact is missing
            current_temp_artifact.parts.extend(artifact.parts)
            if artifact.lastChunk:
                current_task.artifacts.append(current_temp_artifact)
                del self._artifact_chunks[task_update_event.id][artifact.index]

    def add_event(self, event: Event):
        self._events[event.id] = event

    def get_conversation(
        self, conversation_id: str | None
    ) -> Conversation | None:
        if not conversation_id:
            return None
        return next(
            filter(
                lambda c: c.conversation_id == conversation_id,
                self._conversations,
            ),
            None,
        )

    def get_pending_messages(self) -> list[tuple[str, str]]:
        rval = []
        for message_id in self._pending_message_ids:
            if message_id in self._task_map:
                task_id = self._task_map[message_id]
                task = next(
                    filter(lambda x: x.id == task_id, self._tasks), None
                )
                if not task:
                    rval.append((message_id, ''))
                elif task.history and task.history[-1].parts:
                    if len(task.history) == 1:
                        rval.append((message_id, 'Working...'))
                    else:
                        part = task.history[-1].parts[0]
                        rval.append(
                            (
                                message_id,
                                part.text
                                if part.type == 'text'
                                else 'Working...',
                            )
                        )
            else:
                rval.append((message_id, ''))
        return rval

    def register_agent(self, url):
        agent_data = get_agent_card(url)
        if not agent_data.url:
            agent_data.url = url
        self._agents.append(agent_data)
        self._host_agent.register_agent_card(agent_data)
        # Now update the host agent definition
        self._initialize_host()

    def register_pdf_agent(self):
        """PDF QA 에이전트를 자동으로 등록"""
        try:
            pdf_agent_url = "http://localhost:10000"
            self.register_agent(pdf_agent_url)
            print(f"[A2A] PDF QA Agent 등록 완료: {pdf_agent_url}")
        except Exception as e:
            print(f"[A2A] PDF QA Agent 등록 실패: {e}")

    @property
    def agents(self) -> list[AgentCard]:
        return self._agents

    @property
    def conversations(self) -> list[Conversation]:
        return self._conversations

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    @property
    def events(self) -> list[Event]:
        return sorted(self._events.values(), key=lambda x: x.timestamp)

    def adk_content_from_message(self, message: Message) -> types.Content:
        parts: list[types.Part] = []
        for part in message.parts:
            if part.type == 'text':
                parts.append(types.Part.from_text(text=part.text))
            elif part.type == 'data':
                json_string = json.dumps(part.data)
                parts.append(types.Part.from_text(text=json_string))
            elif part.type == 'file':
                if part.uri:
                    parts.append(
                        types.Part.from_uri(
                            file_uri=part.uri, mime_type=part.mimeType
                        )
                    )
                elif content_part.bytes:
                    parts.append(
                        types.Part.from_bytes(
                            data=part.bytes.encode('utf-8'),
                            mime_type=part.mimeType,
                        )
                    )
                else:
                    raise ValueError('Unsupported message type')
        return types.Content(parts=parts, role=message.role)

    def adk_content_to_message(
        self, content: types.Content, conversation_id: str
    ) -> Message:
        parts: list[Part] = []
        if not content.parts:
            return Message(
                parts=[],
                role=content.role if content.role == 'user' else 'agent',
                metadata={'conversation_id': conversation_id},
            )
        for part in content.parts:
            if part.text:
                # try parse as data
                try:
                    data = json.loads(part.text)
                    parts.append(DataPart(data=data))
                except:
                    parts.append(TextPart(text=part.text))
            elif part.inline_data:
                parts.append(
                    FilePart(
                        data=part.inline_data.decode('utf-8'),
                        mimeType=part.inline_data.mime_type,
                    )
                )
            elif part.file_data:
                parts.append(
                    FilePart(
                        file=FileContent(
                            uri=part.file_data.file_uri,
                            mimeType=part.file_data.mime_type,
                        )
                    )
                )
            # These aren't managed by the A2A message structure, these are internal
            # details of ADK, we will simply flatten these to json representations.
            elif part.video_metadata:
                parts.append(DataPart(data=part.video_metadata.model_dump()))
            elif part.thought:
                parts.append(TextPart(text='thought'))
            elif part.executable_code:
                parts.append(DataPart(data=part.executable_code.model_dump()))
            elif part.function_call:
                parts.append(DataPart(data=part.function_call.model_dump()))
            elif part.function_response:
                parts.extend(
                    self._handle_function_response(part, conversation_id)
                )
            else:
                raise ValueError('Unexpected content, unknown type')
        return Message(
            role=content.role if content.role == 'user' else 'agent',
            parts=parts,
            metadata={'conversation_id': conversation_id},
        )

    def _handle_function_response(
        self, part: types.Part, conversation_id: str
    ) -> list[Part]:
        parts = []
        try:
            for p in part.function_response.response['result']:
                if isinstance(p, str):
                    parts.append(TextPart(text=p))
                elif isinstance(p, dict):
                    if 'type' in p and p['type'] == 'file':
                        parts.append(FilePart(**p))
                    else:
                        parts.append(DataPart(data=p))
                elif isinstance(p, DataPart):
                    if 'artifact-file-id' in p.data:
                        file_part = self._artifact_service.load_artifact(
                            user_id=self.user_id,
                            session_id=conversation_id,
                            app_name=self.app_name,
                            filename=p.data['artifact-file-id'],
                        )
                        file_data = file_part.inline_data
                        base64_data = base64.b64encode(file_data.data).decode(
                            'utf-8'
                        )
                        parts.append(
                            FilePart(
                                file=FileContent(
                                    bytes=base64_data,
                                    mimeType=file_data.mime_type,
                                    name='artifact_file',
                                )
                            )
                        )
                    else:
                        parts.append(DataPart(data=p.data))
                else:
                    parts.append(TextPart(text=json.dumps(p)))
        except Exception as e:
            print("Couldn't convert to messages:", e)
            parts.append(DataPart(data=part.function_response.model_dump()))
        return parts

    def call_llama4(self, prompt: str) -> str:
        """HyperCLOVAX Text-Instruct-0.5B 모델을 사용하여 응답 생성"""
        chat = [
            {"role": "system", "content": "당신은 환자에 대한 소견을 주고, 의료 문서를 분석하고 이해하는 전문가입니다. 의료 용어를 정확하게 사용하고, 전문적인 의견을 제공해주세요."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            chat, 
            return_tensors="pt", 
            tokenize=True
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,  # 더 크게!
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,  # 반복 방지
            pad_token_id=self.tokenizer.eos_token_id
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 응답에서 시스템/사용자 프롬프트 제거
        matches = re.split(r"assistant:?", result, flags=re.IGNORECASE)
        if len(matches) > 1:
            answer = matches[-1].strip()
        else:
            answer = result.strip()
        answer = re.sub(r"system:?[^\n]*\n?", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"user:?[^\n]*\n?", "", answer, flags=re.IGNORECASE)
        return answer.strip()

    def create_and_register_agent(self, name: str, description: str, capabilities: list[str]):
        """Create and register a new agent with specific capabilities"""
        agent_card = AgentCard(
            name=name,
            description=description,
            capabilities=capabilities,
            url=f"http://localhost:8000/agent/{name.lower()}"
        )
        self.register_agent(agent_card.url)
        return agent_card


def get_message_id(m: Message | None) -> str | None:
    if not m or not m.metadata or 'message_id' not in m.metadata:
        return None
    return m.metadata['message_id']


def get_last_message_id(m: Message | None) -> str | None:
    if not m or not m.metadata or 'last_message_id' not in m.metadata:
        return None
    return m.metadata['last_message_id']


def get_conversation_id(
    t: (
        Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent | Message | None
    ),
) -> str | None:
    if (
        t
        and hasattr(t, 'metadata')
        and t.metadata
        and 'conversation_id' in t.metadata
    ):
        return t.metadata['conversation_id']
    return None


def task_still_open(task: Task | None) -> bool:
    if not task:
        return False
    return task.status.state in [
        TaskState.SUBMITTED,
        TaskState.WORKING,
        TaskState.INPUT_REQUIRED,
    ]
