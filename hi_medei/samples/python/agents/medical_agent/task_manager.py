"""Task manager for A2A protocol compatibility."""

import asyncio
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterable, Dict, List, Optional

from agent import PatientDataManagerAgent


class TaskState(str, Enum):
    """A2A 표준 태스크 상태"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class TaskManager:
    """A2A 프로토콜 호환 태스크 매니저"""
    
    def __init__(self, agent: PatientDataManagerAgent):
        self.agent = agent
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    async def send_task(
        self,
        task_id: str,
        session_id: Optional[str] = None,
        message: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """A2A 표준 tasks/send 메서드 구현"""
        
        # 태스크 초기화
        task = {
            "id": task_id,
            "sessionId": session_id,
            "status": {
                "state": TaskState.SUBMITTED,
                "timestamp": datetime.now().isoformat()
            },
            "message": message,
            "artifacts": [],
            "history": []
        }
        
        self.tasks[task_id] = task
        
        # 세션 초기화 (필요한 경우)
        if session_id and session_id not in self.sessions:
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
                "tasks": []
            }
        
        if session_id:
        self.sessions[session_id]["tasks"].append(task_id)
        
        try:
            # 태스크 실행
            task["status"]["state"] = TaskState.WORKING
            task["status"]["timestamp"] = datetime.now().isoformat()
            
            # 메시지에서 텍스트 추출
            user_input = self._extract_text_from_message(message)
            print(f"[DEBUG] 추출된 사용자 입력: {user_input}")
            print(f"[DEBUG] 세션 ID: {session_id}")
            
            # 에이전트 실행
            result = self.agent.invoke(user_input, session_id or "default")
            print(f"[DEBUG] 에이전트 실행 결과: {result}")
            
            # 결과 처리
            if result.get("is_task_complete", False):
                task["status"]["state"] = TaskState.COMPLETED
                task["artifacts"] = [{
                    "parts": [{
                        "type": "text",
                        "text": result.get("content", "")
                    }],
                    "index": 0
                }]
            elif result.get("require_user_input", False):
                task["status"]["state"] = TaskState.INPUT_REQUIRED
            else:
                task["status"]["state"] = TaskState.FAILED
                task["artifacts"] = [{
                    "parts": [{
                        "type": "text",
                        "text": "태스크 처리 중 오류가 발생했습니다."
                    }],
                    "index": 0
                }]
            
            task["status"]["timestamp"] = datetime.now().isoformat()
            
        except Exception as e:
            task["status"]["state"] = TaskState.FAILED
            task["status"]["timestamp"] = datetime.now().isoformat()
            task["artifacts"] = [{
                "parts": [{
                    "type": "text",
                    "text": f"태스크 실행 중 오류 발생: {str(e)}"
                }],
                "index": 0
            }]
        
        return task
    
    async def send_task_subscribe(
        self,
        task_id: str,
        session_id: Optional[str] = None,
        message: Dict[str, Any] = None,
        **kwargs
    ) -> AsyncIterable[Dict[str, Any]]:
        """A2A 표준 tasks/sendSubscribe 메서드 구현 (스트리밍)"""
        
        # 태스크 초기화
        task = {
            "id": task_id,
            "sessionId": session_id,
            "status": {
                "state": TaskState.SUBMITTED,
                "timestamp": datetime.now().isoformat()
            },
            "message": message,
            "artifacts": [],
            "history": []
        }
        
        self.tasks[task_id] = task
        
        try:
            # 시작 상태 전송
            task["status"]["state"] = TaskState.WORKING
            task["status"]["timestamp"] = datetime.now().isoformat()
            
            yield {
                "id": task["id"],
                "status": task["status"],
                "final": False
            }
            
            # 메시지에서 텍스트 추출
            user_input = self._extract_text_from_message(message)
            
            # 에이전트 스트리밍 실행
            final_content = ""
            if hasattr(self.agent, 'stream'):
                async for chunk in self.agent.stream(user_input, session_id or "default"):
                if chunk.get("type") == "final_result":
                    final_content = chunk.get("content", "")
                    break
                elif chunk.get("type") == "status":
                    # 중간 상태 업데이트
                    yield {
                        "id": task["id"],
                        "status": {
                                "state": TaskState.WORKING,
                            "timestamp": datetime.now().isoformat(),
                                "message": {
                                    "role": "assistant",
                                    "parts": [{
                                        "type": "text",
                                        "text": chunk.get("content", "")
                                    }]
                                }
                        },
                            "final": False
                        }
            else:
                # 스트리밍을 지원하지 않는 경우 일반 invoke 사용
                result = self.agent.invoke(user_input, session_id or "default")
                final_content = result.get("content", "")
            
            # 최종 결과 처리
            task["status"]["state"] = TaskState.COMPLETED
            task["status"]["timestamp"] = datetime.now().isoformat()
            task["artifacts"] = [{
                "parts": [{
                    "type": "text",
                    "text": final_content
                }],
                "index": 0
            }]
            
            yield {
                "id": task["id"],
                "status": task["status"],
                "artifacts": task["artifacts"],
                "final": True
            }
            
        except Exception as e:
            task["status"]["state"] = TaskState.FAILED
            task["status"]["timestamp"] = datetime.now().isoformat()
            task["artifacts"] = [{
                "parts": [{
                    "type": "text",
                    "text": f"스트리밍 처리 중 오류 발생: {str(e)}"
                }],
                "index": 0
            }]
            
            yield {
                "id": task["id"],
                "status": task["status"],
                "artifacts": task["artifacts"],
                "final": True
            }
    
    async def get_task(self, task_id: str, history_length: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """A2A 표준 tasks/get 메서드 구현"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id].copy()
        
        # history_length가 지정된 경우 히스토리 제한
        if history_length is not None and "history" in task:
            task["history"] = task["history"][-history_length:] if task["history"] else []
        
        return task
    
    async def cancel_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """A2A 표준 tasks/cancel 메서드 구현"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        if task["status"]["state"] in [TaskState.SUBMITTED, TaskState.WORKING]:
            task["status"]["state"] = TaskState.CANCELED
            task["status"]["timestamp"] = datetime.now().isoformat()
        
        return task
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 정보를 조회합니다."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return {
            "id": session["id"],
            "created_at": session["created_at"],
            "task_count": len(session["tasks"]),
            "tasks": session["tasks"]
        }
    
    def _extract_text_from_message(self, message: Dict[str, Any]) -> str:
        """메시지에서 텍스트를 추출합니다."""
        if not message:
            return ""
            
        if "parts" in message:
            text_parts = []
            for part in message["parts"]:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
        elif "text" in message:
            return message["text"]
        else:
            return str(message)
    
    def get_agent_card(self) -> Dict[str, Any]:
        """A2A 표준 에이전트 카드 반환"""
        return {
            "name": "Patient Data Manager Agent",
            "description": "병원 내 환자 검색 및 진료문서 작성을 위한 의료 AI 에이전트. MCP 프로토콜을 통한 외부 의료 시스템 연동 지원.",
            "url": "http://localhost:10001/",
            "version": "1.0.0",
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
                "mcp_support": True  # MCP 지원 추가
            },
            "skills": [
                {
                    "id": "patient_search",
                    "name": "환자 검색",
                    "description": "환자 이름, ID, 진료과 등을 기준으로 환자 정보를 검색합니다.",
                    "tags": ["patient", "search", "medical"],
                    "examples": ["김철수 환자를 찾아주세요.", "내과 환자 목록을 보여주세요."]
                },
                {
                    "id": "medical_document_generation",
                    "name": "진료문서 작성",
                    "description": "SOAP 노트, 진료 기록 등 의료 문서를 자동 생성합니다.",
                    "tags": ["document", "soap", "medical record"],
                    "examples": ["SOAP 노트를 작성해주세요.", "진료 기록을 정리해주세요."]
                },
                {
                    "id": "drug_interaction_check",
                    "name": "약물 상호작용 검사",
                    "description": "처방된 약물들 간의 상호작용을 검사하고 위험도를 평가합니다.",
                    "tags": ["drug", "interaction", "safety"],
                    "examples": ["이 약물들의 상호작용을 확인해주세요.", "약물 안전성을 검토해주세요."]
                },
                {
                    "id": "mcp_integration",
                    "name": "MCP 외부 시스템 연동",
                    "description": "Model Context Protocol을 통해 외부 의료 데이터베이스 및 시스템과 연동합니다.",
                    "tags": ["mcp", "integration", "external", "database"],
                    "examples": ["병원 데이터베이스에서 검사 결과를 조회해주세요.", "외부 의료 시스템에서 환자 정보를 가져와주세요."]
                }
            ],
            "provider": {
                "name": "Medical AI Platform",
                "organization": "Healthcare Innovation Lab"
            },
            "authentication": {
                "schemes": ["public"]
            },
            # MCP 관련 추가 메타데이터
            "mcp_endpoints": {
                "hospital_db": "http://localhost:8080/mcp/hospital",
                "medical_records": "http://localhost:8081/mcp/records",
                "drug_database": "http://localhost:8082/mcp/drugs",
                "lab_results": "http://localhost:8083/mcp/lab",
                "imaging": "http://localhost:8084/mcp/imaging"
            }
        } 