"""Main entry point for Patient Data Manager Agent A2A server."""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, AsyncIterable, Dict, List, Optional

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 상대 임포트 사용
from agent import PatientDataManagerAgent
from task_manager import TaskManager

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# A2A 표준 JSON-RPC 모델들만 사용


class JSONRPCRequest(BaseModel):
    """JSON-RPC 요청 모델"""
    jsonrpc: str = "2.0"
    id: int
    method: str
    params: Dict[str, Any]


class JSONRPCResponse(BaseModel):
    """JSON-RPC 응답 모델"""
    jsonrpc: str = "2.0"
    id: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


# FastAPI 앱 초기화
app = FastAPI(
    title="Patient Data Manager Agent",
    description="병원 내 환자 검색 및 진료문서 작성을 위한 의료 AI 에이전트",
    version="1.0.0"
)

# 전역 변수
task_manager: Optional[TaskManager] = None


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global task_manager
    
    # API 키들 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "AIzaSyCugPmVpRa8d73W8349ftjqMsJWaIj6NHM")  # 제공받은 키를 기본값으로
    
    if not openai_api_key:
        logger.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        raise ValueError("OPENAI_API_KEY is required")
    
    print(f"🔑 API 키 상태:")
    print(f"  - OpenAI: {'✓' if openai_api_key else '✗'}")
    print(f"  - Gemini: {'✓' if gemini_api_key else '✗'}")
    
    # 데이터 경로 설정 (VectorStore2/medical_data 사용)
    # 현재 스크립트 위치에서 상대 경로로 데이터 디렉토리 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # hi_medei/samples/python/agents/medical_agent -> 프로젝트 루트로 5단계 상위
    project_root = os.path.join(current_dir, "../../../../..")
    data_path = os.path.join(project_root, "VectorStore2", "medical_data")
    data_path = os.path.abspath(data_path)
    
    print(f"🔍 현재 디렉토리: {current_dir}")
    print(f"📁 프로젝트 루트: {os.path.abspath(project_root)}")
    print(f"📊 데이터 경로: {data_path}")
    print(f"✅ 경로 존재 여부: {os.path.exists(data_path)}")
    
    if os.path.exists(data_path):
        json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
        print(f"📋 발견된 JSON 파일들: {json_files}")
    else:
        print(f"❌ 데이터 경로가 존재하지 않습니다: {data_path}")
    
    try:
        # 에이전트 초기화
        agent = PatientDataManagerAgent(
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            data_path=data_path
        )
        
        # 태스크 매니저 초기화
        task_manager = TaskManager(agent)
        
        logger.info("Patient Data Manager Agent 서버가 시작되었습니다.")
        
    except Exception as e:
        logger.error(f"서버 초기화 실패: {e}")
        raise


@app.post("/", response_model=JSONRPCResponse)
async def handle_jsonrpc(request: JSONRPCRequest):
    """JSON-RPC 요청 처리"""
    global task_manager
    
    if not task_manager:
        raise HTTPException(status_code=500, detail="Task manager not initialized")
    
    try:
        method = request.method
        params = request.params
        
        if method == "tasks/send":
            # A2A 표준 태스크 전송
            import uuid
            task_id = str(uuid.uuid4())
            result = await task_manager.send_task(
                task_id=task_id,
                session_id=params.get("sessionId"),
                message=params.get("task")
            )
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "tasks/sendSubscribe":
            # A2A 표준 스트리밍 태스크 전송
            # 이 메서드는 SSE 스트림으로 처리되어야 하므로 여기서는 에러 반환
            return JSONRPCResponse(
                id=request.id,
                error={"code": -32601, "message": "Use /stream endpoint for tasks/sendSubscribe"}
            )
            
        elif method == "tasks/get":
            # A2A 표준 태스크 조회
            task_id = params["id"]
            history_length = params.get("historyLength")
            result = await task_manager.get_task(task_id, history_length)
            
            if result is None:
                return JSONRPCResponse(
                    id=request.id,
                    error={"code": -32602, "message": "Task not found"}
                )
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "tasks/cancel":
            # A2A 표준 태스크 취소
            task_id = params["id"]
            result = await task_manager.cancel_task(task_id)
            
            if result is None:
                return JSONRPCResponse(
                    id=request.id,
                    error={"code": -32602, "message": "Task not found"}
                )
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "agent/info":
            # 에이전트 정보 조회
            agent_card = task_manager.get_agent_card()
            
            return JSONRPCResponse(
                id=request.id,
                result=agent_card
            )
            
        elif method == "invoke":
            # 직접 에이전트 호출
            query = params.get("query", "")
            session_id = params.get("session_id", "default")
            
            result = task_manager.agent.invoke(query, session_id)
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "mcp/connect":
            # MCP 서버 연결 테스트
            endpoint = params.get("endpoint", "pubmed")
            
            from mcp_client import get_mcp_client
            client = await get_mcp_client()
            result = await client.test_connection(endpoint)
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "mcp/list_endpoints":
            # 사용 가능한 MCP 엔드포인트 목록 반환
            from mcp_client import get_mcp_client
            client = await get_mcp_client()
            result = await client.get_all_endpoints()
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "mcp/search_pubmed":
            # PubMed 논문 검색
            query = params.get("query", "")
            max_results = params.get("max_results", 5)
            
            from mcp_client import get_mcp_client
            client = await get_mcp_client()
            result = await client.search_pubmed(query, max_results)
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "mcp/search_medical_condition":
            # 의학적 상태 연구 검색
            condition = params.get("condition", "")
            max_results = params.get("max_results", 5)
            
            from mcp_client import get_mcp_client
            client = await get_mcp_client()
            result = await client.search_medical_condition(condition, max_results)
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "mcp/save_memory":
            # 메모리에 정보 저장
            session_id = params.get("session_id", "default")
            content = params.get("content", "")
            entry_type = params.get("entry_type", "conversation")
            patient_id = params.get("patient_id")
            
            from mcp_client import get_mcp_client
            client = await get_mcp_client()
            result = await client.save_memory(session_id, content, entry_type, patient_id)
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        elif method == "mcp/get_memory":
            # 메모리 조회
            session_id = params.get("session_id")
            patient_id = params.get("patient_id")
            limit = params.get("limit", 50)
            
            from mcp_client import get_mcp_client
            client = await get_mcp_client()
            
            if session_id:
                result = await client.get_session_memory(session_id, limit)
            elif patient_id:
                result = await client.get_patient_memory(patient_id, limit)
            else:
                result = {"success": False, "error": "session_id 또는 patient_id가 필요합니다."}
            
            return JSONRPCResponse(
                id=request.id,
                result=result
            )
            
        else:
            return JSONRPCResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {method}"}
            )
            
    except Exception as e:
        logger.error(f"JSON-RPC 처리 중 오류: {e}")
        return JSONRPCResponse(
            id=request.id,
            error={"code": -32603, "message": f"Internal error: {str(e)}"}
        )


@app.post("/stream")
async def handle_stream(request: JSONRPCRequest):
    """A2A 표준 스트리밍 처리 (tasks/sendSubscribe)"""
    global task_manager
    
    if not task_manager:
        raise HTTPException(status_code=500, detail="Task manager not initialized")
    
    # JSON-RPC 요청 검증
    if request.method != "tasks/sendSubscribe":
        raise HTTPException(status_code=400, detail="Only tasks/sendSubscribe method supported")
    
    params = request.params
    
    async def generate_stream():
        try:
            async for chunk in task_manager.send_task_subscribe(
                task_id=params["id"],
                session_id=params.get("sessionId"),
                message=params["message"]
            ):
                # A2A 표준 JSON-RPC 응답 형식으로 래핑
                response = {
                    "jsonrpc": "2.0",
                    "id": request.id,
                    "result": chunk
                }
                yield f"data: {json.dumps(response)}\n\n"
                
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request.id,
                "error": {
                    "code": -32603,
                    "message": f"스트리밍 오류: {str(e)}"
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "agent": "Patient Data Manager Agent",
        "version": "1.0.0"
    }


@app.get("/agent-card")
async def get_agent_card():
    """에이전트 카드 정보 반환"""
    global task_manager
    
    if not task_manager:
        raise HTTPException(status_code=500, detail="Task manager not initialized")
    
    return task_manager.get_agent_card()


@app.get("/.well-known/agent.json")
async def get_agent_json():
    """A2A 표준 에이전트 카드 엔드포인트"""
    global task_manager
    
    if not task_manager:
        raise HTTPException(status_code=500, detail="Task manager not initialized")
    
    return task_manager.get_agent_card()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Patient Data Manager Agent Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=10001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Patient Data Manager Agent 서버를 시작합니다...")
    logger.info(f"주소: http://{args.host}:{args.port}")
    logger.info(f"에이전트 카드: http://{args.host}:{args.port}/agent-card")
    logger.info(f"헬스 체크: http://{args.host}:{args.port}/health")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()