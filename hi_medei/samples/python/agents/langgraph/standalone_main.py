import logging
import os
import sys
import asyncio
from typing import Any, AsyncIterable

# UTF-8 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import click
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# .env 파일을 먼저 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# LangGraph 에이전트 임포트 (환경변수 로드 후)
from agent import PDFQAAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePDFQAServer:
    def __init__(self, host='localhost', port=10000):
        self.host = host
        self.port = port
        self.agent = PDFQAAgent()
        self.app = Starlette()
        self._setup_routes()
        self._setup_middleware()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        # 기본 상태 확인 (GET만)
        @self.app.route('/', methods=['GET', 'OPTIONS'])
        async def root(request: Request):
            return JSONResponse({
                'status': 'ok',
                'agent': 'PDF QA Agent',
                'version': '1.0.0'
            })

        # A2A 표준 메인 엔드포인트 (POST)
        @self.app.route('/', methods=['POST'])
        async def a2a_main_endpoint(request: Request):
            try:
                data = await request.json()
                
                # JSON-RPC 2.0 형식 확인
                if data.get('jsonrpc') != '2.0':
                    return JSONResponse({'error': 'Invalid JSON-RPC version'}, status_code=400)
                
                method = data.get('method')
                params = data.get('params', {})
                request_id = data.get('id')
                
                if method == 'tasks/send':
                    # 메시지에서 텍스트 추출
                    message = params.get('message', {})
                    parts = message.get('parts', [])
                    query = ""
                    for part in parts:
                        if part.get('type') == 'text':
                            query = part.get('text', '')
                            break
                    
                    if not query:
                        return JSONResponse({
                            'jsonrpc': '2.0',
                            'id': request_id,
                            'error': {'code': -32602, 'message': 'Invalid params: no text found'}
                        })
                    
                    # 에이전트 호출
                    result = self.agent.invoke(query)
                    
                    # A2A 형식으로 응답
                    return JSONResponse({
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'id': params.get('id', 'task-' + str(request_id)),
                            'sessionId': params.get('sessionId'),
                            'status': {
                                'state': 'completed',
                                'timestamp': '2025-05-27T01:45:07.685638'
                            },
                            'artifacts': [{
                                'parts': [{'type': 'text', 'text': result.get('content', '')}],
                                'index': 0
                            }],
                            'history': []
                        }
                    })
                
                elif method == 'tasks/get':
                    # 태스크 상태 조회
                    task_id = params.get('id')
                    return JSONResponse({
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'id': task_id,
                            'status': {'state': 'completed'},
                            'artifacts': [{'parts': [{'type': 'text', 'text': 'Task completed'}], 'index': 0}]
                        }
                    })
                
                else:
                    return JSONResponse({
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'error': {'code': -32601, 'message': f'Method not found: {method}'}
                    })
                    
            except Exception as e:
                logger.error(f"A2A 메인 엔드포인트 처리 중 오류: {e}")
                return JSONResponse({
                    'jsonrpc': '2.0',
                    'id': data.get('id') if 'data' in locals() else None,
                    'error': {'code': -32603, 'message': f'Internal error: {str(e)}'}
                })

        # A2A 에이전트 카드
        @self.app.route('/.well-known/agent.json', methods=['GET'])
        async def get_agent_card(request: Request):
            return JSONResponse({
                "name": "PDF QA Agent",
                "description": "Answers questions about PDF documents using OpenAI GPT-3.5 Turbo and ChromaDB",
                "url": f"http://{self.host}:{self.port}/",
                "version": "1.0.0",
                "defaultInputModes": ["text", "text/plain", "application/pdf"],
                "defaultOutputModes": ["text", "text/plain"],
                "capabilities": {
                    "streaming": False,
                    "pushNotifications": False
                },
                "skills": [{
                    "id": "pdf_qa",
                    "name": "PDF QA Tool",
                    "description": "Answers questions about PDF documents using vector search and LLM",
                    "tags": ["pdf", "qa", "vector search", "medical document"],
                    "examples": ["이 PDF에서 환자 진단 정보를 찾아줘.", "문서에서 주요 소견을 요약해줘."]
                }],
                "provider": {
                    "name": "PDF QA Agent Provider",
                    "organization": "Medical AI"
                },
                "authentication": {
                    "schemes": ["public"]
                }
            })

        # PDF 인제스트 엔드포인트
        @self.app.route('/ingest_pdf', methods=['POST'])
        async def ingest_pdf_endpoint(request: Request):
            try:
                data = await request.json()
                file_path = data.get('file_path')
                
                logger.info(f"PDF 파일 인제스트 요청 수신: {file_path}")
                
                if not file_path:
                    return JSONResponse({'error': 'file_path is required'}, status_code=400)
                
                result = self.agent.ingest(file_path)
                
                logger.info(f"PDF 파일 인제스트 성공: {file_path}")
                return JSONResponse({'result': result, 'file_path': file_path, 'status': 'success'})
            except FileNotFoundError as e:
                logger.error(f"PDF 파일을 찾을 수 없음: {e}")
                return JSONResponse({'error': str(e), 'status': 'file_not_found'}, status_code=404)
            except Exception as e:
                logger.error(f"PDF 인제스트 중 오류 발생: {e}")
                return JSONResponse({'error': str(e), 'status': 'error'}, status_code=500)

        # 간단한 JSON-RPC 2.0 엔드포인트
        @self.app.route('/jsonrpc', methods=['POST'])
        async def jsonrpc_endpoint(request: Request):
            try:
                data = await request.json()
                
                # JSON-RPC 2.0 형식 확인
                if data.get('jsonrpc') != '2.0':
                    return JSONResponse({'error': 'Invalid JSON-RPC version'}, status_code=400)
                
                method = data.get('method')
                params = data.get('params', {})
                request_id = data.get('id')
                
                if method == 'tasks/send':
                    # 메시지에서 텍스트 추출
                    message = params.get('message', {})
                    parts = message.get('parts', [])
                    query = ""
                    for part in parts:
                        if part.get('type') == 'text':
                            query = part.get('text', '')
                            break
                    
                    if not query:
                        return JSONResponse({
                            'jsonrpc': '2.0',
                            'id': request_id,
                            'error': {'code': -32602, 'message': 'Invalid params: no text found'}
                        })
                    
                    # 에이전트 호출
                    result = self.agent.invoke(query)
                    
                    # A2A 형식으로 응답
                    return JSONResponse({
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'id': params.get('id', 'task-' + str(request_id)),
                            'sessionId': params.get('sessionId'),
                            'status': {
                                'state': 'completed',
                                'timestamp': '2025-05-27T01:45:07.685638'
                            },
                            'artifacts': [{
                                'parts': [{'type': 'text', 'text': result.get('content', '')}],
                                'index': 0
                            }],
                            'history': []
                        }
                    })
                
                elif method == 'tasks/get':
                    # 태스크 상태 조회
                    task_id = params.get('id')
                    return JSONResponse({
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'result': {
                            'id': task_id,
                            'status': {'state': 'completed'},
                            'artifacts': [{'parts': [{'type': 'text', 'text': 'Task completed'}], 'index': 0}]
                        }
                    })
                
                else:
                    return JSONResponse({
                        'jsonrpc': '2.0',
                        'id': request_id,
                        'error': {'code': -32601, 'message': f'Method not found: {method}'}
                    })
                    
            except Exception as e:
                logger.error(f"JSON-RPC 처리 중 오류: {e}")
                return JSONResponse({
                    'jsonrpc': '2.0',
                    'id': data.get('id') if 'data' in locals() else None,
                    'error': {'code': -32603, 'message': f'Internal error: {str(e)}'}
                })

        # OpenAI 호환 엔드포인트
        @self.app.route('/v1/chat/completions', methods=['POST'])
        async def chat_completions_endpoint(request: Request):
            try:
                data = await request.json()
                messages = data.get('messages', [])
                query = messages[-1].get('content', '') if messages else ''
                
                result = self.agent.invoke(query)
                
                return JSONResponse({
                    'id': 'pdf-qa-response',
                    'choices': [{
                        'message': {
                            'content': result.get('content', ''),
                            'role': 'assistant'
                        }
                    }]
                })
            except Exception as e:
                logger.error(f"Chat completions error: {e}")
                return JSONResponse({'error': str(e)}, status_code=500)

    def start(self):
        logger.info(f'Starting PDF QA Agent server on {self.host}:{self.port}')
        uvicorn.run(self.app, host=self.host, port=self.port)

@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the PDF QA Agent server."""
    try:
        # OpenAI API 키 확인
        if not os.getenv('OPENAI_API_KEY'):
            logger.warning('OPENAI_API_KEY 환경변수가 설정되지 않았습니다.')
        
        server = SimplePDFQAServer(host=host, port=port)
        server.start()
        
    except Exception as e:
        logger.error(f'서버 시작 중 오류 발생: {e}')
        exit(1)

if __name__ == '__main__':
    main() 