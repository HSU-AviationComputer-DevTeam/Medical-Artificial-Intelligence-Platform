import logging
import os
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import click

from hi_medei.samples.python.agents.langgraph.agent import PDFQAAgent
from hi_medei.samples.python.agents.langgraph.task_manager import AgentTaskManager
from hi_medei.samples.python.common.server import A2AServer
from hi_medei.samples.python.common.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MissingAPIKeyError,
)
from hi_medei.samples.python.common.utils.push_notification_auth import PushNotificationSenderAuth
from dotenv import load_dotenv

from starlette.requests import Request
from starlette.responses import JSONResponse


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the PDF QA Agent server."""
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise MissingAPIKeyError(
                'OPENAI_API_KEY environment variable not set.'
            )

        capabilities = AgentCapabilities(streaming=False, pushNotifications=True)
        skill = AgentSkill(
            id='pdf_qa',
            name='PDF QA Tool',
            description='Answers questions about PDF documents using vector search and LLM',
            tags=['pdf', 'qa', 'vector search', 'medical document'],
            examples=['이 PDF에서 환자 진단 정보를 찾아줘.', '문서에서 주요 소견을 요약해줘.'],
        )
        agent_card = AgentCard(
            name='PDF QA Agent',
            description='Answers questions about PDF documents using OpenAI GPT-3.5 Turbo and ChromaDB',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=PDFQAAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=PDFQAAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=PDFQAAgent(),
                notification_sender_auth=notification_sender_auth,
            ),
            host=host,
            port=port,
        )

        server.app.add_route(
            '/.well-known/jwks.json',
            notification_sender_auth.handle_jwks_endpoint,
            methods=['GET'],
        )

        # CORS 미들웨어 추가
        from starlette.middleware.cors import CORSMiddleware
        server.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # PDF 인제스트용 엔드포인트 추가 (add_route 방식)
        async def ingest_pdf_endpoint(request: Request):
            try:
                data = await request.json()
                file_path = data.get('file_path')
                
                logger.info(f"PDF 파일 인제스트 요청 수신: {file_path}")
                
                if not file_path:
                    return JSONResponse({'error': 'file_path is required'}, status_code=400)
                
                # 파일 존재 여부 확인 로직은 agent.ingest() 내부에서 처리됨
                agent = server.task_manager.agent
                result = agent.ingest(file_path)
                
                logger.info(f"PDF 파일 인제스트 성공: {file_path}")
                return JSONResponse({'result': result, 'file_path': file_path, 'status': 'success'})
            except FileNotFoundError as e:
                logger.error(f"PDF 파일을 찾을 수 없음: {e}")
                return JSONResponse({'error': str(e), 'status': 'file_not_found'}, status_code=404)
            except Exception as e:
                logger.error(f"PDF 인제스트 중 오류 발생: {e}")
                return JSONResponse({'error': str(e), 'status': 'error'}, status_code=500)

        server.app.add_route('/ingest_pdf', ingest_pdf_endpoint, methods=['POST'])
        
        # 상태 확인용 헬스체크 엔드포인트 추가
        async def root(request: Request):
            return JSONResponse({
                'status': 'ok',
                'agent': 'PDF QA Agent',
                'version': '1.0.0'
            })
        server.app.add_route('/', root, methods=['GET', 'POST', 'OPTIONS'])
        
        # A2A 시스템이 에이전트 상태를 확인하기 위한 엔드포인트
        async def status(request: Request):
            return JSONResponse({
                'status': 'ok',
                'agent': 'PDF QA Agent',
                'version': '1.0.0'
            })
        server.app.add_route('/status', status, methods=['GET', 'POST', 'OPTIONS'])
            
        # A2A 시스템의 에이전트 카드 검색을 위한 엔드포인트
        async def get_agent_card(request: Request):
            try:
                return JSONResponse(agent_card.model_dump())
            except Exception as e:
                # Pydantic 버전 차이 대응
                try:
                    return JSONResponse(agent_card.dict())
                except:
                    return JSONResponse({
                        'name': 'PDF QA Agent',
                        'description': 'Answers questions about PDF documents using OpenAI GPT-3.5 Turbo and ChromaDB',
                        'version': '1.0.0'
                    })
        server.app.add_route('/agent-card', get_agent_card, methods=['GET', 'POST', 'OPTIONS'])
            
        # A2A 시스템이 에이전트의 역량을 확인하기 위한 엔드포인트
        async def get_capabilities(request: Request):
            try:
                return JSONResponse(capabilities.model_dump())
            except Exception as e:
                # Pydantic 버전 차이 대응
                try:
                    return JSONResponse(capabilities.dict())
                except:
                    return JSONResponse({
                        'streaming': False,
                        'pushNotifications': True
                    })
        server.app.add_route('/capabilities', get_capabilities, methods=['GET', 'POST', 'OPTIONS'])

        # A2A 시스템이 사용하는 중요한 /v1 엔드포인트 추가
        async def v1_endpoint(request: Request):
            return JSONResponse({
                'status': 'ok',
                'agent': 'PDF QA Agent'
            })
        server.app.add_route('/v1', v1_endpoint, methods=['GET', 'POST', 'OPTIONS'])
        
        # A2A 시스템이 사용하는 /chat/completions 엔드포인트 추가
        async def chat_completions_endpoint(request: Request):
            try:
                data = await request.json()
                query = data.get('messages', [{}])[-1].get('content', '')
                agent = server.task_manager.agent
                result = agent.invoke(query)
                
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
                
        server.app.add_route('/v1/chat/completions', chat_completions_endpoint, methods=['POST', 'OPTIONS'])

        logger.info(f'Starting server on {host}:{port}')
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        exit(1)


if __name__ == '__main__':
    main()
