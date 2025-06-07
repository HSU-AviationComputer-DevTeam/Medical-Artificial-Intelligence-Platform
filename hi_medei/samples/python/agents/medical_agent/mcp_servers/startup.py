#!/usr/bin/env python3
"""
간단한 서버 매니저
FastAPI 기반 서버들을 시작하고 관리하는 도구입니다.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional

from .memory_server import MemoryServer
from .pubmed_server import PubMedServer

logger = logging.getLogger(__name__)

class MCPServerManager:
    """FastAPI 서버들을 관리하는 매니저"""
    
    def __init__(self):
        self.servers: Dict[str, any] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
        # 기본 서버 설정
        self.server_configs = {
            "pubmed": {
                "class": PubMedServer,
                "port": 8080,
                "description": "PubMed 의학 논문 검색 서버"
            },
            "memory": {
                "class": MemoryServer,
                "port": 8081,
                "description": "환자 대화 기록 메모리 서버"
            }
        }
    
    async def start_all_servers(self, server_names: Optional[List[str]] = None):
        """모든 서버들을 시작합니다."""
        server_names = server_names or list(self.server_configs.keys())
        
        logger.info("🚀 FastAPI 서버들을 시작합니다...")
        
        for server_name in server_names:
            if server_name not in self.server_configs:
                logger.warning(f"❌ 알 수 없는 서버: {server_name}")
                continue
            
            try:
                config = self.server_configs[server_name]
                server_class = config["class"]
                port = config["port"]
                description = config["description"]
                
                logger.info(f"📡 {description} (포트 {port}) 시작 중...")
                
                # 서버 인스턴스 생성
                if server_name == "memory":
                    server = server_class(port=port, db_path=f"medical_memory_{port}.db")
                else:
                    server = server_class(port=port)
                
                self.servers[server_name] = server
                
                # 서버 태스크 시작
                task = asyncio.create_task(
                    server.start_server(),
                    name=f"server_{server_name}"
                )
                self.tasks[server_name] = task
                
                # 서버 시작 확인을 위한 짧은 지연
                await asyncio.sleep(1.0)
                
                logger.info(f"✅ {description} 시작 완료")
                
            except Exception as e:
                logger.error(f"❌ {server_name} 서버 시작 실패: {e}")
        
        self.running = True
        logger.info("🎉 모든 서버들이 시작되었습니다!")
        self._print_server_status()
    
    async def stop_all_servers(self):
        """모든 서버들을 중지합니다."""
        logger.info("🛑 서버들을 중지합니다...")
        
        self.running = False
        
        for server_name, task in self.tasks.items():
            try:
                task.cancel()
                await asyncio.wait_for(task, timeout=5.0)
                logger.info(f"✅ {server_name} 서버 중지 완료")
            except asyncio.CancelledError:
                logger.info(f"✅ {server_name} 서버 중지됨")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️  {server_name} 서버 중지 타임아웃")
            except Exception as e:
                logger.error(f"❌ {server_name} 서버 중지 오류: {e}")
        
        self.tasks.clear()
        self.servers.clear()
        logger.info("🏁 모든 서버들이 중지되었습니다.")
    
    def _print_server_status(self):
        """서버 상태를 출력합니다."""
        print("\n" + "="*60)
        print("🏥 hi_medei FastAPI 서버 상태")
        print("="*60)
        
        for server_name, config in self.server_configs.items():
            if server_name in self.servers:
                status = "🟢 실행 중"
            else:
                status = "🔴 중지됨"
            
            print(f"{status} {config['description']}")
            print(f"    📡 포트: {config['port']}")
            print(f"    🔗 URL: http://localhost:{config['port']}")
            print()
        
        print("="*60)
        print("💡 사용법:")
        print("   - PubMed 검색: POST http://localhost:8080/tools/search_pubmed")
        print("   - 메모리 저장: POST http://localhost:8081/tools/save_memory")
        print("   - 서버 중지: Ctrl+C")
        print("="*60 + "\n")
    
    async def wait_for_shutdown(self):
        """종료 신호를 기다립니다."""
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            logger.info("🛑 종료 신호를 받았습니다...")
            shutdown_event.set()
        
        # 윈도우와 유닉스 계열 모두 지원
        if sys.platform == "win32":
            signal.signal(signal.SIGINT, lambda s, f: signal_handler())
            signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
        else:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, signal_handler)
        
        await shutdown_event.wait()
    
    async def run_forever(self, server_names: Optional[List[str]] = None):
        """서버들을 시작하고 무한히 실행합니다."""
        try:
            await self.start_all_servers(server_names)
            await self.wait_for_shutdown()
        except KeyboardInterrupt:
            logger.info("🛑 사용자에 의해 중단되었습니다.")
        except Exception as e:
            logger.error(f"❌ 실행 중 오류 발생: {e}")
        finally:
            await self.stop_all_servers()

async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="hi_medei FastAPI 서버 매니저")
    parser.add_argument(
        "--servers", 
        nargs="+", 
        choices=["pubmed", "memory"],
        help="시작할 서버들 (기본: 모든 서버)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 서버 매니저 실행
    manager = MCPServerManager()
    await manager.run_forever(args.servers)

if __name__ == "__main__":
    asyncio.run(main()) 