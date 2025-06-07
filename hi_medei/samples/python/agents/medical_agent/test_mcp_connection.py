#!/usr/bin/env python3
"""
MCP 연결 테스트 스크립트
A2A 서버와 MCP 엔드포인트 간의 연결을 테스트합니다.
"""

import asyncio
import json
import logging
from typing import Any, Dict

import aiohttp

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class A2AMCPTester:
    """A2A-MCP 연결 테스트 클래스"""
    
    def __init__(self, a2a_server_url: str = "http://localhost:10001"):
        self.a2a_server_url = a2a_server_url
    
    async def test_a2a_server_health(self) -> Dict[str, Any]:
        """A2A 서버 헬스체크"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.a2a_server_url}/health") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("✅ A2A 서버 연결 성공")
                        return {"status": "success", "data": result}
                    else:
                        logger.error(f"❌ A2A 서버 연결 실패: HTTP {response.status}")
                        return {"status": "error", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"❌ A2A 서버 연결 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_mcp_endpoints_list(self) -> Dict[str, Any]:
        """MCP 엔드포인트 목록 조회"""
        try:
            request_data = {
                "jsonrpc": "2.0",
                "method": "mcp/list_endpoints",
                "params": {},
                "id": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.a2a_server_url,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("✅ MCP 엔드포인트 목록 조회 성공")
                        return {"status": "success", "data": result}
                    else:
                        logger.error(f"❌ MCP 엔드포인트 목록 조회 실패: HTTP {response.status}")
                        return {"status": "error", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"❌ MCP 엔드포인트 목록 조회 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_mcp_connection(self, endpoint: str = "hospital_db", query: str = "connection test") -> Dict[str, Any]:
        """특정 MCP 엔드포인트 연결 테스트"""
        try:
            request_data = {
                "jsonrpc": "2.0",
                "method": "mcp/connect",
                "params": {
                    "endpoint": endpoint,
                    "query": query
                },
                "id": 2
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.a2a_server_url,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ MCP 엔드포인트 '{endpoint}' 연결 테스트 성공")
                        return {"status": "success", "data": result}
                    else:
                        logger.error(f"❌ MCP 엔드포인트 '{endpoint}' 연결 테스트 실패: HTTP {response.status}")
                        return {"status": "error", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"❌ MCP 엔드포인트 '{endpoint}' 연결 테스트 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_agent_card(self) -> Dict[str, Any]:
        """A2A 에이전트 카드 조회 (MCP 지원 확인)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.a2a_server_url}/agent-card") as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("✅ 에이전트 카드 조회 성공")
                        
                        # MCP 지원 확인
                        capabilities = result.get("capabilities", {})
                        mcp_support = capabilities.get("mcp_support", False)
                        mcp_endpoints = result.get("mcp_endpoints", {})
                        
                        logger.info(f"📊 MCP 지원: {'✅' if mcp_support else '❌'}")
                        logger.info(f"📊 MCP 엔드포인트 수: {len(mcp_endpoints)}")
                        
                        return {"status": "success", "data": result}
                    else:
                        logger.error(f"❌ 에이전트 카드 조회 실패: HTTP {response.status}")
                        return {"status": "error", "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"❌ 에이전트 카드 조회 실패: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_all_tests(self):
        """모든 테스트 실행"""
        logger.info("🚀 A2A-MCP 연결 테스트 시작")
        logger.info("="*60)
        
        # 1. A2A 서버 헬스체크
        logger.info("1️⃣ A2A 서버 헬스체크")
        health_result = await self.test_a2a_server_health()
        print(f"   결과: {json.dumps(health_result, ensure_ascii=False, indent=2)}")
        
        # 2. 에이전트 카드 조회
        logger.info("\n2️⃣ 에이전트 카드 조회 (MCP 지원 확인)")
        card_result = await self.test_agent_card()
        print(f"   결과: {json.dumps(card_result, ensure_ascii=False, indent=2)}")
        
        # 3. MCP 엔드포인트 목록 조회
        logger.info("\n3️⃣ MCP 엔드포인트 목록 조회")
        endpoints_result = await self.test_mcp_endpoints_list()
        print(f"   결과: {json.dumps(endpoints_result, ensure_ascii=False, indent=2)}")
        
        # 4. 각 MCP 엔드포인트 연결 테스트
        if endpoints_result["status"] == "success":
            endpoints = endpoints_result["data"]["result"]["endpoints"]
            logger.info(f"\n4️⃣ 각 MCP 엔드포인트 연결 테스트 ({len(endpoints)}개)")
            
            for i, endpoint in enumerate(endpoints, 1):
                logger.info(f"   4-{i}. '{endpoint}' 엔드포인트 테스트")
                connection_result = await self.test_mcp_connection(endpoint, f"{endpoint} 연결 테스트")
                print(f"        결과: {json.dumps(connection_result, ensure_ascii=False, indent=2)}")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 A2A-MCP 연결 테스트 완료")

async def main():
    """메인 함수"""
    tester = A2AMCPTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 