#!/usr/bin/env python3
"""
A2A-MCP 통합 테스트 스크립트
hi_medei 의료 AI 에이전트의 A2A와 MCP 연동을 테스트합니다.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict

import aiohttp

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class A2AMCPIntegrationTester:
    """A2A-MCP 통합 테스트 클래스"""
    
    def __init__(self, a2a_url: str = "http://localhost:10001"):
        self.a2a_url = a2a_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def call_a2a_jsonrpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """A2A 서버에 JSON-RPC 요청을 보냅니다."""
        request_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            async with self.session.post(self.a2a_url, json=request_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "error": {
                            "code": response.status,
                            "message": error_text
                        }
                    }
        except Exception as e:
            return {
                "error": {
                    "code": -1,
                    "message": str(e)
                }
            }
    
    async def test_a2a_health(self) -> bool:
        """A2A 서버 헬스체크"""
        print("🔍 A2A 서버 상태 확인 중...")
        
        try:
            async with self.session.get(f"{self.a2a_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ A2A 서버 정상: {data}")
                    return True
                else:
                    print(f"❌ A2A 서버 오류: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ A2A 서버 연결 실패: {e}")
            return False
    
    async def test_mcp_endpoints(self) -> bool:
        """MCP 엔드포인트 목록 테스트"""
        print("\n🔍 MCP 엔드포인트 목록 조회 중...")
        
        result = await self.call_a2a_jsonrpc("mcp/list_endpoints", {})
        
        if "error" in result:
            print(f"❌ MCP 엔드포인트 조회 실패: {result['error']}")
            return False
        
        endpoints = result.get("result", {})
        print(f"✅ MCP 엔드포인트 목록:")
        print(json.dumps(endpoints, indent=2, ensure_ascii=False))
        
        return endpoints.get("success", False)
    
    async def test_mcp_connection(self, endpoint: str = "pubmed") -> bool:
        """MCP 서버 연결 테스트"""
        print(f"\n🔍 MCP 서버 ({endpoint}) 연결 테스트 중...")
        
        result = await self.call_a2a_jsonrpc("mcp/connect", {"endpoint": endpoint})
        
        if "error" in result:
            print(f"❌ MCP 연결 테스트 실패: {result['error']}")
            return False
        
        connection_result = result.get("result", {})
        success = connection_result.get("success", False)
        
        if success:
            print(f"✅ MCP 서버 연결 성공: {connection_result['message']}")
        else:
            print(f"❌ MCP 서버 연결 실패: {connection_result.get('error', 'Unknown error')}")
        
        return success
    
    async def test_pubmed_search(self, query: str = "diabetes mellitus") -> bool:
        """PubMed 검색 테스트"""
        print(f"\n🔍 PubMed 검색 테스트 중: '{query}'")
        
        result = await self.call_a2a_jsonrpc("mcp/search_pubmed", {
            "query": query,
            "max_results": 3
        })
        
        if "error" in result:
            print(f"❌ PubMed 검색 실패: {result['error']}")
            return False
        
        search_result = result.get("result", {})
        success = search_result.get("success", False)
        
        if success:
            articles = search_result.get("result", {}).get("articles", [])
            print(f"✅ PubMed 검색 성공: {len(articles)}개 논문 발견")
            
            for i, article in enumerate(articles[:2], 1):
                print(f"  📄 논문 {i}: {article.get('title', 'No title')[:100]}...")
        else:
            print(f"❌ PubMed 검색 실패: {search_result.get('error', 'Unknown error')}")
        
        return success
    
    async def test_medical_condition_search(self, condition: str = "hypertension") -> bool:
        """의학적 상태 검색 테스트"""
        print(f"\n🔍 의학적 상태 검색 테스트 중: '{condition}'")
        
        result = await self.call_a2a_jsonrpc("mcp/search_medical_condition", {
            "condition": condition,
            "max_results": 2
        })
        
        if "error" in result:
            print(f"❌ 의학적 상태 검색 실패: {result['error']}")
            return False
        
        search_result = result.get("result", {})
        success = search_result.get("success", False)
        
        if success:
            studies = search_result.get("result", {}).get("recent_studies", [])
            print(f"✅ 의학적 상태 검색 성공: {len(studies)}개 최신 연구 발견")
            
            for i, study in enumerate(studies[:1], 1):
                print(f"  📊 연구 {i}: {study.get('title', 'No title')[:100]}...")
        else:
            print(f"❌ 의학적 상태 검색 실패: {search_result.get('error', 'Unknown error')}")
        
        return success
    
    async def test_memory_operations(self) -> bool:
        """메모리 저장/조회 테스트"""
        print(f"\n🔍 메모리 저장/조회 테스트 중...")
        
        # 메모리 저장 테스트
        save_result = await self.call_a2a_jsonrpc("mcp/save_memory", {
            "session_id": "test_session_123",
            "content": "환자가 당뇨병 증상에 대해 문의했습니다.",
            "entry_type": "conversation",
            "patient_id": "P001"
        })
        
        if "error" in save_result:
            print(f"❌ 메모리 저장 실패: {save_result['error']}")
            return False
        
        save_success = save_result.get("result", {}).get("success", False)
        if not save_success:
            print(f"❌ 메모리 저장 실패: {save_result.get('result', {}).get('error', 'Unknown error')}")
            return False
        
        print("✅ 메모리 저장 성공")
        
        # 메모리 조회 테스트
        get_result = await self.call_a2a_jsonrpc("mcp/get_memory", {
            "session_id": "test_session_123",
            "limit": 10
        })
        
        if "error" in get_result:
            print(f"❌ 메모리 조회 실패: {get_result['error']}")
            return False
        
        get_success = get_result.get("result", {}).get("success", False)
        if get_success:
            entries = get_result.get("result", {}).get("entries", [])
            print(f"✅ 메모리 조회 성공: {len(entries)}개 엔트리 발견")
        else:
            print(f"❌ 메모리 조회 실패: {get_result.get('result', {}).get('error', 'Unknown error')}")
        
        return get_success
    
    async def test_medical_agent_integration(self) -> bool:
        """의료 에이전트와 MCP 통합 테스트"""
        print(f"\n🔍 의료 에이전트-MCP 통합 테스트 중...")
        
        # 환자 검색 + PubMed 연구 조합 시나리오
        query = "고혈압 환자의 최신 치료 방법을 알려주세요"
        
        result = await self.call_a2a_jsonrpc("invoke", {
            "query": query,
            "session_id": "integration_test"
        })
        
        if "error" in result:
            print(f"❌ 의료 에이전트 호출 실패: {result['error']}")
            return False
        
        agent_result = result.get("result")
        if agent_result:
            print(f"✅ 의료 에이전트 응답 성공")
            print(f"  📝 응답 길이: {len(str(agent_result))} 문자")
            
            # MCP 관련 키워드가 포함되어 있는지 확인
            response_text = str(agent_result).lower()
            mcp_keywords = ["pubmed", "논문", "연구", "최신"]
            mcp_found = any(keyword in response_text for keyword in mcp_keywords)
            
            if mcp_found:
                print("  🔗 MCP 연동 키워드 발견됨")
            else:
                print("  ⚠️  MCP 연동 키워드 미발견")
            
            return True
        else:
            print("❌ 의료 에이전트 빈 응답")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """모든 테스트를 실행합니다."""
        print("🧪 A2A-MCP 통합 테스트를 시작합니다!")
        print("="*70)
        
        test_results = {}
        
        # 1. A2A 서버 헬스체크
        test_results["a2a_health"] = await self.test_a2a_health()
        
        # 2. MCP 엔드포인트 목록
        test_results["mcp_endpoints"] = await self.test_mcp_endpoints()
        
        # 3. MCP 연결 테스트
        test_results["mcp_connection"] = await self.test_mcp_connection("pubmed")
        
        # 4. PubMed 검색 테스트
        test_results["pubmed_search"] = await self.test_pubmed_search()
        
        # 5. 의학적 상태 검색 테스트
        test_results["medical_condition_search"] = await self.test_medical_condition_search()
        
        # 6. 메모리 저장/조회 테스트
        test_results["memory_operations"] = await self.test_memory_operations()
        
        # 7. 의료 에이전트 통합 테스트
        test_results["agent_integration"] = await self.test_medical_agent_integration()
        
        return test_results
    
    def print_test_summary(self, results: Dict[str, bool]):
        """테스트 결과 요약을 출력합니다."""
        print("\n" + "="*70)
        print("🧪 테스트 결과 요약")
        print("="*70)
        
        passed = 0
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ 통과" if success else "❌ 실패"
            test_display = test_name.replace("_", " ").title()
            print(f"{status} {test_display}")
            if success:
                passed += 1
        
        print("="*70)
        print(f"📊 전체 결과: {passed}/{total} 테스트 통과 ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 모든 테스트가 성공적으로 통과했습니다!")
        else:
            print("⚠️  일부 테스트가 실패했습니다. MCP 서버 상태를 확인해주세요.")
        
        print("="*70)

async def main():
    """메인 함수"""
    print("🏥 hi_medei A2A-MCP 통합 테스트를 시작합니다...")
    
    async with A2AMCPIntegrationTester() as tester:
        results = await tester.run_all_tests()
        tester.print_test_summary(results)

if __name__ == "__main__":
    asyncio.run(main()) 