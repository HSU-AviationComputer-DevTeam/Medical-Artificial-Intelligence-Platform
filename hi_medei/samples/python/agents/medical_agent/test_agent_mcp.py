#!/usr/bin/env python3
"""
Agent의 MCP 사용 테스트 스크립트
"""

import json

import requests


def test_agent_mcp():
    """Agent가 MCP를 사용하는지 테스트"""
    
    print("🧪 Agent MCP 사용 테스트 시작...")
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "당뇨병 최신 연구 (MCP 예상)",
            "query": "당뇨병 최신 연구를 알려줘",
            "expected_mcp": True
        },
        {
            "name": "고혈압 논문 검색 (MCP 예상)", 
            "query": "고혈압 논문을 찾아줘",
            "expected_mcp": True
        },
        {
            "name": "당뇨병 환자 검색 (하이브리드 예상)",
            "query": "당뇨병 환자를 찾아줘", 
            "expected_mcp": False
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   질문: {test_case['query']}")
        
        try:
            response = requests.post(
                'http://localhost:10001/',
                json={
                    'jsonrpc': '2.0',
                    'id': i,
                    'method': 'invoke',
                    'params': {
                        'query': test_case['query'],
                        'session_id': f'test_{i}'
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('result', {})
                content = result.get('content', '')
                
                # MCP 사용 여부 판단
                mcp_used = any(keyword in content for keyword in [
                    '📚 **MCP PubMed 검색', 
                    'MCP 검색 결과',
                    'MCP PubMed',
                    'PMID:'
                ])
                
                hybrid_used = any(keyword in content for keyword in [
                    '🧠 **벡터 검색 결과',
                    '📊 **구조화된 데이터 검색'
                ])
                
                print(f"   📊 결과: {'MCP 사용됨' if mcp_used else 'MCP 사용 안 됨'}")
                print(f"   📋 응답: {content[:150]}...")
                
                if test_case['expected_mcp'] and mcp_used:
                    print("   ✅ 예상대로 MCP 사용됨")
                elif not test_case['expected_mcp'] and hybrid_used:
                    print("   ✅ 예상대로 하이브리드 검색 사용됨")
                else:
                    print("   ❌ 예상과 다른 결과")
                    
            else:
                print(f"   ❌ HTTP 오류: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 요청 실패: {e}")
    
    print("\n🎯 테스트 완료!")

if __name__ == "__main__":
    test_agent_mcp() 