#!/usr/bin/env python3
"""간단한 에이전트 테스트"""

import os
from agent import PatientDataManagerAgent

def test_agent():
    """에이전트 테스트"""
    print("=== 에이전트 테스트 ===")
    
    # 환경변수 설정
    openai_api_key = os.getenv("OPENAI_API_KEY", "test-key")
    data_path = "/Users/sindong-u/coding/project/hi_medei/data"
    
    print(f"데이터 경로: {data_path}")
    print(f"경로 존재 여부: {os.path.exists(data_path)}")
    
    try:
        # 에이전트 초기화
        agent = PatientDataManagerAgent(
            openai_api_key=openai_api_key,
            data_path=data_path
        )
        
        print(f"\n도구 수: {len(agent.tools)}")
        for tool in agent.tools:
            print(f"  - {tool.name}")
        
        # 테스트 쿼리
        test_queries = [
            "홍길1 환자 정보를 찾아주세요",
            "당뇨병 환자들을 검색해주세요"
        ]
        
        for query in test_queries:
            print(f"\n=== 테스트: {query} ===")
            result = agent.invoke(query)
            print(f"결과: {result['content']}")
            
    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent() 