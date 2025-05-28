#!/usr/bin/env python3
"""에이전트 invoke 메서드 디버깅 스크립트"""

import os
import sys
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from agent import PatientDataManagerAgent

def test_agent_invoke():
    """에이전트 invoke 메서드를 직접 테스트합니다."""
    
    # OpenAI API 키 설정
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    # 데이터 경로 설정
    data_path = "/Users/sindong-u/coding/project/hi_medei/data"
    
    try:
        print("=== 에이전트 초기화 ===")
        agent = PatientDataManagerAgent(
            openai_api_key=openai_api_key,
            data_path=data_path
        )
        print("에이전트 초기화 완료")
        
        print("\n=== 홍길1 환자 검색 테스트 ===")
        query = "홍길1 환자의 상세 정보를 알려주세요"
        result = agent.invoke(query, "test-session")
        
        print(f"결과: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_invoke() 