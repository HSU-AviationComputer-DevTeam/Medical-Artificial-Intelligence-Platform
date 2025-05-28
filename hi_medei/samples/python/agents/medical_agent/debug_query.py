#!/usr/bin/env python3
"""쿼리 분석 디버깅"""

import re
import json
from medical_tools import PatientSearchTool

def test_query_analysis():
    """쿼리 분석 테스트"""
    print("=== 쿼리 분석 테스트 ===")
    
    data_path = "/Users/sindong-u/coding/project/hi_medei/data"
    patient_search = PatientSearchTool(data_path=data_path)
    
    # 테스트 쿼리들
    queries = [
        "홍길1 환자 정보를 찾아주세요",
        "당뇨병 환자들을 찾아주세요"
    ]
    
    for query in queries:
        print(f"\n--- 쿼리: {query} ---")
        query_lower = query.lower()
        
        # 환자 이름 패턴 검색
        name_patterns = [r'홍길\d+', r'김철\d+', r'박민\d+', r'이영\d+', r'최수\d+']
        
        found_name = False
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                name = match.group()
                print(f"환자 이름 패턴 발견: {name}")
                result = patient_search._run(f"이름: {name}")
                result_data = json.loads(result)
                print(f"검색 결과: {result_data['total_count']}명")
                found_name = True
                break
        
        if not found_name:
            # 질병명 검색
            diseases = ['당뇨병', '고혈압', '담낭염', '위염', '감기', '독감']
            found_disease = False
            for disease in diseases:
                if disease in query:
                    print(f"질병명 발견: {disease}")
                    result = patient_search._run(f"진단: {disease}")
                    result_data = json.loads(result)
                    print(f"검색 결과: {result_data['total_count']}명")
                    found_disease = True
                    break
            
            if not found_disease:
                print("패턴을 찾을 수 없습니다.")

if __name__ == "__main__":
    test_query_analysis() 