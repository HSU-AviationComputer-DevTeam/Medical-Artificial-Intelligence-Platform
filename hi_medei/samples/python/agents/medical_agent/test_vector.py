#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv

from medical_tools import HybridSearchTool, VectorSearchTool

# 환경변수 로드
load_dotenv()

def test_vector_search():
    print("=== 벡터 검색 테스트 ===")
    
    # OpenAI API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    print(f"✅ OpenAI API 키 확인됨: {api_key[:10]}...")
    
    # 벡터 검색 도구 테스트
    print("\n1. VectorSearchTool 테스트")
    vector_tool = VectorSearchTool(openai_api_key=api_key)
    vector_result = vector_tool._run("당뇨병 증상")
    print("벡터 검색 결과:")
    print(vector_result[:500], "...")
    
    # 하이브리드 검색 도구 테스트
    print("\n2. HybridSearchTool 테스트")
    hybrid_tool = HybridSearchTool(
        data_path="../../../../../VectorStore2/medical_data",
        openai_api_key=api_key
    )
    hybrid_result = hybrid_tool._run("당뇨병환자 검색해줘")
    print("하이브리드 검색 결과:")
    print(hybrid_result[:500], "...")

if __name__ == "__main__":
    test_vector_search() 