#!/usr/bin/env python3
"""
올바른 E5 vs Gemini 벡터 스토어 벤치마크 테스트
각 모델은 자신의 벡터 스토어를 사용
"""

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 필요한 라이브러리 임포트
try:
    import faiss
    from google import genai
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    print("✅ 필요한 라이브러리 로드 완료")
except ImportError as e:
    print(f"❌ 라이브러리 임포트 오류: {e}")
    print("pip install faiss-cpu langchain langchain-community google-generativeai 실행 필요")
    sys.exit(1)

class GeminiEmbeddings:
    """Gemini 임베딩 클래스 (첨부된 코드 참고)"""
    
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.last_api_call_time = 0
        self.api_call_interval = 15  # 15초 간격
    
    def _wait_for_rate_limit(self):
        """API 요청 전 대기 시간 관리"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        if time_since_last_call < self.api_call_interval:
            wait_time = self.api_call_interval - time_since_last_call
            print(f"⏳ API 제한으로 {wait_time:.1f}초 대기 중...")
            time.sleep(wait_time)
        
        self.last_api_call_time = time.time()
    
    def embed_documents(self, texts):
        """문서 임베딩 - 3072차원 벡터 생성"""
        if isinstance(texts, str):
            texts = [texts]
        
        self._wait_for_rate_limit()
        
        result = self.client.models.embed_content(
            model="models/gemini-embedding-exp-03-07",  # 3072차원 모델
            contents=texts,
            config={"task_type": "retrieval_document"}
        )
        return [embedding.values for embedding in result.embeddings]
    
    def embed_query(self, text):
        """쿼리 임베딩 - 3072차원 벡터 생성"""
        self._wait_for_rate_limit()
        
        result = self.client.models.embed_content(
            model="models/gemini-embedding-exp-03-07",  # 3072차원 모델
            contents=[text],
            config={"task_type": "retrieval_query"}
        )
        return result.embeddings[0].values
    
    def __call__(self, text):
        """직접 호출 시 쿼리 임베딩 실행"""
        return self.embed_query(text)

class ProperVectorStoreBenchmark:
    """올바른 벡터 스토어 벤치마크"""
    
    def __init__(self, gemini_api_key=None):
        self.results = {}
        self.gemini_api_key = gemini_api_key
        
    def load_e5_vector_store(self, store_path: str):
        """E5 벡터 스토어 로드 (E5 모델 사용)"""
        try:
            print(f"🔄 E5 벡터 스토어 로딩: {store_path}")
            
            # E5 임베딩 모델 초기화
            embedding_model = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # E5용 FAISS 벡터 스토어 로드
            vector_store = FAISS.load_local(
                store_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            
            print(f"✅ E5 벡터 스토어 로드 완료")
            print(f"   - 벡터 개수: {vector_store.index.ntotal}")
            print(f"   - 벡터 차원: {vector_store.index.d}")
            
            return vector_store, embedding_model
            
        except Exception as e:
            print(f"❌ E5 벡터 스토어 로드 실패: {e}")
            return None, None
    
    def load_gemini_vector_store(self, store_path: str):
        """Gemini 벡터 스토어 로드 (Gemini 모델 사용)"""
        try:
            print(f"🔄 Gemini 벡터 스토어 로딩: {store_path}")
            
            if not self.gemini_api_key:
                # 환경변수에서 API 키 읽기
                self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
                if not self.gemini_api_key:
                    print("⚠️ GOOGLE_API_KEY 환경변수가 설정되지 않음")
                    return None, None
            
            # Gemini 임베딩 모델 초기화 (첨부된 코드 방식 사용)
            embedding_model = GeminiEmbeddings(self.gemini_api_key)
            
            # Gemini용 FAISS 벡터 스토어 로드
            vector_store = FAISS.load_local(
                store_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            
            print(f"✅ Gemini 벡터 스토어 로드 완료")
            print(f"   - 벡터 개수: {vector_store.index.ntotal}")
            print(f"   - 벡터 차원: {vector_store.index.d}")
            
            return vector_store, embedding_model
            
        except Exception as e:
            print(f"❌ Gemini 벡터 스토어 로드 실패: {e}")
            return None, None

    def precision_at_k(self, relevant_docs: List[str], retrieved_docs: List[Dict], k: int) -> float:
        """Precision@K 계산 (실제 검색 결과용)"""
        if k == 0 or not retrieved_docs:
            return 0.0
        
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = 0
        
        for doc in retrieved_k:
            # 문서 내용에서 관련 키워드 찾기
            content = doc.page_content.lower()
            for relevant_concept in relevant_docs:
                if relevant_concept.lower() in content:
                    relevant_retrieved += 1
                    break
        
        return relevant_retrieved / k

    def perform_search_benchmark(self, vector_store, model_name: str, test_queries: List[Dict]):
        """실제 검색 성능 벤치마크"""
        print(f"\n🔍 {model_name} 검색 성능 테스트")
        print("-" * 50)
        
        search_times = []
        precision_scores = []
        search_results = []
        
        for i, query_data in enumerate(test_queries):
            query = query_data["query"]
            expected_concepts = query_data["relevant_concepts"]
            
            print(f"   {i+1}. 쿼리: '{query}'")
            
            # 검색 시간 측정
            start_time = time.time()
            try:
                # 실제 검색 수행
                results = vector_store.similarity_search(query, k=5)
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                # Precision@5 계산
                precision = self.precision_at_k(expected_concepts, results, 5)
                precision_scores.append(precision)
                
                print(f"      검색 시간: {search_time*1000:.1f}ms")
                print(f"      Precision@5: {precision:.3f}")
                
                # 상위 결과 미리보기
                if results:
                    preview = results[0].page_content[:100].replace('\n', ' ')
                    print(f"      상위 결과: {preview}...")
                
                search_results.append({
                    "query": query,
                    "search_time": search_time,
                    "precision": precision,
                    "num_results": len(results)
                })
                
            except Exception as e:
                print(f"      ❌ 검색 실패: {e}")
                search_times.append(float('inf'))
                precision_scores.append(0.0)
        
        # 성능 요약
        valid_times = [t for t in search_times if t != float('inf')]
        avg_search_time = np.mean(valid_times) if valid_times else float('inf')
        avg_precision = np.mean(precision_scores)
        
        print(f"\n📊 {model_name} 성능 요약:")
        if avg_search_time != float('inf'):
            print(f"   평균 검색 시간: {avg_search_time*1000:.1f}ms")
            print(f"   평균 Precision@5: {avg_precision:.3f}")
            print(f"   초당 처리량: {1/avg_search_time:.1f} QPS")
        else:
            print(f"   ❌ 모든 검색 실패")
        
        return {
            "model_name": model_name,
            "avg_search_time": avg_search_time,
            "avg_precision": avg_precision,
            "qps": 1/avg_search_time if avg_search_time != float('inf') else 0,
            "search_results": search_results
        }

def create_medical_test_queries():
    """의료 도메인 테스트 쿼리 생성"""
    return [
        {
            "query": "당뇨병 환자 혈당 관리",
            "relevant_concepts": ["당뇨병", "혈당", "인슐린", "혈당조절"]
        },
        {
            "query": "고혈압 진단 치료",
            "relevant_concepts": ["고혈압", "혈압", "혈압측정", "혈압약"]
        },
        {
            "query": "심장병 증상",
            "relevant_concepts": ["심장", "심근경색", "협심증", "흉통"]
        },
        {
            "query": "폐렴 진단",
            "relevant_concepts": ["폐렴", "기침", "발열", "호흡"]
        },
        {
            "query": "뇌졸중 응급처치",
            "relevant_concepts": ["뇌졸중", "마비", "응급", "뇌"]
        },
        {
            "query": "간염 검사",
            "relevant_concepts": ["간염", "간", "간기능", "검사"]
        }
    ]

def run_proper_benchmark():
    """올바른 벡터 스토어 벤치마크 실행"""
    
    print("🚀 올바른 E5 vs Gemini 벡터 스토어 벤치마크 시작")
    print("=" * 60)
    print("💡 각 모델은 자신의 벡터 스토어를 사용합니다")
    print("   - E5 모델 → medical_vector_store_e5")
    print("   - Gemini 모델 → medical_vector_store")
    print("=" * 60)
    
    # Gemini API 키 확인
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    if not gemini_api_key:
        print("⚠️ GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        gemini_api_key = input("🔑 Google API 키를 입력하세요: ").strip()
        if not gemini_api_key:
            print("❌ Gemini API 키가 필요합니다.")
            return {}
    
    # 벤치마크 객체 생성
    benchmark = ProperVectorStoreBenchmark(gemini_api_key)
    
    # 테스트 쿼리 준비
    test_queries = create_medical_test_queries()
    print(f"📋 준비된 테스트 쿼리: {len(test_queries)}개")
    
    # 벡터 스토어 경로 설정 (올바른 매칭)
    base_path = "VectorStore2/vector_stores"
    e5_path = f"{base_path}/medical_vector_store_e5"        # E5 전용
    gemini_path = f"{base_path}/medical_vector_store"       # Gemini 전용
    
    results = {}
    
    # 1. E5 벡터 스토어 테스트 (E5 모델 사용)
    if os.path.exists(e5_path):
        print(f"\n🔍 E5 벡터 스토어 테스트 (E5 모델 사용)")
        e5_store, e5_model = benchmark.load_e5_vector_store(e5_path)
        
        if e5_store:
            e5_results = benchmark.perform_search_benchmark(
                e5_store, "E5-Large", test_queries
            )
            results["E5"] = e5_results
        else:
            print("❌ E5 벡터 스토어 로드 실패")
    else:
        print(f"⚠️ E5 벡터 스토어 경로 없음: {e5_path}")
    
    # 2. Gemini 벡터 스토어 테스트 (Gemini 모델 사용)
    if os.path.exists(gemini_path):
        print(f"\n🔍 Gemini 벡터 스토어 테스트 (Gemini 모델 사용)")
        gemini_store, gemini_model = benchmark.load_gemini_vector_store(gemini_path)
        
        if gemini_store:
            gemini_results = benchmark.perform_search_benchmark(
                gemini_store, "Gemini-exp-03-07", test_queries
            )
            results["Gemini"] = gemini_results
        else:
            print("❌ Gemini 벡터 스토어 로드 실패")
    else:
        print(f"⚠️ Gemini 벡터 스토어 경로 없음: {gemini_path}")
    
    # 3. 결과 비교 및 출력
    print(f"\n📊 올바른 벤치마크 결과 비교")
    print("=" * 60)
    
    if results:
        comparison_data = []
        for model, data in results.items():
            if data["avg_search_time"] != float('inf'):
                comparison_data.append({
                    "모델": data["model_name"],
                    "벡터차원": "1024D" if model == "E5" else "3072D",
                    "평균 검색시간(ms)": f"{data['avg_search_time']*1000:.1f}",
                    "평균 Precision@5": f"{data['avg_precision']:.3f}",
                    "처리량(QPS)": f"{data['qps']:.1f}",
                    "종합점수": f"{(data['avg_precision'] * 60 + min(data['qps']/10, 1) * 40):.1f}"
                })
            else:
                comparison_data.append({
                    "모델": data["model_name"],
                    "벡터차원": "1024D" if model == "E5" else "3072D",
                    "평균 검색시간(ms)": "실패",
                    "평균 Precision@5": f"{data['avg_precision']:.3f}",
                    "처리량(QPS)": "0",
                    "종합점수": "0"
                })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # 승자 판정 (유효한 결과만)
        valid_results = {k: v for k, v in results.items() if v["avg_search_time"] != float('inf')}
        
        if len(valid_results) == 2:
            e5_score = valid_results["E5"]["avg_precision"] * 0.6 + min(valid_results["E5"]["qps"]/1000, 1) * 0.4
            gemini_score = valid_results["Gemini"]["avg_precision"] * 0.6 + min(valid_results["Gemini"]["qps"]/1000, 1) * 0.4
            
            print(f"\n🏆 최종 승자:")
            if e5_score > gemini_score:
                print(f"   🥇 E5-Large 승리! (점수: {e5_score:.3f} vs {gemini_score:.3f})")
                print(f"   💪 빠른 속도와 안정성으로 승리")
            elif gemini_score > e5_score:
                print(f"   🥇 Gemini 승리! (점수: {gemini_score:.3f} vs {e5_score:.3f})")
                print(f"   🧠 고차원 임베딩으로 승리")
            else:
                print(f"   🤝 무승부! (점수: {e5_score:.3f})")
        elif len(valid_results) == 1:
            winner = list(valid_results.keys())[0]
            print(f"\n🏆 {winner} 모델만 성공적으로 테스트됨")
        else:
            print(f"\n❌ 모든 모델이 실패했습니다.")
    
    # 4. 결과 저장
    output_file = "proper_benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # float('inf') 처리
        def clean_results(obj):
            if isinstance(obj, dict):
                return {k: clean_results(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_results(v) for v in obj]
            elif obj == float('inf'):
                return None
            else:
                return obj
        
        json.dump(clean_results(results), f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과 저장: {output_file}")
    
    print(f"\n✅ 올바른 벤치마크 완료!")
    print(f"💡 각 모델이 자신의 벡터 스토어를 올바르게 사용했습니다.")
    return results

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.WARNING)
    
    try:
        results = run_proper_benchmark()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 벤치마크 실행 오류: {e}")
        import traceback
        traceback.print_exc() 