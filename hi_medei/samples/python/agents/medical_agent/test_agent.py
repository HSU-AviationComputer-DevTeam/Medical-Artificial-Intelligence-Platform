"""Test script for Patient Data Manager Agent."""

import os
import asyncio
import json
from dotenv import load_dotenv

from hi_medei.samples.python.agents.medical_agent.agent import PatientDataManagerAgent


async def test_medical_agent():
    """의료 에이전트 테스트"""
    
    # 환경변수 로드
    load_dotenv()
    
    # API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        return
    
    print("🏥 Patient Data Manager Agent 테스트 시작")
    print("=" * 50)
    
    try:
        # 에이전트 초기화
        agent = PatientDataManagerAgent(
            openai_api_key=openai_api_key,
            data_path="../../../../data"
        )
        print("✅ 에이전트 초기화 완료")
        
        # 테스트 케이스들
        test_cases = [
            {
                "name": "환자 검색 테스트",
                "query": "김철수라는 이름의 환자를 찾아주세요."
            },
            {
                "name": "증상 기반 검색 테스트", 
                "query": "복통 증상이 있는 환자들을 찾아주세요."
            },
            {
                "name": "SOAP 노트 생성 테스트",
                "query": """다음 환자 정보로 SOAP 노트를 생성해주세요:
                환자명: 홍길동, 나이: 45세, 주 호소: 복통, 
                증상: 복통, 구토, 발열, 활력징후: 혈압 140/90, 맥박 100, 체온 38.5도"""
            },
            {
                "name": "응급도 평가 테스트",
                "query": """다음 환자의 응급도를 평가해주세요:
                증상: 흉통, 호흡곤란, 활력징후: 혈압 200/120, 맥박 130, 체온 37.2도"""
            }
        ]
        
        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 테스트 {i}: {test_case['name']}")
            print("-" * 30)
            print(f"질문: {test_case['query']}")
            print("\n응답:")
            
            try:
                # 동기 방식 테스트
                result = agent.invoke(test_case['query'])
                print(result.get('content', '응답 없음'))
                
            except Exception as e:
                print(f"❌ 테스트 실패: {e}")
        
        # 개별 기능 테스트
        print(f"\n🔧 개별 기능 테스트")
        print("-" * 30)
        
        # 환자 검색 테스트
        print("1. 환자 검색 기능:")
        search_result = agent.search_patients("김", "name", 3)
        print(f"   검색 결과: {search_result.get('total_count', 0)}명")
        
        # 샘플 환자 데이터로 SOAP 노트 생성 테스트
        print("\n2. SOAP 노트 생성 기능:")
        sample_patient = {
            "name": "테스트환자",
            "age": 30,
            "chief_complaint": "두통",
            "symptoms": ["두통", "어지러움"],
            "vital_signs": {
                "혈압": "120/80",
                "맥박": "72",
                "체온": "36.5"
            },
            "diagnosis": "긴장성 두통"
        }
        
        soap_result = agent.generate_soap_note(sample_patient)
        if "error" not in soap_result:
            print("   ✅ SOAP 노트 생성 성공")
        else:
            print(f"   ❌ SOAP 노트 생성 실패: {soap_result['error']}")
        
        # 약물 상호작용 검사 테스트
        print("\n3. 약물 상호작용 검사:")
        sample_medications = [
            {"name": "아스피린", "dosage": "100mg"},
            {"name": "와파린", "dosage": "5mg"}
        ]
        
        interaction_result = agent.check_drug_interactions(sample_medications)
        if "error" not in interaction_result:
            interactions = interaction_result.get('interactions', [])
            print(f"   발견된 상호작용: {len(interactions)}개")
        else:
            print(f"   ❌ 상호작용 검사 실패: {interaction_result['error']}")
        
        # 응급도 평가 테스트
        print("\n4. 응급도 평가:")
        sample_emergency_data = {
            "symptoms": ["흉통", "호흡곤란"],
            "vital_signs": {
                "systolic_bp": 200,
                "heart_rate": 130,
                "temperature": 37.2
            }
        }
        
        urgency_result = agent.assess_urgency(sample_emergency_data)
        if "error" not in urgency_result:
            urgency_level = urgency_result.get('urgency_level', '알 수 없음')
            print(f"   평가된 응급도: {urgency_level}")
        else:
            print(f"   ❌ 응급도 평가 실패: {urgency_result['error']}")
        
        print(f"\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 에이전트 초기화 실패: {e}")


def test_collaboration_feature():
    """LangGraph 에이전트와의 협업 기능 테스트"""
    print(f"\n🤝 협업 기능 테스트")
    print("-" * 30)
    
    try:
        # 환경변수 로드
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            print("❌ OPENAI_API_KEY가 필요합니다.")
            return
        
        agent = PatientDataManagerAgent(
            openai_api_key=openai_api_key,
            data_path="../../../../data"
        )
        
        # 샘플 환자 데이터
        patient_data = {
            "name": "이영희",
            "age": 55,
            "gender": "female",
            "chief_complaint": "지속적인 복통과 체중 감소",
            "symptoms": ["복통", "체중감소", "식욕부진", "피로감"],
            "vital_signs": {
                "혈압": "130/85",
                "맥박": "88",
                "체온": "37.1"
            },
            "medical_history": ["고혈압", "당뇨병"]
        }
        
        medical_question = "복통과 체중 감소를 동반한 55세 여성 환자의 감별진단과 추가 검사에 대해 알려주세요."
        
        # 협업 쿼리 생성
        collaboration_query = agent.collaborate_with_langgraph_agent(
            patient_data, 
            medical_question
        )
        
        print("LangGraph 에이전트에게 전달할 협업 쿼리:")
        print(collaboration_query)
        print("\n✅ 협업 데이터 준비 완료")
        
    except Exception as e:
        print(f"❌ 협업 기능 테스트 실패: {e}")


if __name__ == "__main__":
    print("🚀 Patient Data Manager Agent 테스트 시작")
    
    # 기본 기능 테스트
    asyncio.run(test_medical_agent())
    
    # 협업 기능 테스트
    test_collaboration_feature()
    
    print(f"\n🎉 모든 테스트가 완료되었습니다!")
    print(f"\n📝 다음 단계:")
    print(f"1. 환경변수 설정: cp env_example.txt .env")
    print(f"2. OpenAI API 키 설정")
    print(f"3. 서버 실행: python -m hi_medei.samples.python.agents.medical_agent")
    print(f"4. 클라이언트 연결: cd ../../../hosts/cli && uv run .") 