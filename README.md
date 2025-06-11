# Hi Medie - Medical AI Platform 

<table>
  <tr align="center">
    <td width="320px">
      <a href="https://github.com/isshoman123" target="_blank">
        <img src="https://avatars.githubusercontent.com/isshoman123" alt="isshoman123" />
      </a>
    </td>
    <td width="320px">
      <a href="https://github.com/dongsinwoo" target="_blank">
        <img src="https://avatars.githubusercontent.com/dongsinwoo" alt="dongsinwoo" />
      </a>
    </td>
    </td>
        <td width="320px">
      <a href="https://github.com/Jeon3458" target="_blank">
        <img src="https://avatars.githubusercontent.com/Jeon3458" alt="	Jeon3458" />
      </a>
    </td>
    <td width="320px">
      <a href="https://github.com/espada105" target="_blank">
        <img src="https://avatars.githubusercontent.com/espada105" alt="espada105" />
      </a>
  </tr>
  <tr align="center">
    <td>
      Jaewon Kim
    </td>
    <td>
      Dongwoo Shin
    </td>
    <td>
      Hyeonseong Jeon
    </td>
    <td>
      Seongin Hong
    </td>
  </tr>
    <tr align="center">
    <td>
      jaewon
    </td>
    <td>
      dongwoo  
    </td>
    <td>
      hyeonseong
    </td>
    <td>
      Building MCP servers, loading MCPs, building vector stores, generating virtual medical patient data, connecting AI agents to MCPs
    </td>
  </tr>  
</table>

## 의료 인공지능 플랫폼

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![A2A](https://img.shields.io/badge/Protocol-A2A-orange.svg)
![MCP](https://img.shields.io/badge/Protocol-MCP-purple.svg)

**Agent2Agent(A2A) 프로토콜과 Model Context Protocol(MCP)을 활용한 의료 AI 에이전트 플랫폼**

---

## 프로젝트 개요

Hi Medie는 의료진의 업무 효율성을 향상시키기 위해 개발된 AI 기반 의료 플랫폼입니다. A2A와 MCP 프로토콜을 통합하여 다양한 의료 시스템과 연동하고, 의료진이 환자 데이터를 효율적으로 관리하고 진료 문서를 자동 생성할 수 있도록 지원합니다.

### 주요 목표
- **표준화된 의료 AI 에이전트 통신**: A2A 프로토콜 기반 에이전트 간 협업
- **외부 시스템 연동**: MCP를 통한 병원 DB, 의료 기록, 논문 검색 시스템 연결
- **의료 업무 자동화**: 진료 문서 작성, 환자 검색, 처방 관리 자동화
- **확장 가능한 아키텍처**: 새로운 의료 시스템 쉽게 통합 가능

---

## 아키텍처

```
[의료진] ↔ [A2A 클라이언트] ↔ [A2A 서버] ↔ [MCP 클라이언트] ↔ [의료 시스템들]
                                      ↓
                              [의료 AI 에이전트]
                                      ↓
                              [환자 데이터 / 의료 지식]
```

### 핵심 컴포넌트
- **A2A 서버**: 에이전트 간 통신 및 태스크 관리
- **MCP 클라이언트**: 외부 의료 시스템 연동
- **의료 AI 에이전트**: 환자 데이터 분석 및 진료 지원
- **벡터 데이터베이스**: 환자 유사성 검색 및 의료 지식 검색

---

## 주요 기능

### 환자 관리 시스템
- **스마트 환자 검색**: ID, 이름, 증상 기반 다차원 검색
- **벡터 유사성 검색**: 증상 기반 유사 환자 탐색
- **진료과별 분류**: 내과/외과/당일진료 환자 관리
- **처방 이력 추적**: 특정 약물 처방 이력 검색

### 진료 문서 자동화
- **SOAP 노트 생성**: 구조화된 진료 기록 자동 작성
- **처방전 작성**: AI 기반 약물 처방 추천
- **진료 요약서**: 환자 전체 이력 종합 분석
- **의료진 인계서**: 교대 시 환자 상태 인계 문서

### AI 분석 엔진
- **증상 분석**: 환자 증상 기반 진단 지원
- **패턴 분석**: 과거 진료 이력 기반 예측
- **응급도 평가**: 환자 상태 우선순위 자동 판정
- **약물 상호작용 검사**: 처방 안전성 검증

### MCP 외부 시스템 연동
- **PubMed 연동**: 의학 논문 실시간 검색
- **병원 DB 연결**: 환자 기록 시스템 통합
- **의료 기록 시스템**: 전자의무기록(EMR) 연동
- **약물 데이터베이스**: 처방 정보 및 부작용 확인
- **검사 결과 시스템**: 임상병리/영상의학 결과 조회

---

## 설치 및 실행

### 1. 환경 설정

```bash
# 레포지토리 클론
git clone https://github.com/your-repo/Medical-Artificial-Intelligence-Platform.git
cd Medical-Artificial-Intelligence-Platform

# Python 환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r hi_medei_requirements.txt
```

### 2. 환경 변수 설정

```bash
# hi_medei/samples/python/agents/medical_agent/.env 파일 생성
echo "OPENAI_API_KEY=your_openai_api_key" > hi_medei/samples/python/agents/medical_agent/.env
echo "ANTHROPIC_API_KEY=your_anthropic_api_key" >> hi_medei/samples/python/agents/medical_agent/.env
```

### 3. MCP 서버 시작

```bash
cd hi_medei/samples/python/agents/medical_agent

# MCP 서버들 시작 (별도 터미널)
python start_mcp_servers.py
```

### 4. A2A 서버 시작

```bash
# A2A 서버 시작 (별도 터미널)
python __main__.py --port 10001
```

### 5. 클라이언트 연결

```bash
# CLI 클라이언트 실행 (별도 터미널)
cd hi_medei/samples/python/hosts/cli
python -m cli
```

---

## 테스트

### 연결 테스트
```bash
# MCP-A2A 연결 테스트
cd hi_medei/samples/python/agents/medical_agent
python test_mcp_connection.py
```

### 성능 벤치마크
```bash
# 벤치마크 테스트 실행
python benchmark_evaluation.py
python proper_benchmark_test.py
```

### 단위 테스트
```bash
# 전체 테스트 실행
cd hi_medei
python -m pytest tests/
```

---

## 사용 예시

### A2A 프로토콜을 통한 에이전트 호출

```python
import asyncio
from common.client import A2AClient

async def main():
    # 의료 에이전트에 연결
    client = A2AClient("http://localhost:10001")
    
    # 환자 검색 요청
    response = await client.send_task({
        "message": {
            "text": "김철수 환자의 최근 진료 기록을 조회해주세요"
        }
    })
    
    print(response)

asyncio.run(main())
```

### MCP를 통한 외부 시스템 호출

```bash
curl -X POST http://localhost:10001 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "mcp/search_pubmed",
    "params": {
      "query": "diabetes treatment guidelines",
      "max_results": 5
    },
    "id": 1
  }'
```

---

## 개발 가이드

### 프로젝트 구조

```
Medical-Artificial-Intelligence-Platform/
├── hi_medei/                          # A2A 프로토콜 기반 의료 플랫폼
│   ├── samples/python/agents/medical_agent/  # 의료 AI 에이전트
│   │   ├── __main__.py               # A2A 서버 메인
│   │   ├── agent.py                  # 의료 에이전트 로직
│   │   ├── medical_tools.py          # 의료 도구들
│   │   ├── mcp_config.py            # MCP 연결 관리
│   │   ├── mcp_client.py            # MCP 클라이언트
│   │   └── task_manager.py          # 태스크 관리
│   ├── docs/                         # A2A 프로토콜 문서
│   └── specification/                # A2A 프로토콜 스펙
├── data/                             # 환자 데이터
├── VectorStore2/                     # 벡터 임베딩
├── GeminiVectorStore/               # Gemini 벡터 스토어
└── benchmarks/                       # 성능 테스트
```

### 새로운 MCP 서버 추가

```python
# mcp_config.py에 새 엔드포인트 추가
new_endpoint = MCPEndpoint(
    name="new_system",
    url="http://localhost:8085/mcp/new",
    description="새로운 의료 시스템"
)
mcp_manager.add_endpoint(new_endpoint)
```

### 커스텀 의료 도구 개발

```python
from medical_tools import BaseTool

class CustomMedicalTool(BaseTool):
    name = "custom_tool"
    description = "커스텀 의료 도구"
    
    def _run(self, query: str) -> str:
        # 도구 로직 구현
        return "처리 결과"
```

---

## 보안 및 준수사항

### 의료 정보 보호
- **HIPAA 준수**: 환자 정보 암호화 및 접근 제어
- **데이터 익명화**: 개발/테스트 환경에서 실제 환자 데이터 보호
- **감사 로그**: 모든 의료 데이터 접근 이력 기록

---

**© 2024 Medical AI Platform Team. All rights reserved.**
