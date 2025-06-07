# Patient Data Manager Agent with A2A Protocol

병원 내 환자 데이터 검색 및 진료문서 작성을 위한 의료 AI 에이전트입니다.
**A2A(Agent-to-Agent) 프로토콜과 MCP(Model Context Protocol) 연동을 지원합니다.**

## 주요 기능

### 🔍 환자 검색 시스템

- **환자 ID/이름 검색**: 기본 환자 정보 조회
- **증상 기반 검색**: 벡터 검색을 활용한 유사 증상 환자 검색
- **진료과별 검색**: 내과/외과/당일진료 분류별 환자 검색
- **처방 이력 검색**: 특정 약물 처방 이력이 있는 환자 검색

### 📝 진료문서 작성

- **SOAP 노트 생성**: 구조화된 진료 기록 자동 생성
- **처방전 작성**: AI 기반 약물 처방 추천 및 상호작용 검사
- **진료 요약서**: 환자의 전체 진료 이력 요약
- **의료진 인계서**: 교대 시 필요한 환자 상태 인계 문서

### 🧠 AI 분석 기능

- **증상 분석**: 환자 증상을 분석하여 가능한 진단 제시
- **과거 진료 패턴 분석**: 환자의 과거 진료 이력 기반 예측
- **응급도 평가**: 환자 상태의 응급도 자동 평가

### MCP(Model Context Protocol) 지원

이 에이전트는 MCP를 통해 다음과 같은 외부 의료 시스템과 연동할 수 있습니다:

#### 실제 구현된 MCP 서버들

- 🔬 **`pubmed`**: PubMed 의학 논문 검색 서버 (포트 8080)
- 💾 **`memory`**: 환자 대화 기록 메모리 서버 (포트 8081)
- 📁 **`file_system`**: 의료 문서 파일 시스템 서버 (포트 8082)

#### A2A-MCP 연동 API

1. **MCP 엔드포인트 목록 조회**

```json
{
  "jsonrpc": "2.0",
  "method": "mcp/list_endpoints",
  "params": {},
  "id": 1
}
```

2. **MCP 연결 테스트**

```json
{
  "jsonrpc": "2.0",
  "method": "mcp/connect",
  "params": {
    "endpoint": "pubmed"
  },
  "id": 2
}
```

3. **PubMed 논문 검색**

```json
{
  "jsonrpc": "2.0",
  "method": "mcp/search_pubmed",
  "params": {
    "query": "diabetes mellitus treatment",
    "max_results": 5
  },
  "id": 3
}
```

4. **환자 대화 기록 저장**

```json
{
  "jsonrpc": "2.0",
  "method": "mcp/save_memory",
  "params": {
    "session_id": "session_123",
    "content": "환자가 당뇨병 증상에 대해 문의했습니다.",
    "entry_type": "conversation",
    "patient_id": "P001"
  },
  "id": 4
}
```

## 기존 LangGraph 에이전트와의 협업

### 협업 시나리오

1. **종합 진료 상담**: 환자 데이터 + 의료 문헌 검색
2. **처방 최적화**: 환자 이력 + 최신 가이드라인
3. **진단 지원**: 유사 사례 + 의료 지식 통합

## 기술 스택

- **LangChain/LangGraph**: 에이전트 프레임워크
- **ChromaDB**: 벡터 데이터베이스
- **OpenAI**: LLM 및 임베딩
- **Pydantic**: 데이터 모델 검증
- **A2A Protocol**: 에이전트 간 통신

## 설치 및 실행

1. 환경 설정:

   ```bash
   cd hi_medei/samples/python/agents/medical_agent
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

2. 에이전트 실행:

   ```bash
   uv run . --port 10001
   ```

3. 클라이언트 연결:

   ```bash
   cd ../../../hosts/cli
   uv run .
   ```

4. **MCP 서버들 시작**:

   ```bash
   # 터미널 1에서 MCP 서버들 시작
   cd hi_medei/samples/python/agents/medical_agent
   python start_mcp_servers.py
   ```

5. **A2A 서버 시작**:

   ```bash
   # 터미널 2에서 A2A 서버 시작
   cd hi_medei/samples/python/agents/medical_agent
   python __main__.py --port 10001
   ```

6. **통합 테스트 실행**:
   ```bash
   # 터미널 3에서 테스트 실행
   cd hi_medei/samples/python/agents/medical_agent
   python test_mcp_integration.py
   ```

## 데이터 소스

- `../../../../data/`: 환자 데이터 (내과환자, 외과환자, 당일진료환자, 약처방)
- `../../../../VectorStore2/`: 벡터 임베딩 데이터

## 제한사항

- 현재는 텍스트 기반 입출력만 지원
- MCP 서버가 실행되지 않은 경우 시뮬레이션 응답 제공
- 실제 의료 환경에서는 HIPAA 등 의료 정보 보호 규정 준수 필요
- 데모용 데이터 사용 (실제 환자 데이터 아님)

### API 엔드포인트

#### A2A 표준 엔드포인트

- `POST /`: JSON-RPC 요청 처리
- `POST /stream`: 스트리밍 요청 처리
- `GET /health`: 헬스체크
- `GET /agent-card`: 에이전트 카드 정보
- `GET /.well-known/agent.json`: A2A 표준 에이전트 카드

#### MCP 연동 엔드포인트

- `POST /` + `mcp/list_endpoints`: MCP 엔드포인트 목록 조회
- `POST /` + `mcp/connect`: MCP 연결 테스트

### 사용 예시

#### MCP를 통한 외부 시스템 조회

```bash
curl -X POST http://localhost:10001 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "mcp/connect",
    "params": {
      "endpoint": "hospital_db",
      "query": "김철수 환자 정보 조회"
    },
    "id": 1
  }'
```

### 아키텍처

```
[A2A 클라이언트]
    ↓ (JSON-RPC over HTTP)
[Patient Data Manager Agent (A2A 서버)]
    ↓ (MCP JSON-RPC)
[외부 의료 시스템들 (MCP 서버들)]
    - 병원 DB
    - 의료 기록 시스템
    - 약물 DB
    - 검사 결과 시스템
    - 영상의학 시스템
```

이 구조에서:

- **A2A**: 에이전트 간 협업과 태스크 관리
- **MCP**: 외부 도구와 데이터 소스 연결
