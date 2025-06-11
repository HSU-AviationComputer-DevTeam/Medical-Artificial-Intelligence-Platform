# Simplified Medical Image Agent

OpenAI Vision API와 HyperCLOVAX를 활용한 간소화된 의료 영상 분석 에이전트

## 🎯 개요

이 프로젝트는 복잡한 로컬 AI 모델 대신 OpenAI Vision API를 사용하여 의료 영상을 분석하고, 그 결과를 HyperCLOVAX에게 전달하여 종합적인 의료 해석을 제공하는 간단하고 효율적인 시스템입니다.

## 🏗️ 시스템 아키텍처

```
User Image Upload → Demo UI → A2A Orchestrator → OpenAI Vision API → Structured Analysis → HyperCLOVAX → Medical Report
```

### 주요 컴포넌트

1. **Simple Vision Pipeline** (`simple_vision_pipeline.py`)

   - OpenAI GPT-4 Vision API를 사용한 이미지 분석
   - 구조화된 분석 결과 생성 (findings, anatomical structures, abnormalities)

2. **Simplified Agent** (`agent.py`)

   - 기본적인 이미지 품질 평가
   - 전처리 및 상담 기능

3. **Task Manager** (`task_manager.py`)

   - A2A 프로토콜 처리
   - OpenAI → HyperCLOVAX 파이프라인 오케스트레이션

4. **A2A Integration** (`adk_host_manager.py`)
   - JSON-RPC 2.0 기반 통신
   - 포트 12000에서 요청 수신, 포트 10002로 전달

## 🚀 주요 기능

- ✅ **OpenAI Vision API 기반 분석**: 복잡한 로컬 모델 대신 API 호출
- ✅ **HyperCLOVAX 통합**: 최종 의료 해석은 한국어 의료 LLM이 담당
- ✅ **A2A 프로토콜 지원**: 병원 시스템과의 표준 연동
- ✅ **실시간 처리**: 메모리 기반 이미지 처리, 파일 저장 없음
- ✅ **간소화된 아키텍처**: 최소한의 의존성, 빠른 배포

## 📦 설치

```bash
# 1. 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정
export OPENAI_API_KEY="your-openai-api-key"
```

## 🏃‍♂️ 실행

```bash
# Medical Image Agent 시작 (포트 10002)
python -m hi_medei.samples.python.agents.medical_image_agent

# 또는 직접 실행
cd hi_medei/samples/python/agents/medical_image_agent
python __main__.py
```

## 🔧 API 엔드포인트

### A2A 프로토콜

- **POST** `/` - A2A JSON-RPC 2.0 요청 처리
- **GET** `/health` - 헬스체크
- **GET** `/agent/info` - 에이전트 정보

### 지원 메서드

- `analyze_image` - 의료 영상 분석
- `health_check` - 시스템 상태 확인

## 📋 사용 예제

### A2A 이미지 분석 요청

```json
{
  "jsonrpc": "2.0",
  "method": "analyze_image",
  "params": {
    "patient_id": "P123456",
    "image_data": "base64_encoded_image_data",
    "modality": "XRAY",
    "body_part": "CHEST",
    "priority": "ROUTINE"
  },
  "id": "analysis_001"
}
```

### 응답 예제

```json
{
  "jsonrpc": "2.0",
  "result": {
    "request_id": "analysis_001",
    "patient_id": "P123456",
    "overall_impression": "정상 흉부 엑스레이 소견입니다.",
    "findings": ["정상 폐야", "정상 심장 크기"],
    "recommendations": ["정기 건강검진 권장"],
    "urgency_level": "ROUTINE",
    "ai_confidence": 0.92
  },
  "id": "analysis_001"
}
```

## 🔗 연동 시스템

### Demo UI (Mesop)

- 포트 12000에서 UI 제공
- 이미지 업로드 및 base64 인코딩
- A2A 요청을 10002 포트로 전달

### HyperCLOVAX Integration

- 모델: `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B`
- OpenAI 분석 결과를 의료 전문 해석으로 변환
- 한국어 의료 용어 및 임상 권고사항 제공

## 📁 주요 파일 구조

```
medical_image_agent/
├── simple_vision_pipeline.py    # OpenAI Vision API 파이프라인
├── agent.py                     # 간소화된 에이전트
├── task_manager.py              # A2A 태스크 관리
├── image_tools.py               # 기본 이미지 도구들
├── models.py                    # 데이터 모델
├── __main__.py                  # 서버 실행 파일
├── requirements.txt             # 의존성 목록
└── README.md                    # 이 파일
```

## 🔧 환경 변수

```bash
# 필수
OPENAI_API_KEY=your-openai-api-key

# 선택적
MEDICAL_IMAGE_AGENT_HOST=0.0.0.0
MEDICAL_IMAGE_AGENT_PORT=10002
```

## 🚨 중요 사항

1. **의료 목적 제한**: 이 시스템은 의료진을 보조하는 도구로, 최종 진단은 반드시 전문의가 수행해야 합니다.

2. **개인정보 보호**: 의료 영상 데이터는 메모리에서만 처리되며 디스크에 저장되지 않습니다.

3. **API 사용량**: OpenAI Vision API 호출 비용을 고려하여 사용하세요.

## 🔄 기존 시스템과의 차이점

### 이전 (복잡한 시스템)

- DenseNet-121, Vision Transformer 등 로컬 AI 모델
- 복잡한 모델 관리 및 GPU 요구사항
- 대용량 의존성 (torch, transformers, MONAI 등)

### 현재 (간소화된 시스템)

- OpenAI Vision API 호출
- 최소한의 의존성 및 CPU 기반 실행
- 빠른 배포 및 확장 가능

## 📞 문의

의료 영상 분석 파이프라인 관련 문의사항이 있으시면 연락해 주세요.
