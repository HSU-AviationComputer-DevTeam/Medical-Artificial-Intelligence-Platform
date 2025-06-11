import re

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from pydantic import BaseModel

app = FastAPI()

analyzer = AnalyzerEngine()

# 한국 휴대전화 번호 패턴
kr_phone_pattern = Pattern(
    name="KR_PHONE",
    regex=r"01[016789]-\d{3,4}-\d{4}",
    score=1.0
)
kr_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[kr_phone_pattern],
    supported_language="en"
)
analyzer.registry.add_recognizer(kr_phone_recognizer)

class TextInput(BaseModel):
    text: str
@app.get("/.well-known/agent.json")
def well_known_agent():
    return JSONResponse(content={
        "id": "masking-agent",
        "name": "개인정보 마스킹 에이전트",
        "description": "한국어 이름과 전화번호를 마스킹하는 에이전트입니다. 민감한 의료 정보를 처리합니다",
        "version": "1.0.0",
        "url": "http://localhost:8000/a2a",
        "language": "ko",
        "type": "REST",
        "capabilities": {
            "masking": True
        },
        "input_modes": ["text"],
        "output_modes": ["text"],
        "streaming": False,
        "actions": [
            {
                "id": "mask_text",
                "name": "마스킹 수행",
                "description": "문장 내 개인정보 마스킹 처리 실행",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "마스킹할 문장"
                        }
                    },
                    "required": ["text"]
                }
            }
        ],
        "skills": [
            {
                "id": "korean-ner",
                "name": "Korean Named Entity Recognition",
                "description": "3글자 이름 및 조사 기반 한글 인명 추출"
            },
            {
                "id": "pii-filtering",
                "name": "PII Masking",
                "description": "전화번호 및 기타 개인정보 마스킹"
            }
        ]
    })



    
@app.post("/a2a")
def refined_mask(input: TextInput):
    text = input.text

    # 1. Mask PHONE_NUMBER
    results = analyzer.analyze(text=text, language='en')
    results = sorted(results, key=lambda r: r.start)
    offset = 0
    for result in results:
        entity_text = text[result.start + offset:result.end + offset]
        if result.entity_type == "PHONE_NUMBER":
            groups = re.split(r"-", entity_text)
            if len(groups) == 3:
                masked = f"{groups[0]}-****-{groups[2]}"
            else:
                masked = "******"
            text = text[:result.start + offset] + masked + text[result.end + offset:]
            offset += len(masked) - (result.end - result.start)

    return {
    "name": "masking-agent",
    "description": "Handles masking of Korean phone numbers and names",
    "capabilities": ["masking"],
    "tool": {
        "type": "llm",
        "output": {
            "masked_text": text
        }
    }
}
