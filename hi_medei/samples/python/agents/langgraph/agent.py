from collections.abc import AsyncIterable
from typing import Any, Literal
import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel
import PyPDF2

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

memory = MemorySaver()

# OpenAI API 키 설정 - 환경변수에서 가져옴
from dotenv import load_dotenv
load_dotenv()  # 다시 한번 로드

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # 공백 제거하여 다시 시도
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    print("현재 환경변수:", repr(os.getenv("OPENAI_API_KEY")))
    raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

# PDF 파싱 함수
def parse_pdf(file_path: str) -> str:
    # 파일 경로가 상대 경로인 경우 여러 기준 경로에서 찾기 시도
    if not os.path.isabs(file_path):
        # 가능한 기준 경로 목록
        base_dirs = [
            os.path.abspath("."),  # 현재 실행 디렉토리
            os.path.abspath("../../../.."),  # 프로젝트 루트
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),  # 파일 기준 프로젝트 루트
        ]
        
        # 여러 경로 조합을 시도하여 파일 찾기
        for base_dir in base_dirs:
            # 직접 경로
            full_path = os.path.join(base_dir, file_path)
            if os.path.exists(full_path):
                file_path = full_path
                break
                
            # temp_uploads 디렉토리 체크
            temp_path = os.path.join(base_dir, "temp_uploads", os.path.basename(file_path))
            if os.path.exists(temp_path):
                file_path = temp_path
                break
                
            # A2A/temp_uploads 디렉토리 체크
            a2a_temp_path = os.path.join(base_dir, "A2A", "temp_uploads", os.path.basename(file_path))
            if os.path.exists(a2a_temp_path):
                file_path = a2a_temp_path
                break
                
            # A2A/samples/python/temp_uploads 디렉토리 체크
            samples_temp_path = os.path.join(base_dir, "A2A", "samples", "python", "temp_uploads", os.path.basename(file_path))
            if os.path.exists(samples_temp_path):
                file_path = samples_temp_path
                break
    
    # 여전히 파일을 찾을 수 없는 경우
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
    
    # 파일 읽기
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text

# 벡터스토어 관리
def get_vectorstore():
    # 벡터 DB 설정
    persist_directory = "chroma_db"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

# PDF 인제스트 함수
def ingest_pdf(file_path: str):
    text = parse_pdf(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    vectordb = get_vectorstore()
    vectordb.add_documents(docs)
    vectordb.persist()
    return len(docs)

# 문서 검색 함수
def search_docs(query: str, k=4):
    vectordb = get_vectorstore()
    docs = vectordb.similarity_search(query, k=k)
    return docs

class PDFQAAgent:
    SYSTEM_INSTRUCTION = (
        'You are a medical document QA assistant. Answer user questions based on the provided PDF context. '
        'Always cite the most relevant reference chunks. If the answer is not in the document, say so.'
    )

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

    def ingest(self, file_path: str) -> str:
        num_chunks = ingest_pdf(file_path)
        return f"PDF 인제스트 완료! 총 {num_chunks}개의 청크가 저장되었습니다."

    def invoke(self, query, sessionId=None) -> dict:
        docs = search_docs(query, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        try:
            answer = self.llm.invoke(prompt)
            references = [d.page_content[:200] for d in docs]
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': answer.content,
                'references': references,
            }
        except Exception as e:
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f"오류 발생: {e}",
                'references': [],
            }

    async def stream(self, query, sessionId=None) -> AsyncIterable[dict[str, Any]]:
        # 스트리밍 미지원, 단일 응답만 반환
        yield self.invoke(query, sessionId)

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain', 'application/pdf']
