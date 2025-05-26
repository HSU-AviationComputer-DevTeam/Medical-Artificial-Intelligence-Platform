import os
import logging
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #openMP 충돌방지


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 설정 ---
GOOGLE_API_KEY = input("Google API 키를 입력하세요: ") 
if not GOOGLE_API_KEY.strip():
    logger.error("유효한 Google API 키가 필요합니다.")
    exit()

VECTOR_STORE_PATH = "C:\code\medical_llama3.2"  # index.faiss, index.pkl 있는 경로 설정
OLLAMA_MODEL_NAME = "llama-3.2-Korean-Bllossom-AICA-5B"   # LLM 연결에 사용할 모델

# 🚨 사용자가 요청한 특정 실험적 모델. 현재 지원되지 않을 수 있습니다.
EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07" 

# --- 2. Gemini 임베딩 클래스 정의 (요청 모델 사용) ---
class CustomGeminiEmbeddings:
    """Google Gemini 특정 모델을 사용하는 커스텀 클래스."""
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """여러 문서를 임베딩합니다."""
        if isinstance(texts, str): texts = [texts]
        logger.info(f"'{self.model}'로 문서 임베딩 중...")
        try:
            # google-generativeai 라이브러리 직접 호출
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT" # 대문자 사용 시도
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"'{self.model}' 문서 임베딩 실패: {e}")
            logger.warning("오류 발생! 이 모델은 더 이상 지원되지 않을 수 있습니다.")
            raise # 오류를 다시 발생시켜 중단

    def embed_query(self, text):
        """단일 쿼리를 임베딩합니다."""
        logger.info(f"'{self.model}'로 쿼리 임베딩 중...")
        try:
            result = genai.embed_content(
                model=self.model,
                content=[text],
                task_type="RETRIEVAL_QUERY" # 대문자 사용 시도
            )
            return result['embedding'][0]
        except Exception as e:
            logger.error(f"'{self.model}' 쿼리 임베딩 실패: {e}")
            logger.warning("오류 발생! 이 모델은 더 이상 지원되지 않을 수 있습니다.")
            raise # 오류를 다시 발생시켜 중단

    def __call__(self, text):
        return self.embed_query(text)

# --- 3. 임베딩 객체 및 벡터 스토어 로드 ---
try:
    # Google API 구성
    genai.configure(api_key=GOOGLE_API_KEY.strip())
    
    # 커스텀 임베딩 객체 생성
    embeddings = CustomGeminiEmbeddings(model=EMBEDDING_MODEL)
    logger.info(f"'{EMBEDDING_MODEL}' 커스텀 임베딩 객체를 생성했습니다.")

except Exception as e:
    logger.error(f"Google API 구성 또는 임베딩 객체 생성 중 오류 발생: {e}")
    logger.error(traceback.format_exc())
    exit()

try:
    if os.path.exists(VECTOR_STORE_PATH) and \
       os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        logger.info(f"{VECTOR_STORE_PATH}에서 벡터 스토어 로드 중...")
        vectorstore = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH, 
            embeddings=embeddings, # 커스텀 임베딩 객체 전달
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS 벡터 스토어를 성공적으로 로드했습니다.")
    else:
        logger.error(f"오류: 벡터 스토어 경로를 찾을 수 없습니다: {VECTOR_STORE_PATH}")
        exit()
except Exception as e:
    logger.error(f"벡터 스토어 로드 중 오류 발생: {e}")
    logger.error("🚨 사용된 임베딩 모델과 벡터 스토어가 호환되는지 확인하세요.")
    logger.error(traceback.format_exc())
    exit()

# --- 4. LLM 로드 (HuggingFace Llama 3.2 Bllossom) ---
try:
    logger.info("Hugging Face 모델을 로컬에서 로드 중...")

    model_name_or_path = "Bllossom/llama-3.2-Korean-Bllossom-AICA-5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",          # GPU 사용
        torch_dtype="auto"
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    logger.info(f"HuggingFace '{model_name_or_path}' LLM을 성공적으로 로드했습니다.")

except Exception as e:
    logger.error(f"HuggingFace LLM 로딩 중 오류 발생: {e}")
    logger.error(traceback.format_exc())
    exit()

# --- 5. QA 체인 생성 (프롬프트 커스터마이징 포함) ---

# 한국어 지시가 포함된 프롬프트 템플릿 정의
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 의료 정보를 분석하는 한국어(korea) AI입니다.
다음 문맥(context)을 참고하여 질문에 한국어(korea)로 답변해주세요.
영어를 사용하지마세요.

문맥:
{context}

질문:
{question}
""",
)

# 체인 생성 시 prompt 명시
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)
logger.info("RetrievalQA 체인을 생성했습니다.")

# --- 6. 질문 및 답변 ---
def ask_question(query):
    print(f"\n========================================")
    print(f"💬 질문: {query}")
    print(f"========================================")
    try:
        result = qa_chain.invoke({"query": query})
        print("\n💡 답변:")
        print(result['result'])
        print("\n📚 근거 문서:")
        for i, doc in enumerate(result['source_documents']):
            doc_type = doc.metadata.get('document_type', 'N/A')
            p_name = doc.metadata.get('name', 'N/A')
            p_id = doc.metadata.get('patient_id', 'N/A')
            print(f"  [{i+1}] 유형:{doc_type}, 환자:{p_name}({p_id}) | 내용: {doc.page_content[:300]}...")
    except Exception as e:
        logger.error(f"질의응답 중 오류 발생: {e}")
        logger.error(traceback.format_exc())

# --- 실행 ---
if __name__ == "__main__":
    while True:
        user_query = input("\n질문을 입력하세요 (종료하려면 'q' 입력): ")
        if user_query.lower() == 'q':
            break
        if user_query.strip():
            ask_question(user_query)
        else:
            print("질문을 입력해주세요.")
    print("\n프로그램을 종료합니다.")