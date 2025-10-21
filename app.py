import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv

# 1. Sabitler ve Kurulum
# Ortam değişkenlerini (API anahtarını) yükle
load_dotenv() 

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "models/text-embedding-004" 
LLM_MODEL = "gemini-2.5-flash" 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Lütfen GEMINI_API_KEY ortam değişkenini ayarlayın (örn: set komutuyla veya .env dosyasında).")
    st.stop()

# 2. RAG Bileşenlerini Yükleme ve Başlatma
# @st.cache_resource sayesinde uygulama yeniden yüklendiğinde bu bileşenler tekrar oluşturulmaz.
@st.cache_resource
def get_rag_chain(api_key):
    # Embedding Modeli ve Vektör Deposu
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key # Anahtar buraya iletildi
    )
    try:
        # Vektör depoyu yükle
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        # Retriever (Geri Getirici) oluştur
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    except Exception as e:
        st.error(f"Vektör veritabanı yüklenirken hata oluştu. setup_db.py çalıştı mı? Hata: {e}")
        st.stop()
        
    # LLM (Large Language Model)
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.1,
        google_api_key=api_key # Anahtar buraya iletildi
    )

    # Prompt Şablonu
    template = """Sen bir Erasmus programı asistanısın. Sadece verilen bağlamı kullanarak, kullanıcının sorusuna kibar ve doğru bir yanıt ver.
    Eğer bağlamda yeterli bilgi yoksa, 'Bu konuda elimde yeterli bilgi yok.' veya 'Sadece Erasmus programı ile ilgili sorulara yanıt verebilirim.' diye yanıtla.
    
    Bağlam (Context):
    {context}
    
    Soru (Question):
    {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # RAG Zinciri için yardımcı fonksiyon (Dokümanları okunabilir metne çevirir)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Cevap oluşturma zinciri
    rag_chain_from_docs = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
    )
    
    # Ana RAG zinciri (Cevap ve kaynak dokümanları döndürmek için)
    rag_chain_with_source = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    ) | {
        "answer": rag_chain_from_docs,
        "docs": itemgetter("context")
    }

    return rag_chain_with_source

# 3. Streamlit Arayüzü
st.set_page_config(page_title="Erasmus RAG Chatbot", layout="wide")
st.title("🇪🇺 Erasmus Bilgi Asistanı")
st.caption("LangChain, ChromaDB ve Gemini ile oluşturulmuş RAG tabanlı chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# RAG zincirini yükle
# Anahtarın global olarak ayarlandığından emin olunduğu için burası çalışır
rag_chain = get_rag_chain(GEMINI_API_KEY)

# Kullanıcı girişi
if prompt := st.chat_input("Erasmus ile ilgili bir soru sorun..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan yanıtını oluştur ve ekle
    with st.spinner("Asistan yanıtı hazırlanıyor..."):
        # RAG Zincirini çağır
        response = rag_chain.invoke({"question": prompt})

        answer = response['answer'].content
        
        # Kaynakları formatla
        sources = []
        for i, doc in enumerate(response['docs']):
            # CSV Loader'dan gelen metadata'ları kullan
            kategori = doc.metadata.get('kategori', 'N/A')