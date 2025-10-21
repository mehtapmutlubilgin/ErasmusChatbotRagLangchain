import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
# HuggingFace (Yerel) embedding modelini kullanıyoruz
from langchain_community.embeddings import HuggingFaceEmbeddings 
# Generation için Gemini kullanıyoruz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv

# 1. Sabitler ve Kurulum
load_dotenv() 

# setup_db.py dosyasında oluşturduğumuz yerel veritabanı adı
CHROMA_PATH = "chroma_db_local" 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Yerel model
LLM_MODEL = "gemini-2.5-flash" 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Lütfen GEMINI_API_KEY ortam değişkenini ayarlayın (örn: set komutuyla).")
    st.stop()

# 2. RAG Bileşenlerini Yükleme ve Başlatma
# Caching (st.cache_resource) sorunlarını önlemek için kaldırıldı.
def get_rag_chain(api_key):
    # Embedding Modeli (HuggingFace yerel modelini kullanıyoruz)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    try:
        # Vektör depoyu yükle
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        # Retriever (Geri Getirici) oluştur
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) 
    except Exception as e:
        st.error(f"Vektör veritabanı yüklenirken kritik hata: {e}")
        st.stop()
        
    # LLM (Large Language Model) - Gemini Generation
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.4,
        google_api_key=api_key,
        verbose=True  # Terminalde zincir adımlarını görmemizi sağlar
    )

    # Prompt Şablonu
    template = """Sen bir Erasmus programı asistanısın. Sadece verilen bağlamı kullanarak, kullanıcının sorusuna kibar ve doğru bir yanıt ver.
    Eğer bağlamda yeterli bilgi yoksa, 'Bu konuda elimde yeterli bilgi yok.' veya 'Sadece Erasmus programı ile ilgili sorulara yanıt verebilirim.' diye yanıtla.
    
    Bağlam (Context):
    {context}
    
    Soru (Question):
    {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # RAG Zinciri için yardımcı fonksiyon
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
st.caption("LangChain, ChromaDB (Yerel Embedding) ve Gemini ile oluşturulmuş RAG tabanlı chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# RAG zincirini yükle
rag_chain = get_rag_chain(GEMINI_API_KEY)

# Kullanıcı girişi
if prompt := st.chat_input("Erasmus ile ilgili bir soru sorun..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistan yanıtını oluştur ve ekle
    with st.spinner("Asistan yanıtı hazırlanıyor..."):
        try:
            # RAG Zincirini çağır
            response = rag_chain.invoke({"question": prompt})

            # Yanıtı ve kaynakları çekme
            answer = response['answer'].content
            
            # Kaynakları formatla
            sources = []
            for i, doc in enumerate(response['docs']):
                kategori = doc.metadata.get('kategori', 'N/A')
                soru = doc.metadata.get('soru', 'N/A')
                cevap = doc.metadata.get('cevap', doc.page_content)
                
                # Kaynak metnini 'cevap' sütunundan alıyoruz.
                sources.append(f"**Kaynak {i+1}** (Kategori: {kategori}, Soru: {soru}): {cevap}")
            
            sources_text = "\n\n**Bağlamda Kullanılan Kaynaklar:**\n" + "\n\n".join(sources)
            
            full_response = answer + "\n\n---\n" + sources_text

            # Asistan yanıtını göster ve geçmişe kaydet
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Kullanılan Kaynakları Gör"):
                    st.markdown(sources_text)
                
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
             st.error(f"Yanıt oluşturulurken bir hata oluştu. Hata: {e}")
             st.session_state.messages.append({"role": "assistant", "content": f"Hata: {e}"})