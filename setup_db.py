import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# .env dosyasından GEMINI_API_KEY'i yükle
load_dotenv()
# Sabitler
CSV_FILE = "erasmus_dataset.csv"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "models/text-embedding-004" # Google'ın önerdiği embedding modeli

def load_and_split_data(file_path):
    # LangChain CSV Loader kullanarak veriyi yükle
    # 'cevap' sütununu content olarak kullanıyoruz.
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    
    # Text Splitter ile belgeleri parçala
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(data)
    print(f"Toplamda {len(data)} belge yüklendi. Parçalandıktan sonra {len(documents)} adet doküman parçası oluştu.")
    return documents

def create_vector_store(documents):
    # API Anahtarını os.environ'dan çekin
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    if not gemini_api_key:
        print("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen CMD'de 'set GEMINI_API_KEY=...' komutuyla ayarlayın.")
        return

    # Embedding Modelini başlat
    # Anahtarı doğrudan parametre olarak iletiyoruz.
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=gemini_api_key 
    )

    # ChromaDB'yi oluştur ve dokümanları kaydet
    print("Vektör veritabanı oluşturuluyor ve dokümanlar gömülüyor (embedding yapılıyor)...")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Dokümanlar {CHROMA_PATH} klasörüne başarıyla kaydedildi.")

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"HATA: {CSV_FILE} dosyası bulunamadı. Lütfen kontrol edin.")
    else:
        documents = load_and_split_data(CSV_FILE)
        if documents:
            create_vector_store(documents)