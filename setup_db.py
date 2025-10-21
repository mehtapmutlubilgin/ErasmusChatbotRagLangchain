import os
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# BURAYI DEĞİŞTİRDİK:
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from dotenv import load_dotenv

# .env dosyasından GEMINI_API_KEY'i yükle
load_dotenv()
# Sabitler
CSV_FILE = "erasmus_dataset.csv"
CHROMA_PATH = "chroma_db_local" # Yeni bir klasör kullanalım
# BURAYI DEĞİŞTİRDİK:
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# LLM kısmı setup_db.py'de yok, o yüzden burada sadece embedding değişiyor.


# ====================================================================
# 1. VERİ YÜKLEME VE PARÇALAMA FONKSİYONU
# ====================================================================

def load_and_split_data(file_path):
    # LangChain CSV Loader kullanarak veriyi yükle
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


# ====================================================================
# 2. VEKTÖR DEPO OLUŞTURMA FONKSİYONU
# ====================================================================

def create_vector_store(documents):
    # Artık Google API anahtarına ihtiyacımız yok

    # Embedding Modelini başlat (Yerel model)
    # Bu model, dosyayı internetten bir kere indirir, sonra hep yerelden çalışır.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # ChromaDB'yi oluştur ve dokümanları kaydet
    print("Vektör veritabanı oluşturuluyor ve dokümanlar yerel model ile gömülüyor...")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Dokümanlar {CHROMA_PATH} klasörüne başarıyla kaydedildi.")


# ====================================================================
# 3. ANA ÇALIŞTIRMA BLOĞU
# ====================================================================

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        print(f"HATA: {CSV_FILE} dosyası bulunamadı. Lütfen kontrol edin.")
    else:
        documents = load_and_split_data(CSV_FILE)
        if documents:
            create_vector_store(documents)