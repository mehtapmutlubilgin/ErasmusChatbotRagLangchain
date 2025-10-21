# 🎓 🇪🇺 Erasmus Bilgi Asistanı - RAG Chatbot Projesi

## Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, RAG (Retrieval Augmented Generation) temelli bir chatbot sistemidir. Chatbot, Erasmus+ programı hakkındaki sıkça sorulan sorulara (SS) hızlı, doğru ve bağlama dayalı yanıtlar sağlamak amacıyla bir web arayüzü üzerinden sunulmaktadır.

---

### 1.Projenin Amacı

Projenin temel amacı, Büyük Dil Modellerini (LLM) özel bir bilgi kaynağıyla (Erasmus veri seti) besleyerek halüsinasyon riskini en aza indirmek ve kullanıcılara güvenilir, kanıta dayalı yanıtlar sunan etkileşimli bir sohbet arayüzü sağlamaktır.

### 2.Veri Seti Hakkında Bilgi

**Dosya Adı:** `erasmus_dataset.csv`
**İçerik:** Veri seti, Erasmus+ programı hakkında derlenmiş 50 adet soru-cevap çifti içermektedir. Veriler, Öğrenci, Öğretmen ve Genel olmak üzere üç ana kategoriyi kapsar.
**Metodoloji:** Hazır bir SS veri seti kullanılmıştır. Bu veri, RAG mimarisinde kullanılmak üzere parçalanmış ve vektörleştirilmiştir.

### 3.Kullanılan Başlıca Yöntemler ve Teknolojiler

Bu RAG sistemi, modern yapay zeka araçları kullanılarak inşa edilmiştir:

| Teknoloji | Amaç | Detay |
| :--- | :--- | :--- |
| **Orkestrasyon** | RAG zinciri yönetimi. | **LangChain**. |
| **Vektör Veritabanı** | Metin vektörlerinin depolanması. | **ChromaDB**. |
| **Gömme (Embedding)** | Metni vektöre dönüştürme. | **HuggingFace** (`all-MiniLM-L6-v2`). |
| **Üretim (Generation)** | Nihai cevabı oluşturma. | **Gemini API**. |
| **Arayüz** | Chatbot'u yayınlama. | **Streamlit**. |

---

### 4.Çözüm Mimarisi ve İş Akışı

Bu proje, RAG mimarisinin klasik adımlarını takip eden bir zincir kullanır ve Erasmus veri seti ile ilgili spesifik bir problemi çözer:

* **Problem:** Erasmus programı gibi niş bir konuda, LLM'lerin genel bilgisini kullanmak yerine, kesin ve kurumun onayladığı bilgiye dayalı yanıt verme sorununu çözmektedir.

#### İş Akışı Adımları:

1.  **Soru Girişi:** Kullanıcı Streamlit arayüzünden bir soru sorar.
2.  **Gömme (Embedding):** Gelen soru, **HuggingFace** modeli kullanılarak bir vektöre dönüştürülür.
3.  **Geri Getirme (Retrieval):** Soru vektörü, **ChromaDB**'de depolanan Erasmus veri vektörleri arasında en yüksek benzerliğe sahip (top-k) ilgili kaynak metinleri bulur ve geri getirir.
4.  **Prompt Hazırlama:** Geri getirilen kaynak metinler, orijinal soruyla birlikte bir **LangChain Prompt Şablonu** içine yerleştirilir.
5.  **Üretim (Generation):** Hazırlanan bu Prompt, **Gemini LLM**'e gönderilir ve model, sadece sağlanan bağlama dayanarak cevabı üretir.
6.  **Sunum:** Üretilen cevap, kaynak metinlerle birlikte Streamlit arayüzünde kullanıcıya gösterilir.

---

### 5.Elde Edilen Sonuçlar Özeti

Geliştirilen chatbot, Erasmus veri setindeki bilgilere dayanarak tutarlı ve doğru yanıtlar sunmaktadır. LLM'in cevabı üretirken kullandığı kaynak metinler, şeffaflık sağlamak amacıyla arayüzde gösterilmektedir.

### 6. Çalışma Kılavuzu

Projenin kurulum ve çalıştırma adımları için detaylı rehbere buradan ulaşabilirsiniz: [GUIDE.md](GUIDE.md)

## 🔗 Uygulama Linki

Projenin canlı olarak yayınlandığı web arayüzü linki aşağıdadır.

> **https://erasmuschatbotraglangchain-amahtsl8cocd7pvfmnriuc.streamlit.app/**
