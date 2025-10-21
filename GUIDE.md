# ⚙️ Erasmus Bilgi Asistanı - Çalışma ve Ürün Kılavuzu

Bu kılavuz, **Akbank GenAI Bootcamp** projesinin mimarisini, yerel kurulum adımlarını ve Streamlit arayüzünün nasıl kullanılacağını detaylı olarak açıklamaktadır.

---

## 🧩 1. Çözüm Mimarisi

Proje, **Retrieval-Augmented Generation (RAG)** mimarisini kullanarak Büyük Dil Modellerinin (LLM) spesifik bir bilgi kaynağına erişimini sağlar.

### 🔧 Mimari Bileşenler

| Bileşen | Teknoloji | Amaç |
| :--- | :--- | :--- |
| **Orkestrasyon** | LangChain (LCEL) | RAG zincirindeki tüm adımları yönetir. |
| **Veri Yükleme** | Pandas & CSVLoader | `erasmus_dataset.csv` dosyasını LangChain formatına çevirir. |
| **Parçalama** | RecursiveCharacterTextSplitter | Metinleri vektörleştirmeye uygun parçalara ayırır. |
| **Gömme (Embedding)** | HuggingFace (`all-MiniLM-L6-v2`) | Metinleri sayısal vektörlere dönüştürür. |
| **Vektör Depo** | ChromaDB | Vektörleri kalıcı olarak depolama ve sorgulama. |
| **Üretim (Generation)** | Gemini API (`gemini-2.5-flash`) | Çekilen bağlama dayalı cevabı üretir. |

---

### 🎯 Gömme (Embedding) Modeli Seçim Gerekçesi

Başlangıçta **Gemini API’nin embedding modeli** hedeflenmiştir.  
Ancak yerel geliştirme sürecinde karşılaşılan:

- API anahtarı okuma sorunları  
- DLL yükleme hataları  

nedeniyle, güvenilirliği artırmak için **yerel embedding modeli** tercih edilmiştir.

> **Geçiş Nedeni:**  
> Yerel embedding modeli sayesinde ağ bağlantısı ve sistem DLL hataları ortadan kalkmış, RAG pipeline’ının kararlılığı artırılmıştır.

---

### 💡 Çözülen Problem

Proje, **Erasmus programı** gibi niş bir konuda, LLM’lerin genel bilgisini kullanmak yerine, **kurumun onayladığı bilgiye** dayalı doğru ve kaynaklı yanıt üretme sorununu çözmektedir.

---

## 🧰 2. Çalışma Kılavuzu (Lokal Kurulum)

Projenin yerel ortamda çalıştırılabilmesi için gereken adımlar aşağıda açıklanmıştır.

   **Projeyi Klonlama:**
  
   ```bash
    git clone https://github.com/mehtapmutlubilgin/ErasmusChatbotRagLangchain
    cd ErasmusChatbotRagLangchain
   ```


### 📁 Gerekli Dosyalar

- `requirements.txt`  
- `setup_db.py`  
- `app.py`  
- `.env` (Gemini API anahtarı için)

---

### A. Ortam Kurulumu

1. **Sanal Ortam Oluşturma ve Etkinleştirme**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux / MacOS
   ```

2. **Bağımlılıkların Kurulumu**

   ```bash
   pip install -r requirements.txt
   ```

3.  **API Anahtarının Ayarlanması (.env Kullanımı):**
    * Projenin kök dizininde **`.env`** adında bir dosya oluşturun.
    * Gemini API anahtarınızı aşağıdaki formatta bu dosyanın içine yazın:
        ```
        GEMINI_API_KEY="SİZİN_ANAHTARINIZ"
        ```
    * Kod, `python-dotenv` kütüphanesi sayesinde bu dosyayı otomatik olarak okuyacaktır.
      

4. **Veri Seti Kontrolü:**
 *erasmus_dataset.csv dosyasının projenin ana dizininde bulunduğundan emin olun.

### B. 📦 Vektör Veritabanı Oluşturma

Bu adım, veri setini okur, yerel embedding modeliyle vektörleştirir ve  
`chroma_db_local` klasörünü oluşturur.

İlk çalıştırmada HuggingFace modelini otomatik olarak indirir.

```bash
python setup_db.py
```

---

### C. 🚀 Uygulamayı Çalıştırma

```bash
streamlit run app.py
```

---

## 🌐 3. Web Arayüzü & Product Kılavuzu

### 🔗 Deploy Linki

Uygulamanın aktif dağıtım adresini buraya ekleyin:

> **https://erasmuschatbotraglangchain-amahtsl8cocd7pvfmnriuc.streamlit.app/**

---

### 💬 Çalışma Akışı ve Özellikler

Web arayüzü, kullanıcının sorularını kolayca test edebilmesi için tasarlanmıştır.

| Kabiliyet | Açıklama | Nasıl Test Edilir? |
| :--- | :--- | :--- |
| **Bilgi Çekimi (RAG)** | Yalnızca Erasmus bağlamına dayanarak yanıt üretir. | “Erasmus öğrenim süresi ne kadar?” sorun. Cevabın kaynaktan geldiğini kontrol edin. |
| **Halüsinasyon Önleme** | Kaynakta olmayan soruları reddeder. | “Dünya'nın en büyük gölü nedir?” sorun. Reddedici yanıt beklenir. |
| **Şeffaflık** | Üretilen cevabın kaynaklarını gösterir. | Yanıtın altındaki **“Kullanılan Kaynakları Gör”** alanını açın. |
| **Geri Getirme Doğruluğu** | Sorguyla ilgili en uygun 3 belgeyi çeker. | “Hibesiz Erasmus” hakkında sorun ve belgeleri inceleyin. |

### Örnek Görüntüler:
<img width="1779" height="637" alt="image" src="https://github.com/user-attachments/assets/cf3c8e4a-78c8-4222-8969-abec484b77dd" />

<img width="1764" height="486" alt="image" src="https://github.com/user-attachments/assets/8c941d63-709b-41a0-8e1b-384658fba275" />

<img width="1759" height="452" alt="image" src="https://github.com/user-attachments/assets/10aec371-fa55-440e-b8bb-1fc55a4e980c" />




---

### Contact

* Email:mehtapmutlu.bilgin06@gmail.com
* GitHub:https://github.com/mehtapmutlubilgin
* Linkedin:https://www.linkedin.com/in/mehtap-mutlu-bilgin-925b921b1/

