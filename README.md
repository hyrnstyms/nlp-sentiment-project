# 📊 Türkçe E-Ticaret Duygu Analizi (Sentiment Analysis) ve Web Arayüzü

Bu proje, Türkçe e-ticaret platformlarındaki müşteri yorumlarını Doğal Dil İşleme (NLP) ve Makine Öğrenmesi teknikleri kullanarak **"Olumlu"**, **"Olumsuz"** ve **"Nötr"** olarak yüksek doğruluk oranıyla sınıflandırmayı sağlayan uçtan uca bir veri bilimi boru hattıdır (pipeline).

Proje, tek bir dosyaya yazılmış "spagetti kod" yaklaşımından kaçınılarak; sürdürülebilir, ölçeklenebilir ve sektör standartlarına uygun **modüler bir mimari** ile tasarlanmıştır.

---

## 🚀 Öne Çıkan Özellikler

* **Modüler Veri Boru Hattı:** Veri okuma, ön işleme, özellik çıkarımı ve model eğitimi süreçleri birbirinden bağımsız modüllere ayrılmıştır.
* **Türkçe NLP Ön İşleme:** Sondan eklemeli dil yapısına uygun olarak noktalama, sayı ve emoji temizliği yapılmış; `NLTK` kütüphanesi ile duygu taşımayan kelimeler (stopwords) filtrelenmiştir.
* **Özellik Çıkarımı (Feature Engineering):** Temizlenen metinler, algoritmaların işleyebilmesi için **TF-IDF** (Term Frequency-Inverse Document Frequency) yöntemiyle yüksek boyutlu matematiksel vektörlere dönüştürülmüştür.
* **Canlı Web Arayüzü (Gradio):** Eğitilen modelin anında test edilebilmesi, son kullanıcıya veya müşteriye sunulabilmesi için interaktif bir web uygulaması (UI) entegre edilmiştir.

---

## 🗂️ Veri Seti

Çalışmada Kaggle üzerinde yer alan [Hepsiburada 300K Balanced Reviews](https://www.kaggle.com/datasets/metinhsimimi/hepsiburada-300k-reviews-from-hf) veri seti kullanılmıştır. 

Makine öğrenmesi modellerinde sıkça karşılaşılan "sınıf dengesizliği" (class imbalance) problemini ortadan kaldırmak amacıyla, her bir sınıftan eşit sayıda (toplam 300.000 satır) veri içeren bu set tercih edilmiştir:
* **100.000 Olumlu** (Etiket: 2)
* **100.000 Nötr** (Etiket: 1)
* **100.000 Olumsuz** (Etiket: 0)

---

## 🏗️ Proje Mimarisi ve Dizin Yapısı

Proje aşağıdaki dizin yapısına ve iş akışına göre çalışır:

```text
nlp-sentiment-project/
├── data/                  # Kaggle'dan indirilen ham CSV veri seti
├── models/                # Eğitilmiş algoritmalar (.pkl) ve TF-IDF vektörizeri
├── src/                   # Ana kaynak kodlar
│   ├── config.py          # Proje ayarları, dosya yolları ve hiperparametreler
│   ├── data_loader.py     # Veri okuma ve Türkçe metin temizleme
│   ├── features.py        # Temiz metni TF-IDF matrislerine dönüştürme
│   ├── train.py           # LinearSVC ve XGBoost modellerinin eğitilmesi
│   └── evaluate.py        # Modellerin test seti üzerinde değerlendirilmesi
├── app.py                 # Gradio tabanlı web arayüzü
├── requirements.txt       # Gerekli Python kütüphaneleri
└── README.md              # Proje dokümantasyonu
```

---

## 📈 Model Performansı ve Analiz

300.000 satırlık veri, **%80 Eğitim** ve **%20 Test** olacak şekilde ayrılmıştır. Metin verilerinin oluşturduğu yüksek boyutlu ve seyrek (sparse) matrislerde, destek vektör makinelerinin lineer versiyonu olan **LinearSVC**, ağaç tabanlı XGBoost algoritmasını geride bırakarak temel model seçilmiştir.

| Algoritma | Genel Doğruluk (Accuracy) | Olumsuz F1 | Nötr F1 | Olumlu F1 |
| :--- | :---: | :---: | :---: | :---: |
| **LinearSVC** | **%76.13** | 0.83 | 0.65 | 0.80 |
| **XGBoost** | %75.53 | 0.82 | 0.66 | 0.79 |

**💡 Veri Bilimi Notu:**
Sonuçlar incelendiğinde, modelin "Olumlu" ve "Olumsuz" yorumları birbirinden ayırmada oldukça başarılı (%80-83 F1-Score) olduğu görülmektedir. "Nötr" sınıfındaki görece düşük skorun sebebi, e-ticaret platformlarındaki nötr yorumların genellikle zıt anlamlı duygu durumlarını aynı cümlede barındırmasıdır (*Örn: "Kargo çok hızlı geldi ama ürünün rengini hiç beğenmedim"*). Bu karmaşıklık, kelime frekansına dayalı algoritmalar için endüstride sık karşılaşılan bir durumdur.

---

## 💻 Kurulum ve Çalıştırma Adımları

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Depoyu Klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/nlp-sentiment-project.git
cd nlp-sentiment-project
```

**2. Bağımlılıkları Yükleyin:**
```bash
pip install -r requirements.txt
```

**3. Veri Setini Hazırlayın:**
Kaggle'dan indirdiğiniz veri setinin adını `dataset.csv` olarak değiştirip, proje içerisindeki `data/` klasörüne yerleştirin.

**4. Veriyi İşleyin ve Modelleri Eğitin:**
Aşağıdaki komut; arka planda veri okuma, ön işleme, TF-IDF dönüştürme ve model eğitimi süreçlerini sırayla çalıştırır.
```bash
python src/train.py
```

**5. Performansı Değerlendirin:**
Modellerin test seti üzerindeki detaylı başarı raporunu (Classification Report & Confusion Matrix) görmek için:
```bash
python src/evaluate.py
```

**6. 🌐 Canlı Web Arayüzünü Başlatın:**
Kullanıcıların metin girip anlık analiz yapabilmesi için Gradio arayüzünü başlatın:
```bash
python app.py
```
*Bu komut size yerel bir link (localhost) ve projeyi telefonunuzdan/diğer cihazlardan test edebilmeniz için 72 saatlik açık bir web linki (public URL) sunacaktır.*