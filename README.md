# 📊 Türkçe E-Ticaret Duygu Analizi (Sentiment Analysis)

Bu proje, Türkçe e-ticaret platformlarındaki müşteri yorumlarını Doğal Dil İşleme (NLP) ve Makine Öğrenmesi teknikleri kullanarak **"Olumlu"**, **"Olumsuz"** ve **"Nötr"** olarak yüksek doğruluk oranıyla sınıflandırmayı sağlayan uçtan uca bir veri bilimi boru hattıdır (pipeline).

Projenin temel amacı; büyük veri yığınları içerisindeki düzensiz metinleri analiz ederek, markaların ve kurumların kullanıcı deneyimini algoritmik ve ölçeklenebilir bir şekilde ölçebilmesini sağlayan bir sistem mimarisi kurgulamaktır. Tüm kod tabanı, sürdürülebilir ve ölçeklenebilir bir mühendislik yaklaşımıyla modüler olarak tasarlanmıştır.

---

## 🚀 Öne Çıkan Özellikler

* **Modüler Mimari:** Veri okuma, ön işleme, özellik çıkarımı ve model eğitimi süreçleri birbirinden bağımsız modüllere ayrılarak (Spagetti koddan kaçınılarak) profesyonel bir standartta geliştirilmiştir.
* **Türkçe NLP Ön İşleme:** Sondan eklemeli dil yapısına uygun olarak noktalama, sayı ve emoji temizliği yapılmış; NLTK ile duygu taşımayan kelimeler (stopwords) filtrelenmiştir.
* **Özellik Çıkarımı (Feature Engineering):** Temizlenen metinler, algoritmaların işleyebilmesi için **TF-IDF** (Term Frequency-Inverse Document Frequency) yöntemiyle matematiksel vektörlere dönüştürülmüştür.

---

## 🗂️ Veri Seti

Çalışmada Kaggle üzerinde yer alan [Hepsiburada 300K Balanced Reviews](https://www.kaggle.com/datasets/metinhsimimi/hepsiburada-300k-reviews-from-hf) veri seti kullanılmıştır. 
Makine öğrenmesi modellerinde sıkça karşılaşılan "sınıf dengesizliği" (class imbalance) problemini ortadan kaldırmak amacıyla, her bir sınıftan eşit sayıda veri içeren bu set tercih edilmiştir:
* **100.000 Olumlu** (Sınıf: 2)
* **100.000 Nötr** (Sınıf: 1)
* **100.000 Olumsuz** (Sınıf: 0)

---

## 🏗️ Proje Mimarisi ve Veri Akışı

Proje dizini aşağıdaki iş akışına (pipeline) göre çalışır:

* **`src/config.py`:** Projenin sabit ayarlarını (dosya yolları, train/test oranları, hiperparametreler) merkezi olarak tutar.
* **`src/data_loader.py`:** Ham `csv` dosyasını okur ve Türkçe kurallarına uygun metin temizleme (preprocessing) işlemlerini uygular.
* **`src/features.py`:** Temizlenen metin verilerini TF-IDF matrislerine dönüştürür ve yeni verilerde de kullanılmak üzere `tfidf_vectorizer.pkl` olarak diske kaydeder.
* **`src/train.py`:** Vektörize edilmiş verileri kullanarak **LinearSVC** ve **XGBoost** algoritmalarını eğitir ve modelleri kaydeder.
* **`src/evaluate.py`:** Eğitilen modellerin test verisi üzerindeki başarılarını (Confusion Matrix, Precision, Recall, F1-Score) karşılaştırmalı olarak raporlar.

---

## 📈 Model Performansı ve Analiz

300.000 satırlık veri, %80 Eğitim ve %20 Test olacak şekilde ayrılmıştır. Metin verilerinin oluşturduğu yüksek boyutlu ve seyrek (sparse) matrislerde, destek vektör makinelerinin lineer versiyonu (LinearSVC), ağaç tabanlı XGBoost algoritmasını geride bırakarak temel model seçilmiştir.

| Algoritma | Doğruluk (Accuracy) | Olumsuz F1 | Nötr F1 | Olumlu F1 |
| :--- | :---: | :---: | :---: | :---: |
| **LinearSVC** | **%76.13** | 0.83 | 0.65 | 0.80 |
| **XGBoost** | %75.53 | 0.82 | 0.66 | 0.79 |

**💡 Veri Bilimi Notu:**
Sonuçlar incelendiğinde, modelin "Olumlu" ve "Olumsuz" yorumları birbirinden ayırmada oldukça başarılı (%80-83 F1-Score) olduğu görülmektedir. "Nötr" sınıfındaki görece düşük skorun sebebi, e-ticaret platformlarındaki nötr yorumların genellikle zıt anlamlı duygu durumlarını aynı cümlede barındırmasıdır (*Örn: "Kargo çok hızlı geldi ama ürünün rengini hiç beğenmedim"*). Bu karmaşıklık, kelime frekansına dayalı geleneksel algoritmalar için beklenen bir zorluktur.

---

## 💻 Kurulum ve Çalıştırma

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Depoyu Klonlayın:**
`git clone <repository-url>`
`cd nlp-sentiment-project`

**2. Gerekli Kütüphaneleri Yükleyin:**
Bağımlılıkları kurmak için sanal bir ortam (virtual environment) kullanılması tavsiye edilir.
`pip install -r requirements.txt`

**3. Veri Setini Hazırlayın:**
Kaggle'dan indirdiğiniz veri setinin adını `dataset.csv` olarak değiştirip `data/` klasörünün içerisine yerleştirin.

**4. Veriyi İşleyin ve Modelleri Eğitin:**
Aşağıdaki komut, arka planda veri okuma, temizleme ve TF-IDF dönüşümlerini yaparak algoritmaları eğitecektir.
`python src/train.py`

**5. Performansı Değerlendirin:**
Modellerin test seti üzerindeki detaylı başarı raporunu görmek için:
`python src/evaluate.py`