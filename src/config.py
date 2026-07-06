import os

# Projenin ana dizinini bul (src klasörünün bir üstü)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Veri seti yolu
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

# Eğitilmiş modellerin kaydedileceği klasör
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model ve Vektörizer (TF-IDF) dosyalarının yolları
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")

# --- Makine Öğrenmesi Parametreleri ---

# TF-IDF: 300.000 veri çok büyük olduğu için, RAM'in şişmemesi adına en sık geçen 10.000 kelimeyi kullanacağız.
TFIDF_MAX_FEATURES = 10000 

# Veri bölme ( %80 Eğitim, %20 Test )
TEST_SIZE = 0.2
RANDOM_STATE = 42