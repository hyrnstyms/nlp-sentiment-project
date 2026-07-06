import os
import time
import pickle
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import config
from data_loader import load_and_preprocess_data
from features import prepare_features

def train_svm(X_train, y_train):
    """LinearSVC algoritmasını eğitir ve diske kaydeder."""
    print("\n" + "="*30)
    print("--- SVM Modeli Eğitiliyor ---")
    start_time = time.time()
    
    # Metin verilerinde (sparse matrislerde) LinearSVC çok daha hızlı ve etkilidir
    svm_model = LinearSVC(random_state=config.RANDOM_STATE)
    svm_model.fit(X_train, y_train)
    
    # Modeli .pkl olarak kaydet
    with open(config.SVM_MODEL_PATH, 'wb') as f:
        pickle.dump(svm_model, f)
        
    print(f"SVM Modeli Kaydedildi: {config.SVM_MODEL_PATH}")
    print(f"SVM Eğitim Süresi: {time.time() - start_time:.2f} saniye")
    return svm_model

def train_xgboost(X_train, y_train):
    """XGBoost algoritmasını eğitir ve diske kaydeder."""
    print("\n" + "="*30)
    print("--- XGBoost Modeli Eğitiliyor ---")
    start_time = time.time()
    
    # XGBoost için karar ağacı parametreleri
    # Çok sınıflı (Olumsuz=0, Nötr=1, Olumlu=2) sınıflandırma için logloss metriği kullanıyoruz
    xgb_model = XGBClassifier(
        eval_metric='mlogloss',
        random_state=config.RANDOM_STATE,
        n_estimators=100,
        max_depth=6
    )
    xgb_model.fit(X_train, y_train)
    
    # Modeli .pkl olarak kaydet
    with open(config.XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_model, f)
        
    print(f"XGBoost Modeli Kaydedildi: {config.XGB_MODEL_PATH}")
    print(f"XGBoost Eğitim Süresi: {time.time() - start_time:.2f} saniye")
    return xgb_model

if __name__ == "__main__":
    print("Uçtan Uca Model Eğitim Hattı Başlatılıyor...")
    
    # 1. Veriyi Oku ve Temizle
    df = load_and_preprocess_data()
    
    # 2. TF-IDF Vektörlerini Oluştur
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    # 3. Modelleri Eğit
    train_svm(X_train, y_train)
    train_xgboost(X_train, y_train)
    
    print("\n" + "="*30)
    print("Tüm eğitim işlemleri başarıyla tamamlandı! Modeller 'models/' klasöründe test edilmeyi bekliyor.")