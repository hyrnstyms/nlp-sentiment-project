import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import config

def prepare_features(df):
    """
    Veriyi Eğitim ve Test olarak böler, ardından TF-IDF uygulayarak metinleri vektörlere çevirir.
    Eğitilmiş TF-IDF modelini ileride (canlı sistemde) kullanılmak üzere kaydeder.
    """
    print("\nVeri Eğitim (%80) ve Test (%20) olarak bölünüyor...")
    
    # X: Temizlenmiş metinler, y: Duygu etiketleri (label)
    X = df['cleaned_text']
    y = df['label']
    
    # Sınıf dağılımlarını korumak için stratify=y parametresini kullanıyoruz
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y 
    )
    
    print(f"TF-IDF Vektörizasyonu başlatılıyor (Maksimum kelime: {config.TFIDF_MAX_FEATURES})...")
    
    # TF-IDF modelini tanımla
    vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES)
    
    # DİKKAT: fit_transform sadece eğitim verisine uygulanır!
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Test verisine sadece transform uygulanır (yeni kelimeler öğrenilmez)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Modellerin kaydedileceği klasör yoksa oluştur
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # TF-IDF Vektörizeri diske kaydet (.pkl formatında)
    with open(config.TFIDF_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"TF-IDF Modeli başarıyla kaydedildi: {config.TFIDF_PATH}")
    print(f"Eğitim Seti Matris Boyutu: {X_train_tfidf.shape}")
    print(f"Test Seti Matris Boyutu: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

# Eğer sadece bu dosyayı test etmek istersen çalışacak kod bloğu
if __name__ == "__main__":
    from data_loader import load_and_preprocess_data
    df = load_and_preprocess_data()
    X_train_vec, X_test_vec, y_train, y_test = prepare_features(df)