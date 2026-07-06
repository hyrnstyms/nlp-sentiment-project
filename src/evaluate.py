import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import config
from data_loader import load_and_preprocess_data

def evaluate_model(model_name, model_path, X_test, y_test):
    """Belirtilen modeli diskten yükler ve test verisi üzerinde değerlendirir."""
    print(f"\n{'='*40}")
    print(f"--- {model_name} Modeli Değerlendiriliyor ---")
    
    # Modeli yükle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Tahminleri al
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    acc = accuracy_score(y_test, y_pred)
    print(f"\nGenel Doğruluk (Accuracy): %{acc * 100:.2f}")
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Olumsuz', 'Nötr', 'Olumlu']))
    
    print("Karmaşıklık Matrisi (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    print("Test verisi hazırlanıyor...")
    
    # 1. Veriyi okuyup temizle
    df = load_and_preprocess_data()
    X = df['cleaned_text']
    y = df['label']
    
    # 2. Eğitim ve Test olarak veriyi BİREBİR AYNI random_state ile böl
    # Modeller sadece X_train'i gördü, biz test için X_test'i ayırıyoruz
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # 3. Eğitilmiş TF-IDF Vektörizer'ı diskten yükle
    print(f"\nTF-IDF Modeli yükleniyor: {config.TFIDF_PATH}")
    with open(config.TFIDF_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    # 4. Test metinlerini vektörlere (sayılara) dönüştür
    # DİKKAT: Burada fit_transform değil, sadece transform kullanıyoruz!
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Modelleri değerlendir
    evaluate_model("LinearSVC", config.SVM_MODEL_PATH, X_test_vec, y_test)
    evaluate_model("XGBoost", config.XGB_MODEL_PATH, X_test_vec, y_test)