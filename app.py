import sys
import os

# Python'a 'src' klasörünün içindeki dosyaları da tanımasını söylüyoruz
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import pickle
import config
from data_loader import clean_text

# Uygulama başlarken modelleri bir kez diske okuyup RAM'e alıyoruz
print("Modeller yükleniyor...")
with open(config.SVM_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(config.TFIDF_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

def predict_sentiment(text):
    """Kullanıcıdan gelen metni alır, temizler, vektörize eder ve tahmin döndürür."""
    if not text.strip():
        return "Lütfen analiz edilecek bir yorum girin."
        
    # 1. Metni temizle
    cleaned_text = clean_text(text)
    
    # 2. Vektörize et (Sayılara dönüştür)
    text_vec = vectorizer.transform([cleaned_text])
    
    # 3. Model ile tahmin yap
    prediction = model.predict(text_vec)[0]
    
    # 4. Sonucu kullanıcı dostu formata çevir
    results = {
        0: "Olumsuz 😡", 
        1: "Nötr 😐", 
        2: "Olumlu 🤩"
    }
    
    return results[prediction]

# Gradio Arayüz Tasarımı
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="E-ticaret yorumunu buraya yapıştırın..."),
    outputs=gr.Text(label="Yapay Zeka Analiz Sonucu"),
    title="🎯 Türkçe Duygu Analizi (Sentiment Analysis)",
    description="SVM ve TF-IDF kullanılarak eğitilmiş makine öğrenmesi modeli. Yorumun duygu durumunu anında analiz edin.",
    examples=[
        ["Kargolama inanılmaz hızlıydı, ürünün kalitesi de beklediğimden çok daha iyi. Kesinlikle tavsiye ederim."],
        ["Paketleme o kadar kötüydü ki ürün elime kırık ulaştı. Param tamamen çöpe gitti, sakın almayın."],
        ["Ürün elime zamanında ulaştı, kargo hızı normaldi ama rengi fotoğraftakinden biraz daha soluk duruyor."]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)