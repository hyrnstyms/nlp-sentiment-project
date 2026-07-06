import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import config

# Pandas işlemlerinde ilerleme çubuğu (progress bar) göstermek için
tqdm.pandas()

# NLTK Türkçe etkisiz kelimeleri (ama, ve, veya, için vb.) indir
nltk.download('stopwords', quiet=True)
TR_STOP_WORDS = set(stopwords.words('turkish'))

def clean_text(text):
    """Gelen metni makine öğrenmesi için gürültülerden arındırır."""
    if not isinstance(text, str):
        return ""
    
    # 1. Küçük harfe çevir (Türkçe karakterlere dikkat ederek)
    text = text.lower()
    text = text.replace('i̇', 'i').replace('I', 'ı')
    
    # 2. Sadece harfleri bırak (Rakam, noktalama, emoji ve linkleri siler)
    text = re.sub(r'[^a-zçğışöü]', ' ', text)
    
    # 3. Fazladan boşlukları teke düşür
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Stopwords (Etkisiz kelime) temizliği
    words = text.split()
    words = [w for w in words if w not in TR_STOP_WORDS]
    
    return ' '.join(words)

def load_and_preprocess_data():
    """Veriyi okur ve temizlenmiş halini döndürür."""
    print(f"Veri okunuyor: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    print(f"Veri boyutu: {df.shape}")
    
    # 'text' yerine 'combined_text' kullanıyoruz
    df = df.dropna(subset=['combined_text', 'label'])
    
    print("Metin temizleme işlemi başlıyor... (Bu işlem birkaç dakika sürebilir)")
    # İşlemi 'combined_text' sütununa uyguluyoruz
    df['cleaned_text'] = df['combined_text'].progress_apply(clean_text)
    
    return df

# Test kodu
if __name__ == "__main__":
    df_cleaned = load_and_preprocess_data()
    print("\nTemizleme başarılı! İşte ilk 5 örnek:")
    print(df_cleaned[['combined_text', 'cleaned_text', 'label']].head())