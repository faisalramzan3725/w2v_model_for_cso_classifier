from gensim.models import Word2Vec
import pandas as pd

def train_and_save_word2vec(trigrams_csv_path, model_save_path="word2vec_cso.model", vector_size=100, window=5, min_count=1, workers=4, epochs=20):
    df = pd.read_csv(trigrams_csv_path)
    if 'trigrams' not in df.columns:
        raise ValueError("The input CSV must contain a 'trigrams' column.")
    sentences = df['trigrams'].apply(eval).tolist()  # If trigrams are stored as stringified lists
    print("\n🔧 Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    model.save(model_save_path)
    print(f"Word2Vec model trained and saved as '{model_save_path}'")
    return model