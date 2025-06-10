# train_word2vec.py

import pandas as pd
from gensim.models import Word2Vec

def train_and_save_word2vec(csv_path, model_path='word2vec_cso.model'):
    df = pd.read_csv(csv_path)
    df['trigrams'] = df['trigrams'].apply(eval)  # Convert string to list

    sentences = df['trigrams'].tolist()
    
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    print(f" Word2Vec model saved at: {model_path}")
    return model
