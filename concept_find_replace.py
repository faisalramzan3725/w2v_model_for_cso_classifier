import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser

nltk.download('punkt')

# --------------------------------------
# Preprocessing function
# --------------------------------------
def preprocess(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9_\s]', '', text)
    return text

# --------------------------------------
# Replace concepts with underscores
# --------------------------------------
def replace_concepts(text, concepts):
    text = preprocess(text)
    for concept in sorted(concepts, key=lambda x: -len(x)):
        pattern = re.compile(r'\b' + re.escape(concept) + r'\b', flags=re.IGNORECASE)
        text = pattern.sub(concept.replace(' ', '_'), text)
    return text

# --------------------------------------
# Load and preprocess dataset
# --------------------------------------
def process_dataset(dataset_path, cso_path, save_path="processed_title_abstract_v2.csv"):
    # Load files
    df = pd.read_csv(dataset_path)
    cso_df = pd.read_csv(cso_path)

    # Extract concepts
    cso_concepts = sorted(
        cso_df['Label'].dropna().astype(str).str.lower().unique(),
        key=len,
        reverse=True
    )

    # Only keep needed fields
    df = df[['title', 'abstract']].copy()

    # Apply concept replacement
    df['title_processed'] = df['title'].apply(lambda x: replace_concepts(x, cso_concepts))
    df['abstract_processed'] = df['abstract'].apply(lambda x: replace_concepts(x, cso_concepts))

    # Tokenize
    df['tokens'] = (df['title_processed'] + ' ' + df['abstract_processed']).apply(preprocess).apply(word_tokenize)

    # Bigrams and Trigrams

    sentences = df['tokens'].tolist()
    bigram = Phrases(sentences, min_count=2, threshold=5)
    trigram = Phrases(bigram[sentences], threshold=5)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    df['trigrams'] = [trigram_mod[bigram_mod[doc]] for doc in sentences]

    # Save results
    df[['title_processed', 'abstract_processed']].to_csv(save_path, index=False)

    return df
