# w2v_model_for_cso_classifier
# 🧠 CSO Concept Embedding Workflow Pipeline

This project leverages the [Computer Science Ontology (CSO)](https://cso.kmi.open.ac.uk/) to process academic paper metadata, identify key research topics, and generate word embeddings using Word2Vec for semantic analysis and downstream NLP tasks.

---

## ✨ Diagram

**Figure 1:** Workflow diagram illustrating the five‑step pipeline for creating computer science concept embeddings using CSO (Computer Science Ontology) and a paper dataset (e.g., Semantic Scholar). The process flows from data preprocessing through concept matching to embedding model training and downstream applications.

![image](https://github.com/user-attachments/assets/7a6c8e44-e510-4106-8a31-e24c555ec5a8)

---

## 🚀 Workflow steps / Pseudo code:

### Step 1: Download and Preprocess CSO Concepts

1.1 Download the latest CSO ontology `.ttl` (Turtle) file, which contains structured computer science concepts: <https://cso.kmi.open.ac.uk/downloads>  
1.2 Extract all concept labels using an RDF parser (e.g., **rdflib** in Python).  
1.3 Preprocess the concept labels:
- Convert to lowercase  
- Remove extra spaces or special characters  
- Keep multi‑word terms as‑is (e.g., `computer science`)

### Step 2: Concept Matching and Replacing

2.1 Use CSO concepts as search terms.  
2.2 For each paper (**title + abstract**), search for exact matches of CSO concepts.  
2.3 If a match is found:
- Replace the matched phrase with an underscore‑separated version (e.g., `computer science` → `computer_science`)  
- **Note:** This keeps multi‑word terms as a single token for training

**Example CSO Concepts:**  
`["computer science", "web", "information retrieval", "large language models"]`

- **Original Abstract:**  
  _"...recent advances in computer science and large language models have improved web search..."_  
- **After Replacement:**  
  _"...recent advances in computer_science and large_language_models have improved web search..."_

### Step 3: Train Embedding Model

3.1 Use the cleaned and processed paper dataset to train a **Word2Vec** model using **gensim**’s latest implementation.  
3.2 The model learns vector embeddings where similar scientific terms are close in vector space.

### Step 4: Load Updated CSO Concepts

Refresh/extend your CSO vocabulary (e.g., newer CSO release) before linking to model tokens.

### Step 5: Extend Word2Vec Model Vocabulary

5.1 For each word in the model vocabulary: retrieve semantically similar terms using `most_similar()` with a similarity threshold.  
5.2 Compare retrieved terms against CSO concepts using **Levenshtein** similarity (via **rapidfuzz**).  
5.3 If similarity exceeds a threshold: link the word to relevant CSO concepts and cache the result to avoid recomputation.  
5.4 Save the final cached model with all matched terms.

### Step 6: Topic Modeling or Downstream Tasks

6.1 The final trained Word2Vec model can be used for a range of downstream applications:
- Topic modeling
- Semantic clustering
- Recommendation systems
- Query expansion, etc.

---

## 🧩 Pseudo Code

```python
# Step 1: Load and Preprocess CSO Concepts
cso_labels = load_ttl("cso.ttl")
cso_labels = [clean_text(label) for label in cso_labels]

# Step 2: Load and Preprocess Paper Dataset
titles, abstracts = load_paper_dataset("papers.txt")
documents = preprocess_documents(titles + abstracts)

# Step 3: Match and Replace CSO Concepts in Documents
processed_docs = []
for doc in documents:
    for concept in cso_labels:
        if concept in doc:
            doc = doc.replace(concept, concept.replace(" ", "_"))
    processed_docs.append(doc)

# Optional Step 4: Phrase Mining
phrases = extract_phrases(processed_docs, min_count=5)
for phrase in phrases:
    for i, doc in enumerate(processed_docs):
        if phrase in doc:
            processed_docs[i] = doc.replace(phrase, phrase.replace(" ", "_"))

# Step 5: Train Word2Vec Model
tokenized_docs = [doc.split() for doc in processed_docs]
model = train_word2vec(tokenized_docs)
save_model(model, "scientific_embeddings.model")

# Step 6: Use Embeddings for Topic Modeling or Search, etc.
```

---

## ✨ Features

- **Ontology‑based Concept Extraction** — Parses CSO ontology and extracts research topics  
- **Text Normalization & Tokenization** — Replaces concepts with underscore form (e.g., `large_language_models`), supports phrase mining (bigrams/trigrams)  
- **Word Embedding Training** — Trains Word2Vec on preprocessed paper metadata  
- **Modular Architecture** — Easily extended to semantic search, clustering, and topic modeling

---

## 📁 Project Structure

```
cso_label/                    # Ontology and extracted concepts
├─ CSO/                       # Contains CSO.3.4.1.ttl, CSO.3.5.ttl (ontology files)
└─ cso_label_counts.csv       # Extracted CSO concepts with counts

paper_dataset/                # Paper metadata and processed versions
└─ paper_dataset.txt          # Final dataset (title and abstract)

# Scripts (examples)
1_cso_script.py               # Extract CSO labels from TTL → CSV
2_dataset_partitions.py       # Split large datasets into partitions
3_clean_data.py               # Clean texts and apply CSO underscore replacements
4_strip_tokens.py             # Lowercase, strip punctuation, and tokenize to JSON
5_bigrams_trigrams.py         # Build bigrams/trigrams with Gensim
6_w2v_model.py                # Train Word2Vec (skip‑gram/CBOW) on trigrams
7_caching_word2vec_model.py   # Cache vocab ↔ CSO matches via similarity

requirements.txt              # Project dependencies
setup.py                      # Package metadata & installation
```

---

## 📄 Files and Descriptions

- **1_cso_script.py** — Extracts concept labels from a CSO TTL file using regex/RDF, outputs labels (CSV optional) with logging and error handling.  
- **2_dataset_partitions.py** — Partitions a large paper dataset into smaller chunks while preserving title‑abstract pairs; memory‑efficient with previews and logging.  
- **3_clean_data.py** — Cleans abstracts; replaces space‑separated CSO topics with underscore forms (e.g., `machine learning` → `machine_learning`); logs transformations.  
- **4_strip_tokens.py** — Normalizes and tokenizes documents (lowercasing, punctuation removal); writes JSON outputs with progress logs.  
- **5_bigrams_trigrams.py** — Generates bigrams/trigrams using **Gensim Phrases**; saves processed n‑grams with robust error handling.  
- **6_w2v_model.py** — Trains **Word2Vec** (e.g., skip‑gram) on trigram token streams; configurable vector size/window/min_count; saves binary model.  
- **7_caching_word2vec_model.py** — Matches Word2Vec vocabulary terms to CSO topics using semantic and string similarity; caches matches to JSON for fast reuse.  
- **requirements.txt** — All dependencies (install with `pip install -r requirements.txt`).  
- **setup.py** — Package configuration for `w2v_model_for_cso_classifier` (Python 3.11, e.g., gensim==4.3.3).

> Author: **Faisal Ramzan** (faisal.ramzan@unica.it)  
> Repository: <https://github.com/faisalramzan3725/w2v_model_for_cso_classifier>

---

## 🛠️ Setup Instructions

### 1) Clone the Repository
```bash
git clone https://github.com/yourusername/w2v_model_for_cso_classifier.git
cd w2v_model_for_cso_classifier
```

### 2) Install Dependencies
```bash
pip install -r requirements.txt
# or install manually
pip install pandas gensim nltk rdflib
```

---
