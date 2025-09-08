# w2v_model_for_cso_classifier
# 🧠 CSO Concept Embedding Workflow Pipeline

This project leverages the [Computer Science Ontology (CSO)](https://cso.kmi.open.ac.uk/) to process academic paper metadata, identify key research topics, and generate word embeddings using **Word2Vec** for semantic analysis and downstream NLP tasks.

---

## ✨ Diagram

**Figure 1:** Workflow diagram illustrating the pipeline for creating computer science concept embeddings using CSO and a paper dataset. The process flows from CSO concept extraction and dataset construction through concept matching to embedding model training, caching, and downstream applications.

![image](https://github.com/user-attachments/assets/7a6c8e44-e510-4106-8a31-e24c555ec5a8)

---

## 🚀 Workflow steps / Pseudo code

### Step 1: Download and Preprocess CSO Concepts

1.1 Download the latest CSO ontology **.ttl** (Turtle) file containing structured computer science concepts: <https://cso.kmi.open.ac.uk/downloads>  
1.2 Extract all concept labels using an RDF parser (e.g., **rdflib** in Python).  
1.3 Preprocess the concept labels:  
- Convert to lowercase  
- Remove extra spaces or special characters  
- Keep multi‑word terms as‑is (e.g., "computer science")

### Step 2: Download Paper Dataset

2.1 Run `notebooks/dataset_construction.ipynb` to download and process the dataset, filtering for **English** texts only. (This can take time.)  
- The downloaded dataset is stored in the `paper_dataset/` folder.

### Step 3: Concept Matching and Replacing

3.1 From `cso_label/cso.csv` (extracted CSO concepts), treat CSO concepts as search terms.  
3.2 For each paper **partition** (title + abstract), search for **exact** matches of CSO concepts.  
3.3 If a match is found:  
- Replace the matched phrase with an **underscore‑separated** version (e.g., `"computer science" → "computer_science"`).  
- This keeps multi‑word terms as a **single token** for training.

**Example CSO Concepts**: `["computer science", "web", "information retrieval", "large language models"]`

- **Original Abstract:**  
  `...recent advances in computer science and large language models have improved web search...`  
- **After Replacement:**  
  `...recent advances in computer_science and large_language_models have improved web search...`

### Step 4: Train W2V Embedding Model

- Split the dataset into an equal number of **partitions** and save them in `paper_dataset/`.  
- Clean and process text; build **bigrams** and **trigrams** for more stable phrases.  
- Train a **Word2Vec** model using Gensim’s latest implementation.  
- Save the model (e.g., `models/264M_model.bin`). Training uses **checkpoint techniques** so that failures resume without starting from scratch.

### Step 5: Cached Model Generation

5.1 For each **word** in the model vocabulary:  
- Retrieve semantically similar terms with `most_similar()` (apply a similarity threshold).  
5.2 Compare retrieved terms against CSO concepts using **Levenshtein similarity** (via **rapidfuzz**).  
5.3 If similarity exceeds a threshold:  
- Link the word to relevant CSO concepts.  
- **Cache** the result to avoid recomputation.  
5.4 Save the final cached model with all matched terms (e.g., `cache/cached-token-to-cso-combined.json`).

### Step 6: Topic Modeling or Downstream Tasks

6.1 The final trained Word2Vec model can be used for:  
- Topic modeling  
- Semantic clustering  
- Recommendation systems  
- Query expansion, etc.

---

## 🧩 Pseudo Code

```python
# Step 1: Load and Preprocess CSO Concepts
cso_labels = load_ttl("cso_label/CSO/CSO.3.5.ttl")  # or your chosen TTL
cso_labels = [clean_text(label) for label in cso_labels]  # lowercase, trim, safe punctuation

# Step 2: Download & Preprocess Paper Dataset
titles, abstracts = run_dataset_construction_notebook()  # or custom loader
documents = preprocess_documents(titles + abstracts)     # tokenization/normalization

# Step 3: Match and Replace CSO Concepts in Documents
processed_docs = []
for doc in documents:
    for concept in cso_labels:
        if concept in doc:
            doc = doc.replace(concept, concept.replace(" ", "_"))
    processed_docs.append(doc)

# Step 4: Phrase Mining + Train Word2Vec
phrases = extract_phrases(processed_docs, min_count=5)  # bigrams/trigrams (optional)
for phrase in phrases:
    processed_docs = [d.replace(phrase, phrase.replace(" ", "_")) for d in processed_docs]

tokenized_docs = [d.split() for d in processed_docs]
model = train_word2vec(tokenized_docs, checkpoint_dir="checkpoints/")
save_model(model, "models/264M_model.bin")  # resume on failure via checkpoints

# Step 5: Cached Model Generation
cache = {}
for word in model.wv.index_to_key:
    neighbors = model.wv.most_similar(word, topn=50)
    for neighbor, score in neighbors:
        if score < 0.55:
            continue
        # compare neighbor to CSO labels via Levenshtein (rapidfuzz)
        match = best_fuzzy_match(neighbor, cso_labels)  # returns (label, lev_score)
        if match and match[1] >= 85:
            cache.setdefault(word, set()).add(match[0])

save_json("cache/cached-token-to-cso-combined.json", {k: sorted(v) for k, v in cache.items()})

# Step 6: Use Embeddings for Topic Modeling / Search, etc.
```

---

## ✨ Features

- **Ontology‑based Concept Extraction** — Parses CSO ontology and extracts research topics.  
- **Text Normalization & Tokenization** — Replaces concepts with underscore form (e.g., `large_language_models`), supports phrase mining (bigrams/trigrams).  
- **Word Embedding Training** — Trains **Word2Vec** on preprocessed paper metadata with checkpointing.  
- **Caching & Matching** — Links model tokens to CSO concepts via vector + Levenshtein similarity and caches results.  
- **Modular Architecture** — Extend to semantic search, clustering, topic modeling.

---

## 📁 Project Structure (example)

```
cso_label/
├─ CSO/                         # Place CSO TTLs (e.g., CSO.3.5.ttl)
└─ cso.csv                      # Extracted CSO concept labels

paper_dataset/                  # Downloaded & processed papers (partitions)
├─ part_000.txt
├─ part_001.txt
└─ ...

notebooks/
└─ dataset_construction.ipynb   # Download + English-only filtering

scripts/
├─ 1_cso_script.py              # Extract CSO labels from TTL → CSV
├─ 2_dataset_partitions.py      # Split dataset into partitions
├─ 3_clean_data.py              # Clean texts + CSO underscore replacements
├─ 4_strip_tokens.py            # Lowercase, strip punctuation, tokenize
├─ 5_bigrams_trigrams.py        # Build bigrams/trigrams (Gensim)
├─ 6_w2v_model.py               # Train Word2Vec with checkpoints
└─ 7_caching_word2vec_model.py  # Cache vocab ↔ CSO matches via similarity

models/
└─ 264M_model.bin               # Trained Word2Vec model (example)

cache/
├─ token-to-cso-combined.json
└─ cached-token-to-cso-combined.json

README.md
```

---

## 🛠️ Setup

```bash
git clone https://github.com/yourusername/w2v_model_for_cso_classifier.git
cd w2v_model_for_cso_classifier
pip install -r requirements.txt
# or
pip install pandas gensim nltk rdflib rapidfuzz tqdm
```

---

## 🔎 Notes & Tips
 
- **Checkpoints:** Save model checkpoints periodically to resume after interruptions.  
- **Versions:** python==3.11 and gensim==4.3.3,


---

## 📜 License

MIT (see `LICENSE`). Ensure external datasets are used in accordance with their terms.
