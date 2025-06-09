# kmi_cso_upgradation
# 🧠 CSO Paper Analysis Project

This project leverages the [Computer Science Ontology (CSO)](https://cso.kmi.open.ac.uk/) to process academic paper metadata, identify key research topics, and generate word embeddings using Word2Vec for semantic analysis and downstream NLP tasks.

---

## 📁 Project Structure

Folder: cso_label/ # Ontology and extracted concepts

Folder: CSO/ # Contains CSO.3.4.1.ttl (ontology file)

- cso_label_counts.csv # Extracted CSO concepts with counts

Folder: paper_dataset/ # Paper metadata and processed versions

- paper_dataset.csv # Input dataset (title and abstract)

- processed_title_abstract_v2.csv # Output after CSO + trigram processing


scripts/

- cso_script.py # Extract concepts from CSO ontology

- concept_find_replace.py # Text preprocessing, concept replacement, n-gram generation

- train_word2vec.py # Train and save Word2Vec model

- model.py # Word2Vec hyperparameter configuration


main.ipynb # Main end-to-end notebook pipeline

requirements.txt # Project dependencies

setup.py # (Optional) Setup for pip install


---

## ✨ Features

- **Ontology-based Concept Extraction**
  - Parses CSO ontology and extracts research topics
- **Text Normalization & Tokenization**
  - Replaces concepts with underscore-form (e.g., `large_language_models`)
  - Supports phrase mining (bigrams/trigrams)
- **Word Embedding Training**
  - Trains Word2Vec model on preprocessed paper metadata
- **Modular Architecture**
  - Can be extended to semantic search, clustering, topic modeling

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

bash
git clone https://github.com/yourusername/kmi_project_cso.git

cd kmi_project_cso


### 2. Install Dependencies
pip install -r requirements.txt

Alternatively, install manually:
pip install pandas gensim nltk rdflib

---

🚀 Usage Workflow


Step 1: Extract CSO Concepts
Run the following in a notebook or script:

from scripts.cso_script import extract_labels_to_csv

extract_labels_to_csv("cso_label/CSO/CSO.3.4.1.ttl", "cso_label/cso_label_counts.csv")


Step 2: Process Papers

from scripts.concept_find_replace import process_dataset

process_dataset(
    "paper_dataset/paper_dataset.csv",
    "cso_label/cso_label_counts.csv",
    save_path="paper_dataset/processed_title_abstract_v2.csv"
)

Step 3: Train Word2Vec

from scripts.train_word2vec import train_and_save_word2vec

train_and_save_word2vec(
    "paper_dataset/processed_title_abstract_v2.csv",
    model_path="word2vec_cso.model"
)

Or run everything step-by-step in main.ipynb.



🧪 Example Use

from gensim.models import Word2Vec

model = Word2Vec.load("word2vec_cso.model")

print(model.wv.most_similar("computer_science"))

📦 Packaging (optional)

You can install this as a Python package:

setup.py

from setuptools import setup, find_packages

setup(
    name="cso_paper_analysis",
    version="0.1",
    packages=find_packages(include=["scripts", "scripts.*"]),
    install_requires=["pandas", "gensim", "nltk", "rdflib"],
)
Then run:

pip install -e .


