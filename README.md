# w2v_model_for_cso_classifier
# 🧠 CSO Concept Embedding Workflow Pipeline

This project leverages the [Computer Science Ontology (CSO)](https://cso.kmi.open.ac.uk/) to process academic paper metadata, identify key research topics, and generate word embeddings using Word2Vec for semantic analysis and downstream NLP tasks.

---

## ✨ Diagram

Figure 1: Workflow diagram illustrating the five-step pipeline for creating computer science concept embeddings using CSO (Computer Science Ontology) and Semantic Scholar paper dataset. The process flows from data preprocessing through concept matching to embedding model training and downstream applications.

![image](https://github.com/user-attachments/assets/7a6c8e44-e510-4106-8a31-e24c555ec5a8)

## 🚀 Workflow steps / Pseudo code:

Step 1: Download and Preprocess CSO Concepts

  1.1 Download the CSO ontology .ttl (Turtle) file, which contains structured computer science concepts.
  
  1.2 Extract all concept labels using an RDF parser (e.g., rdflib in Python).
  
  1.3 Preprocess the concept labels:
  
  - Convert to lowercase
  
  - Remove extra spaces or special characters
  
  - Keep multi-word terms as-is (e.g., "computer science")
    

Step 2: Download and Preprocess the Paper Dataset

  2.1 Download the full Semantic Scholar Open Research Corpus or a custom dataset.
  
  2.2 Parse each paper to extract:
  
   - Title
  
  - Abstract
  
  2.3 Preprocess each document:
  
  - Convert text to lowercase
  
  - Normalize special characters, punctuation, and whitespace

Step 3: Concept Matching with NLTK or Gensim

  3.1 Use CSO concepts as search terms.
  
  3.2 For each paper (title + abstract), search for exact matches of CSO concepts.
  
  3.3 If a match is found:

- Replace the matched phrase with an underscore-separated version (e.g., "computer science" → "computer_science")

- Note: This keeps multi-word terms as a single token for training

  Example: CSO Concepts:
  ["computer science", "web", "information retrieval", "large language models"]
  
  - Original Abstract:
  "...recent advances in computer science and large language models have improved web search..."
  
  - After Replacement:
  "...recent advances in computer_science and large_language_models have improved web search..."

Step 4: Phrase Mining (Optional but Recommended)

  4.1 Extract frequent bigrams and trigrams from the paper dataset using NLTK or Gensim's Phrases module.

Step 5: Train Embedding Model

  5.1 Use the cleaned and processed paper dataset to train a Word2Vec model using gensim's latest implementation.
  
  5.2 The model will learn vector embeddings where similar scientific terms are close in vector space.

Step 6: Load Updated CSO Concepts from CSO Reader

  6.1 Load all updated concepts from the CSO Reader utility.

Step 7: Extend Word2Vec Model Vocabulary

  7.1 For each word in the model vocabulary:
  
  - Retrieve semantically similar terms using most_similar() with a similarity threshold.
  
  7.2 Compare retrieved terms against CSO concepts using Levenshtein similarity (via rapidfuzz).
  
  7.3 If similarity exceeds a threshold:
  
  - Link the word to relevant CSO concepts
  
  - Cache the result to avoid recomputation
  
  7.4 Save the final cached model with all matched terms.

Step 8: Topic Modeling or Downstream Tasks

  8.1 The final trained Word2Vec model can be used for a range of downstream applications:
  
  - Topic modeling
   
  - Semantic clustering
  
  - Recommendation systems
  
  - Query expansion, etc.

---

## Pseudo Code:

      
    Step 1: Load and Preprocess CSO Concepts
    
      cso_labels = load_ttl("cso.ttl")
      
      cso_labels = [clean_text(label) for label in cso_labels]
      
    
    Step 2: Load and Preprocess Paper Dataset
      
      titles, abstracts = load_paper_dataset("papers.txt")
      
      documents = preprocess_documents(titles + abstracts)
      
    
    Step 3: Match and Replace CSO Concepts in Documents
    
      processed_docs = []
      
      for doc in documents:
          for concept in cso_labels:
              if concept in doc:
                  doc = doc.replace(concept, concept.replace(" ", "_"))
          processed_docs.append(doc)
    
    
    Optional Step 4: Phrase Mining
    
      phrases = extract_phrases(processed_docs, min_count=5)
      
      for phrase in phrases:
          for i, doc in enumerate(processed_docs):
              if phrase in doc:
                  processed_docs[i] = doc.replace(phrase, phrase.replace(" ", "_"))
                  
    
    Step 5: Train Word2Vec Model
    
      tokenized_docs = [doc.split() for doc in processed_docs]
      
      model = train_word2vec(tokenized_docs)
      
      save_model(model, "scientific_embeddings.model")
      
    
    Step 6: Use Embeddings for Topic Modeling or Search etc..

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


## 📁 Project Structure

📁 cso_label/ # Ontology and extracted concepts

📁 CSO/ # Contains CSO.3.4.1.ttl (ontology file)

- cso_label_counts.csv # Extracted CSO concepts with counts

📁 paper_dataset/ # Paper metadata and processed versions

- paper_dataset.csv # Input dataset (title and abstract)

- processed_title_abstract_v2.csv # Output after CSO + trigram processing

--- 

## Files and Descriptions

1_cso_script.py: This script extracts concept labels from a CSO (Computer Science Ontology) TTL file using regex pattern matching and stores them in a list. The extracted labels can optionally be saved to a CSV file, with error handling and logging functionality implemented throughout the process.

2_paper_dataset.py: This script processes a dataset of academic papers by loading them from a CSV file, filtering for titles and abstracts, and saving them in both TXT and CSV formats. The script uses pandas for data manipulation and includes logging functionality to track the processing steps and any potential errors that may occur during execution.

3_dataset_partitions.py: This script partitions a large dataset of academic paper abstracts into smaller chunks while preserving title-abstract pairs. It implements memory-efficient processing by reading and writing data incrementally, with configurable partition sizes. The script handles large files by splitting them into at least 4 parts based on system memory constraints, with proper logging and preview functionality.

4_clean_data.py: This script processes academic paper abstracts by cleaning text and standardizing space-related topics from CSO (Computer Science Ontology). It reads multiple partition files, replaces space-separated topics with underscore versions (e.g., "machine learning" -> "machine_learning"), and outputs cleaned versions while maintaining a log of transformations performed.

5_strip_tokens.py: This script processes text files containing paper abstracts by cleaning and tokenizing them. It reads input files from 'paper_dataset' directory, converts text to lowercase, removes punctuation, and saves the tokenized output in JSON format while providing progress logs and sample outputs.

6_bigrams_trigrams.py: This script processes text files to generate bigrams and trigrams using Gensim's Phrases model. It reads input files containing tokenized sentences, creates bigrams and trigrams based on frequency thresholds, and saves the processed n-grams to separate output files while handling errors and logging progress.

7_w2v_model.py: This script trains a Word2Vec model on academic paper abstracts using the skip-gram algorithm. It processes trigram-tokenized sentences from multiple files and generates word embeddings of specified dimensions. The trained model is saved in binary format with configurable parameters like vector size, window size and minimum word count.

8_caching_word2vec_model.py: This script matches terms between Word2Vec model vocabulary and Computer Science Ontology (CSO) topics using semantic and string similarity. The matches are cached to a JSON file for later use in topic classification and recommendation systems.

Pipeline.py - Data Processing Pipeline for Academic Papers Analysis

This Python script implements a comprehensive data processing pipeline for analyzing academic papers. The pipeline consists of 8 main steps:

Key Components:

1. Logging Configuration:
   - Custom formatted logging to 'myapp.log'
   - Includes timestamp, log level, and messages
   
2. Dynamic Module Import:
   - Custom import_module() function for dynamic loading of Python modules
   - Handles file path validation and module loading

3. Main Pipeline Steps:
   a. CSO Concept Extraction (Step 1)
   b. Paper Dataset Processing (Step 2) 
   c. Abstract Partitioning (Step 3)
   d. Data Cleaning (Step 4)
   e. Token Stripping (Step 5)
   f. Bigram/Trigram Generation (Step 6)
   g. Word2Vec Model Training (Step 7)
   h. Word2Vec-CSO Similarity Caching (Step 8)

4. Performance Monitoring:
   - Execution time tracking for each step
   - Identification of most time-consuming step
   - Comprehensive error handling and logging



requirements.txt: All project dependencies (e.g., pandas, langdetect, gensim) required to run the scripts. Use pip install -r requirements.txt to install them.


 
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



