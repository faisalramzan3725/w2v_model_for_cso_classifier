import logging
import simplejson as json
import os
from gensim.models import word2vec
import glob

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Parameters
MODEL_NAME = '9M'  # model name
SIZE = 256  # size of embedding for output
WINDOW = 10  # max distance between the current and predicted words
MIN_COUNT = 10  # Reduced for sample data, frequency of word appearance, 1 or 10 etc

def read(file_path):
    """
    Read tokenized sentences from a JSON file.
    """
    logging.info(f"Reading {file_path}")
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:  # Changed to text mode with utf-8
            for line in f:
                try:
                    sentences.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {e}")
        logging.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    except FileNotFoundError:
        logging.warning(f"{file_path} not found. Skipping.")
        return []

def main_word2vec():
    """
    Train a Word2Vec model on trigram-processed sentences.
    """
    logging.info("Started training Word2Vec model")
    print("Loading data ... ")

    # Dynamically find all trigram files
    base_dir = "paper_dataset"
    pattern = os.path.join(base_dir, "abstracts_trigrams_part_v1_*.txt")
    trigram_files = glob.glob(pattern)
    
    if not trigram_files:
        logging.error("No trigram files found. Exiting.")
        print("Error: No trigram files found. Check input files.")
        return

    # Read all trigram files
    sentences = []
    for file_path in trigram_files:
        if os.path.exists(file_path):
            sentences.extend(read(file_path))
        else:
            logging.warning(f"{file_path} not found. Skipping.")

    print(f"Found {len(sentences)} sentences...")

    if not sentences:
        logging.error("No sentences loaded. Exiting.")
        print("Error: No sentences loaded. Check input files.")
        return

    print("\n--- Word2Vec Model Parameters ---")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Vector Size: {SIZE}")
    print(f"Window Size: {WINDOW}")
    print(f"Minimum Count: {MIN_COUNT}")
    print(f"Algorithm: Skip-gram (sg=1)")

    print("\n-------------------------------------\nCreating model...")
    try:
        model = word2vec.Word2Vec(
            sentences,
            vector_size=SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            sg=1, 
            workers=4  # Adjust based on your CPU
        )
    except Exception as e:
        logging.error(f"Error training Word2Vec model: {e}")
        print(f"Error training model: {e}")
        return
    '''
    print("\n--- Test Set: Sample Word Vectors ---")
    try:
        sample_words = ['machine_learning', 'neural_network', 'artificial_intelligence']
        for word in sample_words:
            if word in model.wv:
                vector = model.wv[word][:5]
                print(f"Word: {word}, Vector (first 5 dims): {vector}")
            else:
                print(f"Word: {word} not in vocabulary")
    except Exception as e:
        logging.error(f"Error printing vectors: {e}")
        print(f"Error printing vectors: {e}")
        '''
    print("\n-------------------------------------\nSaving model...")
    output_file = f"{MODEL_NAME}[{SIZE}-{WINDOW}]_sg.bin"
    try:
        model.wv.save_word2vec_format(output_file, binary=True)
        logging.info(f"Saved model to {output_file}")
        print(f"Saved model to {output_file}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main_word2vec()