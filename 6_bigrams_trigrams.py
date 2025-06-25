import logging
import json
import os
from gensim.models.phrases import Phrases, Phraser
import glob

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read(file_path):
    """
    Read tokenized sentences from a JSON file.
    """
    logging.info(f"Reading {file_path}")
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sentences.append(json.loads(line.strip()))
    except FileNotFoundError:
        logging.warning(f"{file_path} not found. Skipping.")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {file_path}: {e}")
        return []
    logging.info(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences

def create_bigrams(sentences):
    """
    Create bigrams from a list of tokenized sentences.
    """
    logging.info("Building bigrams...")
    bigram = Phrases(sentences, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)  # Optimize for faster processing
    logging.info("Modifying sentences with bigrams...")
    sentences_bigram = [bigram_phraser[sentence] for sentence in sentences]
    return sentences_bigram, bigram_phraser

def create_trigrams(sentences, bigram_phraser):
    """
    Create trigrams from bigram-processed sentences.
    """
    logging.info("Building trigrams...")
    trigram = Phrases(sentences, min_count=5, threshold=10)
    trigram_phraser = Phraser(trigram)
    logging.info("Modifying sentences with trigrams...")
    sentences_trigram = [trigram_phraser[bigram_phraser[sentence]] for sentence in sentences]
    return sentences_trigram

def process_file(input_file, bigram_output, trigram_output, log_interval=10000):
    """
    Process a single input file to generate bigram and trigram outputs.
    """
    logging.info(f"Processing {input_file}")
    sentences = read(input_file)
    if not sentences:
        return

    cnt = 0
    printcounter = 0

    # Create bigrams
    logging.info("Creating bigrams...")
    sentences_bigram, bigram_phraser = create_bigrams(sentences)
    logging.info(f"Saving bigrams to {bigram_output}")
    with open(bigram_output, "w+", encoding="utf-8") as file:
        for s in sentences_bigram:
            if printcounter == log_interval:
                logging.info(f"Processed {cnt} sentences for bigrams")
                printcounter = 0
            file.write(json.dumps(s) + '\n')
            cnt += 1
            printcounter += 1

    # Reset counter for trigrams
    cnt = 0
    printcounter = 0

    # Create trigrams
    logging.info("Creating trigrams...")
    sentences_trigram = create_trigrams(sentences_bigram, bigram_phraser)
    logging.info(f"Saving trigrams to {trigram_output}")
    with open(trigram_output, "w+", encoding="utf-8") as file:
        for s in sentences_trigram:
            if printcounter == log_interval:
                logging.info(f"Processed {cnt} sentences for trigrams")
                printcounter = 0
            file.write(json.dumps(s) + '\n')
            cnt += 1
            printcounter += 1

def main_bigrams_trigrams():
    """
    Main function to process all tokenized abstract parts dynamically.
    """
    base_dir = "paper_dataset"
    pattern = os.path.join(base_dir, "abstracts_filtered_striped_part_v1_*.txt")
    partition_files = glob.glob(pattern)
    
    if not partition_files:
        logging.warning(f"No files found matching pattern {pattern}")
        return

    for input_file in partition_files:
        part = int(os.path.basename(input_file).replace("abstracts_filtered_striped_part_v1_", "").replace(".txt", ""))
        bigram_output = os.path.join(base_dir, f"abstracts_bigrams_part_v1_{part}.txt")
        trigram_output = os.path.join(base_dir, f"abstracts_trigrams_part_v1_{part}.txt")
        
        if os.path.exists(input_file):
            process_file(input_file, bigram_output, trigram_output)
        else:
            logging.warning(f"{input_file} not found. Skipping.")

if __name__ == "__main__":
    main_bigrams_trigrams()