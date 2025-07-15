import logging
import json
import os
from gensim.models.phrases import Phrases, Phraser
import glob

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read(file_path)->list:
    """
    Read and parse sentences from a JSON file.

    Args:
        file_path (str): Path to the input JSON file containing sentences.

    Returns:
        list: A list of parsed sentences from the file. Returns empty list if file not found
              or JSON parsing error occurs.

    The function:
    - Attempts to read the specified file line by line
    - Parses each line as JSON 
    - Logs progress and any errors encountered
    - Returns list of successfully parsed sentences
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

def create_bigrams(sentences)->tuple[list, Phraser]:
    """
    Create bigrams from input sentences using Gensim's Phrases model.

    Args:
        sentences (list): List of tokenized sentences to process.

    Returns:
        tuple: A tuple containing:
            - list: Sentences modified to include bigrams
            - Phraser: Trained bigram phraser model for reuse

    The function:
    - Builds a Phrases model to detect common bigrams
    - Uses min_count=5 to only consider pairs appearing 5+ times
    - Uses threshold=10 to control bigram formation sensitivity
    - Creates an optimized Phraser model for faster processing
    - Applies the phraser to transform input sentences
    - Returns both modified sentences and phraser model
    """
    logging.info("Building bigrams...")
    bigram = Phrases(sentences, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)  # Optimize for faster processing
    logging.info("Modifying sentences with bigrams...")
    sentences_bigram = [bigram_phraser[sentence] for sentence in sentences]
    return sentences_bigram, bigram_phraser

def create_trigrams(sentences, bigram_phraser)->list:
    """
    Create trigrams from input sentences using Gensim's Phrases model.
    Args:
        sentences (list): List of tokenized sentences to process.
        bigram_phraser (Phraser): Trained bigram phraser model for reuse.
    Returns:
        list: Sentences modified to include both bigrams and trigrams.
    The function:
    - Builds a Phrases model to detect common trigrams
    - Uses min_count=5 to only consider pairs appearing 5+ times
    - Uses threshold=10 to control trigram formation sensitivity
    - Creates an optimized Phraser model for faster processing
    - Applies the phraser to transform input sentences
    - Returns both modified sentences and phraser model
    """
    logging.info("Building trigrams...")
    trigram = Phrases(sentences, min_count=5, threshold=10)
    trigram_phraser = Phraser(trigram)
    logging.info("Modifying sentences with trigrams...")
    sentences_trigram = [trigram_phraser[bigram_phraser[sentence]] for sentence in sentences]
    return sentences_trigram

def process_file(input_file, bigram_output, trigram_output, log_interval=10000)->None:
    """
    Process a single input file to generate bigrams and trigrams.

    Args:
        input_file (str): Path to the input file containing tokenized sentences.
        bigram_output (str): Path where the processed bigrams will be saved.
        trigram_output (str): Path where the processed trigrams will be saved.
        log_interval (int, optional): Number of sentences to process before logging progress. 
                                    Defaults to 10000.

    Returns:
        None

    The function:
    - Reads sentences from the input file
    - Generates bigrams using the create_bigrams function
    - Saves bigrams to the specified output file
    - Generates trigrams using the create_trigrams function
    - Saves trigrams to the specified output file
    - Logs progress at specified intervals
    - Returns early if input file cannot be read
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

def main_bigrams_trigrams()->None:
    """
    Main function to process multiple text files and generate bigrams and trigrams.

    This function:
    - Looks for input files matching a specific pattern in the paper_dataset directory
    - For each input file found:
        - Generates a corresponding bigram output file
        - Generates a corresponding trigram output file
        - Processes the input file to create bigrams and trigrams
    - Handles missing files and logs appropriate warnings
    - Uses a naming convention for output files based on the input file part number

    The function expects input files named in the format:
    'abstracts_filtered_striped_part_v1_X.txt' where X is the part number

    Output files are generated as:
    - Bigrams: 'abstracts_bigrams_part_v1_X.txt'
    - Trigrams: 'abstracts_trigrams_part_v1_X.txt'

    Returns:
        None
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