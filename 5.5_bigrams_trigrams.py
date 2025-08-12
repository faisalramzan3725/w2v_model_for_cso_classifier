import logging
import json
import os
from gensim.models.phrases import Phrases, Phraser
import glob
from multiprocessing import Pool, cpu_count
from typing import List, Tuple


def read(file_path: str) -> List[List[str]]:
    """
    Read and parse sentences from a JSON file.
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


def create_bigrams(sentences: List[List[str]]) -> Tuple[List[List[str]], Phraser]:
    """
    Create bigrams from input sentences using Gensim's Phrases model.
    """
    logging.info("Building bigrams...")
    bigram = Phrases(sentences, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    logging.info("Modifying sentences with bigrams...")
    sentences_bigram = [bigram_phraser[sentence] for sentence in sentences]
    return sentences_bigram, bigram_phraser


def create_trigrams(sentences: List[List[str]], bigram_phraser: Phraser) -> List[List[str]]:
    """
    Create trigrams from input sentences using Gensim's Phrases model.
    """
    logging.info("Building trigrams...")
    trigram = Phrases(sentences, min_count=5, threshold=10)
    trigram_phraser = Phraser(trigram)
    logging.info("Modifying sentences with trigrams...")
    sentences_trigram = [trigram_phraser[bigram_phraser[sentence]] for sentence in sentences]
    return sentences_trigram


def process_file(input_file: str) -> None:
    """
    Process a single file to create bigrams and trigrams.
    """
    logging.info(f"Started processing: {input_file}")

    base_dir = os.path.dirname(input_file)
    part = int(os.path.basename(input_file)
               .replace("abstracts_filtered_striped_part_v1_", "")
               .replace(".txt", ""))

    bigram_output = os.path.join(base_dir, f"abstracts_bigrams_part_v1_{part}.txt")
    trigram_output = os.path.join(base_dir, f"abstracts_trigrams_part_v1_{part}.txt")

    sentences = read(input_file)
    if not sentences:
        return

    # Create and save bigrams
    sentences_bigram, bigram_phraser = create_bigrams(sentences)
    with open(bigram_output, "w+", encoding="utf-8") as file:
        for s in sentences_bigram:
            file.write(json.dumps(s) + '\n')

    # Create and save trigrams
    sentences_trigram = create_trigrams(sentences_bigram, bigram_phraser)
    with open(trigram_output, "w+", encoding="utf-8") as file:
        for s in sentences_trigram:
            file.write(json.dumps(s) + '\n')

    logging.info(f"Finished processing part {part}")


def get_dataset_files(base_dir: str, patterns: List[str]) -> List[str]:
    """
    Collect all matching files from specified wildcard patterns.
    """
    all_files = []
    for pattern in patterns:
        matched = glob.glob(pattern)
        if not matched:
            logging.warning(f"No files found for pattern: {pattern}")
        all_files.extend(matched)
    return all_files


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    base_dir = "paper_dataset"
    patterns = [
        os.path.join(base_dir, "abstracts_filtered_striped_part_v1_1.txt"),
        os.path.join(base_dir, "abstracts_filtered_striped_part_v1_2.txt"),
        os.path.join(base_dir, "abstracts_filtered_striped_part_v1_3.txt"),
        os.path.join(base_dir, "abstracts_filtered_striped_part_v1_4.txt"),
    ]

    dataset_files = get_dataset_files(base_dir, patterns)

    if not dataset_files:
        logging.warning("No files to process. Exiting.")
        return

    num_processes = min(len(dataset_files), cpu_count())
    logging.info(f"Processing {len(dataset_files)} files with {num_processes} processes.")

    with Pool(processes=num_processes) as pool:
        pool.map(process_file, dataset_files)


if __name__ == "__main__":
    main()
