import logging
import json
import os
from gensim.models.phrases import Phrases, Phraser
import glob
from typing import List, Tuple, Iterator
import math


def count_lines_in_file(file_path: str) -> int:
    """
    Count total number of lines in a file without loading it entirely into memory.
    """
    logging.info(f"Counting lines in {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        logging.info(f"Total lines in {file_path}: {line_count}")
        return line_count
    except FileNotFoundError:
        logging.warning(f"{file_path} not found.")
        return 0


def read_file_chunk(file_path: str, start_line: int, chunk_size: int) -> List[List[str]]:
    """
    Read a specific chunk of lines from a JSON file without loading the entire file.
    """
    logging.info(f"Reading chunk from {file_path}: lines {start_line} to {start_line + chunk_size - 1}")
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Skip to start_line
            for _ in range(start_line):
                f.readline()
            
            # Read chunk_size lines
            for _ in range(chunk_size):
                line = f.readline()
                if not line:  # End of file
                    break
                try:
                    sentences.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Error decoding JSON line: {e}")
                    continue
                    
    except FileNotFoundError:
        logging.warning(f"{file_path} not found. Skipping.")
        return []
    
    logging.info(f"Loaded {len(sentences)} sentences from chunk")
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


def process_chunk(sentences_chunk: List[List[str]], base_dir: str, file_part: int, partition_num: int) -> None:
    """
    Process a single chunk to create bigrams and trigrams.
    """
    logging.info(f"Processing file part {file_part}, partition {partition_num}")
    
    bigram_output = os.path.join(base_dir, f"abstracts_bigrams_part_v1_{file_part}_{partition_num}.txt")
    trigram_output = os.path.join(base_dir, f"abstracts_trigrams_part_v1_{file_part}_{partition_num}.txt")

    # Create and save bigrams
    sentences_bigram, bigram_phraser = create_bigrams(sentences_chunk)
    with open(bigram_output, "w+", encoding="utf-8") as file:
        for s in sentences_bigram:
            file.write(json.dumps(s) + '\n')
    logging.info(f"Saved bigrams to {bigram_output}")

    # Create and save trigrams
    sentences_trigram = create_trigrams(sentences_bigram, bigram_phraser)
    with open(trigram_output, "w+", encoding="utf-8") as file:
        for s in sentences_trigram:
            file.write(json.dumps(s) + '\n')
    logging.info(f"Saved trigrams to {trigram_output}")

    # Clear memory
    del sentences_chunk, sentences_bigram, sentences_trigram, bigram_phraser
    logging.info(f"Finished processing partition {partition_num}")


def process_file(input_file: str) -> None:
    """
    Process a single file by reading it in chunks and creating bigrams/trigrams for each chunk.
    """
    logging.info(f"Started processing: {input_file}")

    base_dir = os.path.dirname(input_file)
    file_part = int(os.path.basename(input_file)
                   .replace("abstracts_filtered_striped_part_v1_", "")
                   .replace(".txt", ""))

    # Count total lines first
    total_lines = count_lines_in_file(input_file)
    if total_lines == 0:
        return
    number_of_partitions = 8
    # Calculate chunk size for 4 partitions
    chunk_size = math.ceil(total_lines / number_of_partitions)

    
    # Process each chunk without loading the entire file
    for partition_num in range(1, 9):  # partitions 1-4
        start_line = (partition_num - 1) * chunk_size
        
        # Don't read beyond the file
        if start_line >= total_lines:
            break
            
        # Adjust chunk size for the last partition
        current_chunk_size = min(chunk_size, total_lines - start_line)
        
        logging.info(f"Processing partition {partition_num}/{number_of_partitions} for file part {file_part}")
        
        # Read only this chunk
        sentences_chunk = read_file_chunk(input_file, start_line, current_chunk_size)
        
        if sentences_chunk:
            process_chunk(sentences_chunk, base_dir, file_part, partition_num)
        else:
            logging.warning(f"No sentences found in partition {partition_num}")

    logging.info(f"Finished processing file part {file_part}")


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
    return sorted(all_files)  # Sort to ensure consistent processing order


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

    logging.info(f"Processing {len(dataset_files)} files sequentially with memory-efficient chunked reading.")

    # Process each file one at a time
    for file_path in dataset_files:
        process_file(file_path)

    logging.info("All files processed successfully!")


if __name__ == "__main__":
    main()