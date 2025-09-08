import logging
import json
import os
from gensim.models.phrases import Phrases, Phraser
import glob
from typing import List, Tuple, Iterator
import math
import gc

"""
This code processes text files containing abstracts to create bigrams and trigrams using Gensim's Phrases model.
It splits each input file into 8 partitions for memory-efficient processing.

Key components:
- Reads large text files in chunks (8 partitions) to avoid memory issues
- Creates bigrams and trigrams from text using Gensim's Phrases
- Processes 4 input files (parts 1-4) sequentially
- Saves processed bigrams and trigrams to separate output files for each partition

The partitioning works as follows:
1. Counts total lines in input file
2. Divides file into 8 equal chunks
3. Processes each chunk separately to create bigrams/trigrams
4. Saves results for each partition in separate output files

For example, if a file has 800 lines:
- Each partition will process 100 lines (800/8)
- Creates 8 output files each for bigrams and trigrams
- Partition 1: lines 0-99
- Partition 2: lines 100-199
And so on...
"""

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


def write_sentences_streaming(sentences: List[List[str]], output_file: str) -> None:
    """
    Write sentences to file one by one to avoid memory buildup during JSON serialization.
    """
    logging.info(f"Writing {len(sentences)} sentences to {output_file}")
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            for i, sentence in enumerate(sentences):
                try:
                    # Write each sentence individually and flush periodically
                    file.write(json.dumps(sentence) + '\n')
                    if i % 1000 == 0:  # Flush every 1000 lines
                        file.flush()
                except (MemoryError, OverflowError) as e:
                    logging.error(f"Memory error writing sentence {i}: {e}")
                    # Try to continue with remaining sentences
                    continue
        logging.info(f"Successfully wrote to {output_file}")
    except Exception as e:
        logging.error(f"Error writing to {output_file}: {e}")
        raise


def process_chunk(sentences_chunk: List[List[str]], base_dir: str, file_part: int, partition_num: int) -> None:
    """
    Process a single chunk to create bigrams and trigrams with improved memory management.
    """
    logging.info(f"Processing file part {file_part}, partition {partition_num}")
    
    bigram_output = os.path.join(base_dir, f"abstracts_bigrams_part_v1_{file_part}_{partition_num}.txt")
    trigram_output = os.path.join(base_dir, f"abstracts_trigrams_part_v1_{file_part}_{partition_num}.txt")

    try:
        # Create and save bigrams
        sentences_bigram, bigram_phraser = create_bigrams(sentences_chunk)
        
        # Write bigrams with streaming approach
        write_sentences_streaming(sentences_bigram, bigram_output)
        
        # Force garbage collection after bigrams
        del sentences_chunk  # Clear original chunk early
        gc.collect()
        
        # Create and save trigrams
        sentences_trigram = create_trigrams(sentences_bigram, bigram_phraser)
        
        # Clear bigrams before writing trigrams
        del sentences_bigram
        gc.collect()
        
        # Write trigrams with streaming approach
        write_sentences_streaming(sentences_trigram, trigram_output)
        
        logging.info(f"Successfully processed partition {partition_num}")
        
    except MemoryError as e:
        logging.error(f"Memory error in partition {partition_num}: {e}")
        raise
    finally:
        # Cleanup
        try:
            del sentences_trigram, bigram_phraser
        except:
            pass
        gc.collect()
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
    
    # Increase number of partitions for better memory management
    number_of_partitions = 8  # Increased from 8 to 16
    
    # Calculate chunk size
    chunk_size = math.ceil(total_lines / number_of_partitions)
    logging.info(f"Processing {total_lines} lines in {number_of_partitions} partitions of ~{chunk_size} lines each")

    # Process each chunk without loading the entire file
    for partition_num in range(1, number_of_partitions + 1):
        start_line = (partition_num - 1) * chunk_size
        
        # Don't read beyond the file
        if start_line >= total_lines:
            break
            
        # Adjust chunk size for the last partition
        current_chunk_size = min(chunk_size, total_lines - start_line)
        
        logging.info(f"Processing partition {partition_num}/{number_of_partitions} for file part {file_part}")
        
        try:
            # Read only this chunk
            sentences_chunk = read_file_chunk(input_file, start_line, current_chunk_size)
            
            if sentences_chunk:
                process_chunk(sentences_chunk, base_dir, file_part, partition_num)
            else:
                logging.warning(f"No sentences found in partition {partition_num}")
                
        except MemoryError as e:
            logging.error(f"Memory error in partition {partition_num}: {e}")
            # Force garbage collection and continue
            gc.collect()
            continue
        except Exception as e:
            logging.error(f"Unexpected error in partition {partition_num}: {e}")
            continue

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
        try:
            process_file(file_path)
            # Force garbage collection between files
            gc.collect()
        except MemoryError as e:
            logging.error(f"Memory error processing {file_path}: {e}")
            continue
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    logging.info("All files processed successfully!")


if __name__ == "__main__":
    main()