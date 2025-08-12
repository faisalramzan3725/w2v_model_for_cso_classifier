import logging
import json
import os
import glob
from multiprocessing import Pool, cpu_count
from typing import List


def clean(line: str) -> List[str]:
    """
    Clean and tokenize a line of text.
    """
    s = line.lower().split(' ')
    return [w.strip(',."!?:;()\'\n') for w in s if w.strip(',."!?:;()\'\n')]


def process_file(input_file: str) -> None:
    """
    Process a text file by cleaning and tokenizing its contents in parallel.
    Output is saved to a corresponding '_striped_' file.
    """
    logging.info(f"Started processing: {input_file}")

    # Derive output filename based on part number
    base_dir = os.path.dirname(input_file)
    part = int(os.path.basename(input_file)
               .replace("abstracts_filtered_part_v1_", "")
               .replace(".txt", ""))
    output_file = os.path.join(base_dir, f"abstracts_filtered_striped_part_v1_{part}.txt")

    tokenized_samples = []
    cnt = 0
    printcounter = 0

    with open(output_file, "w+", encoding="utf-8") as file, open(input_file, encoding="utf-8") as fp:
        for line in fp:
            if printcounter == 10000:
                logging.info(f"{input_file}: Processed {cnt} lines")
                printcounter = 0
            new_line = clean(line)
            if len(tokenized_samples) < 2:
                tokenized_samples.append((line.strip(), new_line))
            file.write(json.dumps(new_line) + '\n')
            cnt += 1
            printcounter += 1

    logging.info(f"Finished processing {input_file} ({cnt} lines)")

    print(f"\n--- Test Set: Tokenized Lines for {input_file} ---")
    for i, (original, tokens) in enumerate(tokenized_samples, 1):
        print(f"Sample {i}:")
        print(f"Original: {original}")
        print(f"Tokens: {tokens}")


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
        os.path.join(base_dir, "abstracts_filtered_part_v1_1.txt"),
        os.path.join(base_dir, "abstracts_filtered_part_v1_2.txt"),
        os.path.join(base_dir, "abstracts_filtered_part_v1_3.txt"),
        os.path.join(base_dir, "abstracts_filtered_part_v1_4.txt"),
    ]

    dataset_files = get_dataset_files(base_dir, patterns)

    if not dataset_files:
        logging.warning("No files to process. Exiting.")
        return

    # Use up to number of cores or number of files
    num_processes = min(len(dataset_files), cpu_count())
    logging.info(f"Processing {len(dataset_files)} files with {num_processes} processes.")

    with Pool(processes=num_processes) as pool:
        pool.map(process_file, dataset_files)


if __name__ == "__main__":
    main()
