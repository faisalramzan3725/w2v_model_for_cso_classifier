import os
import glob
import re
import pandas as pd
import logging
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count


def get_space_topics() -> Tuple[List[str], Dict[str, str]]:
    """
    Load CSO topics and prepare substitutions dictionary.
    """
    cso_df = pd.read_csv('cso_label/concepts.csv')
    substitutions = {}
    topics = sorted(
        cso_df['Label'].dropna().astype(str).str.lower().unique(),
        key=len,
        reverse=True
    )
    for topic in topics:
        if ' ' in topic:
            substitutions[topic] = topic.replace(" ", "_")
    return topics, substitutions


def process_line(line: str, topics: List[str], substitutions: Dict[str, str]) -> Tuple[str, List[str]]:
    line = line.encode().decode('utf-8', errors='ignore')
    line = re.sub(r'\\u[0-9a-fA-F]{4}', lambda m: chr(int(m.group(0)[2:], 16)), line)
    line = re.sub(r'[^\w\s-]', ' ', line)
    line = re.sub(r'\s+', ' ', line.strip())

    to_change = [topic for topic in substitutions if topic in line.lower()]
    for topic in to_change:
        pattern = re.compile(re.escape(topic), re.IGNORECASE)
        line = pattern.sub(substitutions[topic], line)

    line = re.sub(r'#r##n##r##n#', '\n', line)
    return line.strip(), to_change


def process_file(input_file: str):
    """
    Process a single file: read, clean, and write to output.
    This function will be executed in parallel.
    """
    logging.info(f"Started processing: {input_file}")
    output_file = input_file.replace("abstracts_part", "abstracts_filtered_part")
    topics, substitutions = get_space_topics()

    before_after_samples = []
    sample_count = 0
    cnt = 0

    with open(input_file, encoding='utf-8') as infile, open(output_file, "w+", encoding="utf-8") as outfile:
        for line in infile:
            new_line, changed_topics = process_line(line, topics, substitutions)
            outfile.write(new_line + '\n')
            cnt += 1
            if changed_topics and sample_count < 2:
                before_after_samples.append((line.strip(), new_line, changed_topics))
                sample_count += 1

    logging.info(f"Finished processing {input_file} ({cnt} lines)")

    print(f"\n--- Sample Changes in {input_file} ---")
    for i, (before, after, topics) in enumerate(before_after_samples, 1):
        print(f"\nSample {i}:")
        print(f"Before: {before}")
        print(f"After:  {after}")
        print(f"Concepts Replaced: {', '.join(topics)}")


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
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    base_dir = "paper_dataset"
    patterns = [
        os.path.join(base_dir, "abstracts_part_v1_2.txt"),
        os.path.join(base_dir, "abstracts_part_v1_3.txt"),
        os.path.join(base_dir, "abstracts_part_v1_4.txt"),
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
