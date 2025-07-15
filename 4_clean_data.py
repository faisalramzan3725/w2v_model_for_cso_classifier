import logging
from pickle import NONE
import pandas as pd
import re
import os
import glob

def get_space_topics()->tuple[list[str],dict[str,str]]:
    """
    Retrieves space-related topics and their substitutions from a CSO (Computer Science Ontology) file.

    This function reads a CSV file containing CSO concepts and creates:
    1. A list of unique topic labels in lowercase
    2. A dictionary mapping multi-word topics to their underscore-separated versions

    Returns:
        tuple[list[str], dict[str,str]]: A tuple containing:
            - list[str]: Sorted list of unique topic labels
            - dict[str,str]: Dictionary mapping original topics to underscore versions
                            (only for topics containing spaces)

    Example:
        topics, substitutions = get_space_topics()
        # topics = ['machine learning', 'ai', ...]
        # substitutions = {'machine learning': 'machine_learning', ...}
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

def process_line(line, topics, substitutions)->tuple[str,list[str]]:
    """
    Process a line of text by cleaning and replacing space-related topics.

    This function performs several text processing steps:
    1. Normalizes Unicode characters and special characters
    2. Replaces space-related topics with their underscore versions
    3. Handles malformed newline artifacts

    Args:
        line (str): The input text line to process
        topics (list[str]): List of space-related topics to look for
        substitutions (dict[str,str]): Dictionary mapping topics to their underscore versions

    Returns:
        tuple[str,list[str]]: A tuple containing:
            - str: The processed text line
            - list[str]: List of topics that were replaced in the line

    Example:
        line = "Research on machine learning and AI"
        topics = ["machine learning", "ai"]
        subs = {"machine learning": "machine_learning"}
        new_line, changed = process_line(line, topics, subs)
        # new_line = "Research on machine_learning and AI"
        # changed = ["machine learning"]
    """
    # Step 1: Normalize Unicode escapes and special characters
    line = line.encode().decode('utf-8', errors='ignore')  # Handle invalid Unicode
    line = re.sub(r'\\u[0-9a-fA-F]{4}', lambda m: chr(int(m.group(0)[2:], 16)), line)  # Convert \uXXXX to characters
    line = re.sub(r'[^\w\s-]', ' ', line)  # Replace non-alphanumeric/non-space/- with space
    line = re.sub(r'\s+', ' ', line.strip())  # Normalize multiple spaces/newlines

    # Step 2: Replace CSO topics
    to_change = []
    for topic in substitutions.keys():
        if topic in line.lower():
            to_change.append(topic)
    for topic in to_change:
        pattern = re.compile(re.escape(topic), re.IGNORECASE)
        line = pattern.sub(substitutions[topic], line)

    # Step 3: Handle malformed newline artifacts
    line = re.sub(r'#r##n##r##n#', '\n', line)  # Replace custom newline markers with proper newline
    line = line.strip()  # Remove leading/trailing whitespace

    return line, to_change

def process_file(input_file, output_file)->None:
    """
    Process a text file by replacing space-related topics with their underscore versions.

    This function reads an input file line by line, processes each line using the
    process_line() function, and writes the processed lines to an output file.
    It also collects sample changes to demonstrate the transformations performed.

    Args:
        input_file (str): Path to the input text file to process
        output_file (str): Path where the processed output will be written

    Returns:
        None

    Example:
        process_file("input.txt", "output.txt")
        # Processes input.txt and writes cleaned text to output.txt
        # Also prints sample transformations showing before/after changes
    """
    logging.info(f"Processing {input_file}")
    topics, substitutions = get_space_topics()

    with open(output_file, "w+", encoding="utf-8") as outfile, open(input_file, encoding="utf-8") as infile:
        cnt = 0
        sample_count = 0
        before_after_samples = []

        for line in infile:
            new_line, changed_topics = process_line(line, topics, substitutions)
            outfile.write(new_line + '\n')  # Ensure newline after each line
            cnt += 1

            if changed_topics and sample_count < 2:
                before_after_samples.append((line.strip(), new_line, changed_topics))
                sample_count += 1

        logging.info(f"Processed {cnt} lines from {input_file}")

    print(f"\n--- Sample Changes in {input_file} ---")
    for i, (before, after, topics) in enumerate(before_after_samples, 1):
        print(f"\nSample {i}:")
        print(f"Before: {before}")
        print(f"After:  {after}")
        print(f"Concepts Replaced: {', '.join(topics)}")

def main_glue()->None:
    """
    Main function to process multiple text files containing paper abstracts.

    This function:
    1. Sets up logging configuration
    2. Finds all abstract partition files matching the pattern "abstracts_part_v1_*.txt"
    3. For each partition file:
        - Creates a corresponding output filename
        - Processes the file using process_file() if it exists
        - Logs warnings for missing files

    The function processes large datasets split into multiple files, cleaning and
    standardizing the text while replacing space-separated topics with underscore versions.

    Returns:
        None

    Example:
        main_glue()
        # Processes all partition files in paper_dataset directory
        # Creates corresponding filtered output files
        # Logs progress and any warnings
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    base_dir = "paper_dataset"
    pattern = os.path.join(base_dir, "abstracts_part_v1_*.txt")
    partition_files = glob.glob(pattern)
    
    if not partition_files:
        logging.warning(f"No files found matching pattern {pattern}")
        return

    for input_file in partition_files:
        part = int(os.path.basename(input_file).replace("abstracts_part_v1_", "").replace(".txt", ""))
        output_file = os.path.join(base_dir, f"abstracts_filtered_part_v1_{part}.txt")
        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            logging.warning(f"{input_file} not found. Skipping.")

if __name__ == "__main__":
    main_glue()