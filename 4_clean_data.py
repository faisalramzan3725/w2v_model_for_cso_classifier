import logging
import pandas as pd
import re
import os
import glob

def get_space_topics():
    """
    Load CSO concepts and prepare replacements for space-containing topics.
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

def process_line(line, topics, substitutions):
    """
    Normalize text, replace CSO topics with underscore versions, and clean artifacts.
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

def process_file(input_file, output_file):
    """
    Process a single text file and write the cleaned version.
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

def main_glue():
    """
    Main function to process all abstract parts dynamically.
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