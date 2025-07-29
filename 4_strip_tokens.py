import logging
import json
import os
import glob

def clean(line:str)->list[str]:
    """
    Clean and tokenize a line of text.

    This function takes a string input, converts it to lowercase, and splits it into tokens.
    The tokens are then stripped of common punctuation marks.

    Args:
        line (str): The input text line to be cleaned and tokenized.

    Returns:
        list[str]: A list of cleaned and tokenized words from the input line.
    """
    s = line.lower().split(' ')
    return [w.strip(',."!?:;()\'\n') for w in s if w.strip(',."!?:;()\'\n')]

def process_file(input_file, output_file)->None:
    """
    Process a text file by cleaning and tokenizing its contents.

    This function reads an input file line by line, cleans and tokenizes each line,
    and writes the processed tokens to an output file in JSON format. It also keeps
    track of the first two lines as samples for testing purposes.

    Args:
        input_file (str): Path to the input text file to be processed.
        output_file (str): Path where the processed output will be written.

    Returns:
        None

    The function provides progress updates through logging and prints sample
    tokenization results for the first two lines of the input file.
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(f"Processing {input_file}")
    
    tokenized_samples = []
    cnt = 0
    printcounter = 0
    
    with open(output_file, "w+", encoding="utf-8") as file, open(input_file, encoding="utf-8") as fp:
        for line in fp:
            if printcounter == 10000:
                logging.info(f"Processed {cnt} lines")
                printcounter = 0
            new_line = clean(line)
            if len(tokenized_samples) < 2:
                tokenized_samples.append((line.strip(), new_line))
            file.write(json.dumps(new_line) + '\n')
            cnt += 1
            printcounter += 1
    
    logging.info(f"Lines processed: {cnt}")
    
    print(f"\n--- Test Set: Tokenized Lines for {input_file} ---")
    for i, (original, tokens) in enumerate(tokenized_samples, 1):
        print(f"Sample {i}:")
        print(f"Original: {original}")
        print(f"Tokens: {tokens}")
    
    logging.info(f"Finished writing to {output_file}")

def main_strip()->None:
    """
    Main function to process and tokenize text files.

    This function:
    1. Searches for partition files in the 'paper_dataset' directory matching the pattern
       'abstracts_filtered_part_v1_*.txt'
    2. For each found partition file:
        - Creates a corresponding output filename with '_striped_' added
        - Processes the file by cleaning and tokenizing its contents
        - Saves the processed tokens to a new file

    The function uses glob to find matching files and handles cases where:
    - No matching files are found
    - Input files are missing
    
    Returns:
        None
    """
    # Dynamically find all partition files
    base_dir = "paper_dataset"
    pattern = os.path.join(base_dir, "abstracts_filtered_part_v1_*.txt")
    partition_files = glob.glob(pattern)
    
    if not partition_files:
        logging.warning(f"No files found matching pattern {pattern}")
        return

    # Process each found partition file
    for input_file in partition_files:
        # Extract the part number from the filename
        part = int(os.path.basename(input_file).replace("abstracts_filtered_part_v1_", "").replace(".txt", ""))
        output_file = os.path.join(base_dir, f"abstracts_filtered_striped_part_v1_{part}.txt")
        
        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            logging.warning(f"{input_file} not found. Skipping.")

if __name__ == "__main__":
    main_strip()