import logging
import os
import math

# Assuming file_len is defined elsewhere; if not, define it
def file_len(file_path):
    """Count the number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def main_split():
    # Setup logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Input file path
    input_file_path = os.path.join("paper_dataset", "abstracts_v1.txt")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"{input_file_path} not found")

    # Count total lines
    total_lines = file_len(input_file_path)
    total_pairs = total_lines // 2  # Each pair is title + abstract (2 lines)

    # Define affordance: max lines per part (tune based on system memory)
    affordance = 25000  # Maximum lines per partition; adjust based on dataset and memory

    # Compute number of parts based on affordance and total lines
    num_parts = max(4, math.ceil(total_lines / affordance))  # Ensure at least 4 parts
    lines_per_part = math.ceil(total_lines / num_parts)  # Recalculate lines per part
    pairs_per_part = math.ceil(total_pairs / num_parts)  # Pairs per part for pair-wise splitting

    logger.info(f"Total lines: {total_lines}, Total pairs: {total_pairs}")
    logger.info(f"Affordance: {affordance}")
    logger.info(f"Calculated number of parts: {num_parts}")
    logger.info(f"Lines per part: {lines_per_part}, Pairs per part: {pairs_per_part}")

    # Process and write files incrementally
    with open(input_file_path, encoding='utf-8') as infile:
        for part in range(num_parts):
            output_file = os.path.join("paper_dataset", f"abstracts_part_v1_{part+1}.txt")
            start_line = part * lines_per_part
            end_line = min((part + 1) * lines_per_part, total_lines)

            # Read only the relevant chunk
            lines = []
            infile.seek(0)  # Reset file pointer for each partition (inefficient; see note below)
            for i, line in enumerate(infile):
                if start_line <= i < end_line:
                    lines.append(line.strip())
                elif i >= end_line:
                    break

            # Process pairs within the chunk
            with open(output_file, "w+", encoding='utf-8') as outfile:
                preview_lines = []
                for i in range(0, len(lines), 2):  # Iterate over pairs
                    if i // 2 >= pairs_per_part and part == num_parts - 1:  # Last part may have fewer pairs
                        break
                    if i + 1 < len(lines):  # Ensure we have a pair
                        title = lines[i]
                        abstract = lines[i + 1]
                        outfile.write(f"{title}{abstract}\n")  # Write title and abstract consecutively
                        if len(preview_lines) < 2:  # Preview first two lines (one pair)
                            preview_lines.append(f"{title}{abstract}")

            # Print preview for this partition
            print(f"\n--- Test Set: Split Part {part+1} ---")
            for j, line in enumerate(preview_lines, 1):
                print(f"Line {j}: {line}")

            logger.info(f"Partitioned {len(lines)} lines to {output_file}")

    logger.info("Partitioning complete.")

if __name__ == "__main__":
    main_split()
