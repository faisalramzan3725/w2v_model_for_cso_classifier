import re
import csv
import os
import logging

# Setup logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_cso_concepts(ttl_path: str, save_csv: bool = True, csv_path: str = "cso_label/concepts.csv") -> list:
    """
    Extracts concept labels from a TTL file containing CSO (Computer Science Ontology) data.

    This function reads a TTL file, extracts concept labels using regex pattern matching,
    and optionally saves the unique labels to a CSV file.

    Args:
        ttl_path (str): Path to the TTL file containing CSO data
        save_csv (bool, optional): Whether to save extracted labels to CSV. Defaults to True.
        csv_path (str, optional): Path where CSV file will be saved. Defaults to "cso_label/concepts.csv".

    Returns:
        list: A list of unique concept labels extracted from the TTL file.

    Raises:
        FileNotFoundError: If the specified TTL file does not exist
        Exception: For other errors during file processing or CSV saving

    Example:
        >>> labels = extract_cso_concepts("path/to/CSO.ttl")
        >>> print(len(labels))
        14000
    """

    try:
        if not os.path.exists(ttl_path):
            raise FileNotFoundError(f"TTL file not found: {ttl_path}")

        with open(ttl_path, 'r', encoding='utf-8') as file:
            ttl_data = file.read()

        # Regex to extract label strings
        pattern = r'(?:ns1:|ns0:)label\s+"(.+?)"(?:@en)?\s*[;\.]'
        labels = re.findall(pattern, ttl_data)

        if not labels:
            print("Warning: No labels found in the TTL file")
            return []

        unique_labels = list(set(labels))
        print(f"Extracted {len(unique_labels)} unique CSO concepts.") # Number of cso concepts

        if save_csv:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Label"])
                    for label in unique_labels:
                        writer.writerow([label])
                print(f"Saved labels to CSV: {csv_path}")
            except PermissionError:
                print(f"Warning: Could not save CSV file due to permission error: {csv_path}")
            except Exception as e:
                print(f"Warning: Failed to save CSV file: {str(e)}")

        return unique_labels

    except Exception as e:
        logger.error(f"Error extracting CSO concepts: {e}")
        raise

if __name__ == "__main__":
    ttl_path = "cso_label/CSO/CSO.3.4.1.ttl"  # Specified TTL file path
    labels = extract_cso_concepts(ttl_path)
    print(labels[:5])  # Show first 5 labels
    