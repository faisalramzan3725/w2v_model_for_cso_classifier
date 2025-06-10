import re
from collections import Counter
import csv
import os

def extract_cso_concepts(ttl_path: str, save_csv: bool = True, csv_path: str = "cso_label_counts.csv") -> list:
    """
    Extract CSO concepts from a TTL file.

    Args:
        ttl_path (str): Path to the TTL file.
        save_csv (bool): Whether to save the results to CSV.
        csv_path (str): Path for the CSV output.

    Returns:
        list: List of unique CSO concept labels.
    
    Raises:
        FileNotFoundError: If TTL file doesn't exist
        PermissionError: If file access is denied
        Exception: For other unexpected errors
    """
    try:
        # Check if file exists
        if not os.path.exists(ttl_path):
            raise FileNotFoundError(f"TTL file not found: {ttl_path}")

        # Read TTL content
        with open(ttl_path, 'r', encoding='utf-8') as file:
            ttl_data = file.read()

        # Extract labels using regex (more flexible pattern)
        pattern = r'(?:ns1:|ns0:)label\s+"(.+?)"(?:@en)?\s*[;\.]'
        labels = re.findall(pattern, ttl_data)

        if not labels:
            print("Warning: No labels found in the TTL file")
            return []

        # Count frequency
        label_counts = Counter(labels)

        # Sort by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)

        # Optionally save to CSV
        if save_csv:
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Label", "Frequency"])
                    writer.writerows(sorted_labels)
                print(f"Saved {len(sorted_labels)} labels to CSV: {csv_path}")
            except PermissionError:
                print(f"Warning: Could not save CSV file due to permission error: {csv_path}")
            except Exception as e:
                print(f"Warning: Failed to save CSV file: {str(e)}")

        print(f"Extracted {len(label_counts)} unique CSO concepts.")
        return list(label_counts.keys())

    except FileNotFoundError:
        print(f"Error: TTL file not found: {ttl_path}")
        raise
    except PermissionError:
        print(f"Error: Permission denied accessing file: {ttl_path}")
        raise
    except Exception as e:
        print(f"Error: Failed to process TTL file: {str(e)}")
        raise
