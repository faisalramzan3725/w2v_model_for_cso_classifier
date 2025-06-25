# paper_dataset.py
import logging
import pandas as pd
import os

# Setup logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def process_papers_to_txt(file, df):
    for _, row in df.iterrows():
        try:
            title = str(row['title']).replace('\n', ' ').strip()
            file.write(f"{title}\r\n")  # Write title on its own line
            abstract = str(row['abstract']).replace('\n', ' ').strip()
            file.write(f"{abstract}\r\n")  # Write abstract on the next line
        except Exception as e:
            logger.info(f"Error processing paper {row.get('id', 'unknown')}: {e}")

def main_load_papers():
    logger.info('Started processing papers')

    csv_path = os.path.join('paper_dataset', 'paper_dataset_1k.csv')
    
    # Load CSV
    try:
        paper_df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(paper_df)} papers from {csv_path}")
    except FileNotFoundError:
        logger.error(f"{csv_path} not found")
        raise FileNotFoundError(f"{csv_path} not found")

    # Filter columns
    filtered_df = paper_df[['title', 'abstract']].dropna().reset_index(drop=True)
    
    # Filter by language if applicable
    #if 'language' in paper_df.columns:
    #    filtered_df = paper_df[paper_df['language'] == 'en'][['title', 'abstract']].dropna().reset_index(drop=True)
    #    logger.info("Filtered for English papers")

    logger.info(f"Got {len(filtered_df)} papers after filtering")

    # Save to TXT
    txt_path = "paper_dataset/abstracts_v1.txt"
    with open(txt_path, "w+", encoding="utf-8") as txt_file:
        process_papers_to_txt(txt_file, filtered_df)
    logger.info(f"Saved output to {txt_path}")

    # Save to CSV
    csv_output_path = "paper_dataset/abstracts_v1.csv"
    filtered_df.to_csv(csv_output_path, index=False)
    logger.info(f"Saved filtered data to {csv_output_path}")

    logger.info('Finished processing papers')

if __name__ == "__main__":
    main_load_papers()