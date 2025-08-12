import os
import glob
import logging
import simplejson as json
from gensim.models import Word2Vec

# === Configuration ===
MODEL_NAME = '264M'
VECTOR_SIZE = 256
WINDOW_SIZE = 10
MIN_COUNT = 10
EPOCHS = 5
CHECKPOINT_EVERY = 1  # save after every N files
DATA_DIR = 'paper_dataset'
FILE_PATTERN = os.path.join(DATA_DIR, "abstracts_trigrams_part_v1_*.txt")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Logging ===
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_sentences(file_path):
    """
    Load tokenized sentences from a trigram file.
    """
    logging.info(f"Reading {file_path}")
    sentences = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sentences.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line in {file_path}: {e}")
    except FileNotFoundError:
        logging.warning(f"{file_path} not found. Skipping.")
        return []
    logging.info(f"Loaded {len(sentences)} sentences from {file_path}")
    return sentences


def train_with_checkpoints():
    """
    Incrementally trains a Word2Vec model on large datasets using file-by-file
    loading with periodic checkpoints.
    """
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        logging.error("No trigram files found. Exiting.")
        return

    model = None

    for idx, file_path in enumerate(files):
        logging.info(f"Processing file {idx+1}/{len(files)}: {file_path}")
        sentences = read_sentences(file_path)

        if not sentences:
            logging.warning(f"No sentences in {file_path}. Skipping.")
            continue

        # First chunk — initialize model
        if model is None:
            logging.info("Initializing Word2Vec model...")
            model = Word2Vec(
                vector_size=VECTOR_SIZE,
                window=WINDOW_SIZE,
                min_count=MIN_COUNT,
                sg=1,
                workers=4
            )
            model.build_vocab(sentences)
        else:
            logging.info("Updating vocabulary...")
            model.build_vocab(sentences, update=True)

        logging.info("Training on current chunk...")
        model.train(sentences, total_examples=len(sentences), epochs=EPOCHS)

        # Save periodic checkpoints
        if (idx + 1) % CHECKPOINT_EVERY == 0 or (idx + 1) == len(files):
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"{MODEL_NAME}_cp_{idx+1}.bin"
            )
            model.wv.save_word2vec_format(checkpoint_path, binary=True)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

            # Optional: Save full model config so you can reload later
            config_path = os.path.join(
                CHECKPOINT_DIR,
                f"{MODEL_NAME}_cp_{idx+1}.config.txt"
            )
            with open(config_path, "w", encoding="utf-8") as cfg:
                cfg.write(f"Vector Size: {VECTOR_SIZE}\n")
                cfg.write(f"Window Size: {WINDOW_SIZE}\n")
                cfg.write(f"Min Count: {MIN_COUNT}\n")
                cfg.write(f"Epochs per chunk: {EPOCHS}\n")
                cfg.write(f"Algorithm: Skip-gram (sg=1)\n")
            logging.info(f"Config saved: {config_path}")

    # Save final model
    final_model_path = f"{MODEL_NAME}[{VECTOR_SIZE}-{WINDOW_SIZE}]_sg.bin"
    model.wv.save_word2vec_format(final_model_path, binary=True)
    logging.info(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    train_with_checkpoints()
