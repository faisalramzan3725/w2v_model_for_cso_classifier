import gc
import glob
import json
import logging
import os
from typing import Iterator, List, Optional

from gensim.models import Word2Vec

# ======================================================
# Configuration
# ======================================================
MODEL_NAME: str = "264M"

# Word2Vec hyperparameters
VECTOR_SIZE: int = 256
WINDOW_SIZE: int = 10
MIN_COUNT: int = 10
SG: int = 1                 # 1 = skip-gram, 0 = CBOW
NEGATIVE: int = 10          # negative sampling (set 0 if using HS)
HS: int = 0                 # hierarchical softmax (1 to enable, set NEGATIVE=0)
SAMPLE: float = 1e-3        # subsampling rate for frequent words
SEED: int = 42
WORKERS: int = 4

# Training schedule
EPOCHS_PER_FILE: int = 5    # epochs per file

# Data/IO
DATA_DIR: str = "paper_dataset"
FILE_PATTERN: str = os.path.join(DATA_DIR, "abstracts_trigrams_part_v1_*.txt")

CHECKPOINT_DIR: str = "checkpoints"
MANIFEST_PATH: str = os.path.join(CHECKPOINT_DIR, "manifest.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


# ======================================================
# Streaming iterator
# ======================================================
class JsonTokensPerLine:
    """
    Stream tokenized sentences from a JSONL file where each line
    is a JSON array of tokens, e.g.: ["this", "is", "a", "trigram", "sentence"].
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def __iter__(self) -> Iterator[List[str]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tokens = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(tokens, list):
                    yield tokens

    def count(self) -> int:
        """Count valid examples without loading the file into memory."""
        n = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    n += 1
                except json.JSONDecodeError:
                    continue
        return n


# ======================================================
# Manifest helpers
# ======================================================
def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logging.warning("Manifest exists but could not be read; starting fresh.")
    return {"last_completed_index": -1}


def save_manifest(last_completed_index: int) -> None:
    payload = {"last_completed_index": last_completed_index}
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)


# ======================================================
# Core training loop
# ======================================================
def train_with_checkpoints() -> None:
    files = sorted(glob.glob(FILE_PATTERN))
    if not files:
        logging.error("No trigram files found. Exiting.")
        return

    manifest = load_manifest()
    start_idx = manifest.get("last_completed_index", -1) + 1

    model: Optional[Word2Vec] = None

    for idx in range(start_idx, len(files)):
        file_path = files[idx]
        logging.info(f"Processing file {idx + 1}/{len(files)}: {file_path}")

        corpus = JsonTokensPerLine(file_path)
        num_examples = corpus.count()
        if num_examples == 0:
            logging.warning(f"No sentences in {file_path}. Skipping.")
            continue

        # Initialize or update vocabulary
        if model is None:
            logging.info("Initializing Word2Vec model...")
            model = Word2Vec(
                vector_size=VECTOR_SIZE,
                window=WINDOW_SIZE,
                min_count=MIN_COUNT,
                sg=SG,
                workers=WORKERS,
                negative=NEGATIVE,
                hs=HS,
                sample=SAMPLE,
                seed=SEED,
                compute_loss=True,
            )
            model.build_vocab(corpus_iterable=corpus, progress_per=max(1, num_examples // 10))
        else:
            logging.info("Updating vocabulary with new file...")
            model.build_vocab(corpus_iterable=corpus, update=True, progress_per=max(1, num_examples // 10))

        # Train on this file
        corpus = JsonTokensPerLine(file_path)
        logging.info(f"Training on {num_examples} sentences for {EPOCHS_PER_FILE} epochs...")
        model.train(
            corpus_iterable=corpus,
            total_examples=num_examples,
            epochs=EPOCHS_PER_FILE,
            report_delay=5.0,
        )
        logging.info(f"Cumulative loss: {model.get_latest_training_loss()}")

        # --- Checkpoint: save vectors only ---
        ckpt_vec_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_cp_{idx + 1}.bin")
        model.wv.save_word2vec_format(ckpt_vec_path, binary=True)
        logging.info(f"Checkpoint saved: {ckpt_vec_path} (vectors only)")

        # Update manifest AFTER successful checkpoint
        save_manifest(last_completed_index=idx)

        gc.collect()

    # --- Final save: vectors only ---
    if model is not None:
        final_vecs_path = f"{MODEL_NAME}[{VECTOR_SIZE}-{WINDOW_SIZE}]_sg.bin"
        model.wv.save_word2vec_format(final_vecs_path, binary=True)

        logging.info(f"Final vectors saved: {final_vecs_path}")
    else:
        logging.error("Training did not run; no vectors to save.")


if __name__ == "__main__":
    train_with_checkpoints()
