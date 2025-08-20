import os
import glob
import logging
import json
import gc
from typing import Iterator, List, Optional, Tuple
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

# =========================

# Configuration

# =========================
MODEL_NAME = "264M"
VECTOR_SIZE = 256
WINDOW_SIZE = 10
MIN_COUNT = 10
EPOCHS_PER_FILE = 5          # epochs for each file/chunk
SG = 1                       # 1 = skip-gram, 0 = CBOW
WORKERS = 4
NEGATIVE = 10                # negative sampling
HS = 0                       # hierarchical softmax off
SAMPLE = 1e-3                # subsampling
SEED = 42

DATA_DIR = "paper_dataset"

FILE_PATTERN = os.path.join(DATA_DIR, "abstracts_trigrams_part_v1_*.txt")



CHECKPOINT_DIR = "checkpoints"

MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "manifest.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)



logging.basicConfig(

    format="%(asctime)s : %(levelname)s : %(message)s",

    level=logging.INFO

)



# =========================

# Streaming iterator

# =========================

class JsonTokensPerLine:

    """

    Streams tokenized sentences from a JSONL file where each line is a JSON array of tokens.

    Example line: ["this","is","a","trigram","sentence"]

    """

    def __init__(self, file_path: str):

        self.file_path = file_path



    def __iter__(self) -> Iterator[List[str]]:

        with open(self.file_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:

                    continue

                try:

                    tokens = json.loads(line)

                    if isinstance(tokens, list):

                        yield tokens

                except json.JSONDecodeError:

                    # Skip malformed lines

                    continue



    def count(self) -> int:

        """Counts examples without loading them into memory."""

        n = 0

        with open(self.file_path, "r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():

                    continue

                try:

                    _ = json.loads(line)

                    n += 1

                except json.JSONDecodeError:

                    continue

        return n



# =========================

# Manifest helpers

# =========================

def load_manifest() -> dict:

    if os.path.exists(MANIFEST_PATH):

        try:

            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:

                return json.load(f)

        except Exception:

            pass

    return {"last_completed_index": -1, "model_path": None}



def save_manifest(last_completed_index: int, model_path: Optional[str]) -> None:

    payload = {"last_completed_index": last_completed_index, "model_path": model_path}

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:

        json.dump(payload, f)



# =========================

# Core training loop

# =========================

def train_with_checkpoints():

    files = sorted(glob.glob(FILE_PATTERN))

    if not files:

        logging.error("No trigram files found. Exiting.")

        return



    manifest = load_manifest()

    start_idx = manifest.get("last_completed_index", -1) + 1

    resume_model_path = manifest.get("model_path")



    model: Optional[Word2Vec] = None



    if resume_model_path and os.path.exists(resume_model_path):

        logging.info(f"Resuming from checkpoint: {resume_model_path}")

        model = Word2Vec.load(resume_model_path)

    else:

        logging.info("No valid checkpoint found; will initialize a new model when first chunk arrives.")



    for idx in range(start_idx, len(files)):

        file_path = files[idx]

        logging.info(f"Processing file {idx+1}/{len(files)}: {file_path}")



        # Streaming iterable for this file

        corpus = JsonTokensPerLine(file_path)

        num_examples = corpus.count()

        if num_examples == 0:

            logging.warning(f"No sentences in {file_path}. Skipping.")

            continue



        # Initialize or update vocab

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

                compute_loss=True

            )

            model.build_vocab(corpus_iterable=corpus, progress_per=max(1, num_examples//10))

        else:

            logging.info("Updating vocabulary with new file...")

            model.build_vocab(corpus_iterable=corpus, update=True, progress_per=max(1, num_examples//10))



        # Train on this file (re-instantiate iterator because it was consumed by build_vocab)

        corpus = JsonTokensPerLine(file_path)

        logging.info(f"Training on {num_examples} sentences for {EPOCHS_PER_FILE} epochs...")

        model.train(

            corpus_iterable=corpus,

            total_examples=num_examples,

            epochs=EPOCHS_PER_FILE,

            report_delay=5.0

        )

        logging.info(f"Cumulative loss: {model.get_latest_training_loss()}")



        # --- Checkpoint: save full model to resume later ---

        ckpt_model_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_cp_{idx+1}.model")

        model.save(ckpt_model_path)

        # Optional: export vectors for downstream tasks (not for resuming training)

        ckpt_vec_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_cp_{idx+1}.bin")

        model.wv.save_word2vec_format(ckpt_vec_path, binary=True)

        logging.info(f"Checkpoint saved: {ckpt_model_path} (model), {ckpt_vec_path} (vectors)")



        # Update manifest AFTER successful checkpoint

        save_manifest(last_completed_index=idx, model_path=ckpt_model_path)



        # Encourage GC between big files

        gc.collect()



    # Save final model and vectors

    final_model_path = f"{MODEL_NAME}[{VECTOR_SIZE}-{WINDOW_SIZE}]_sg.model"

    final_vecs_path = f"{MODEL_NAME}[{VECTOR_SIZE}-{WINDOW_SIZE}]_sg.bin"

    model.save(final_model_path)

    model.wv.save_word2vec_format(final_vecs_path, binary=True)

    logging.info(f"Final model saved: {final_model_path}")

    logging.info(f"Final vectors saved: {final_vecs_path}")



if __name__ == "__main__":

    train_with_checkpoints()

