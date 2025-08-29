import os
import csv
import json
from typing import Dict, List, Tuple, Set, Iterable
from collections import defaultdict

from gensim.models import KeyedVectors
from rapidfuzz.distance import Levenshtein


# ----------------------------
# Utils
# ----------------------------
def _norm(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_").strip("_").strip()


def atomic_write_json(payload: dict, filename: str):
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    try:
        os.replace(tmp, filename)
    except Exception:
        import shutil
        shutil.move(tmp, filename)


# ----------------------------
# Load concepts from one-column CSV (header: label)
# ----------------------------
def load_concepts_csv_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSO CSV not found: {path}")
    concepts: List[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        rows = list(r)

    if not rows:
        return concepts

    # Detect header
    if len(rows[0]) == 1 and rows[0][0].strip().lower() == "label":
        data_rows = rows[1:]
    else:
        data_rows = rows

    for row in data_rows:
        if not row:
            continue
        label = (row[0] or "").strip()
        if label:
            concepts.append(label)
    return concepts


# Build structures to speed up candidate retrieval
def build_topic_structures(concepts: Iterable[str]):
    """
    Returns:
      concepts_norm: list of normalized labels (same order as originals)
      norm_to_originals: dict[norm] -> list[originals] (handles duplicates after normalization)
      bucket4: dict[first4] -> list[norms]   (coarse candidate buckets)
    """
    concepts = list(dict.fromkeys(concepts))  # de-dup while preserving order
    concepts_norm = []
    norm_to_originals: Dict[str, List[str]] = defaultdict(list)
    bucket4: Dict[str, List[str]] = defaultdict(list)

    for lab in concepts:
        n = _norm(lab)
        concepts_norm.append(n)
        norm_to_originals[n].append(lab)
        key = n[:4] if len(n) >= 4 else n
        bucket4[key].append(n)

    return concepts_norm, norm_to_originals, bucket4


def candidate_buckets(keys: str) -> Set[str]:
    """
    Produce a small set of 4-char bucket keys for a given normalized token.
    Includes prefix and sliding windows to improve recall while keeping it fast.
    """
    n = keys
    out: Set[str] = set()
    if not n:
        return out
    if len(n) <= 4:
        out.add(n)
        return out
    out.add(n[:4])
    # sliding windows of length 4 (limit to first 6 windows to keep it light)
    max_windows = min(len(n) - 3, 6)
    for i in range(max_windows):
        out.add(n[i:i+4])
    return out


def get_related_topics_name_only(
    wet2: str,
    concepts_norm: List[str],
    bucket4: Dict[str, List[str]],
) -> List[str]:
    """
    Name-based candidates only (no graph). We use buckets to avoid scanning everything.
    Returns a list of normalized topic names.
    """
    wet2_norm = _norm(wet2)
    if not wet2_norm:
        return []

    # Gather from buckets
    keys = candidate_buckets(wet2_norm)
    cand: Set[str] = set()
    for k in keys:
        cand.update(bucket4.get(k, []))

    # If buckets yield nothing (rare), fallback to small scan of all concepts with same first char
    if not cand:
        same_initial = [t for t in concepts_norm if t[:1] == wet2_norm[:1]]
        cand = set(same_initial[:2000])  # cap

    # Filter by substring containment to cut the Levenshtein work
    filtered = [t for t in cand if (wet2_norm in t or t in wet2_norm)]
    return filtered


# ----------------------------
# Matching loop with checkpointing
# ----------------------------
def match_terms(
    model: KeyedVectors,
    concepts_norm: List[str],
    bucket4: Dict[str, List[str]],
    top_n: int = 10,
    word_similarity: float = 0.7,
    min_similarity: float = 0.90,
    checkpoint_file: str = "checkpoint.json",
    checkpoint_every: int = 5000,
) -> Dict[str, list]:
    # load checkpoint
    output: Dict[str, list] = {}
    cursor = -1
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ck = json.load(f)
            if (
                isinstance(ck, dict)
                and "__results__" in ck
                and "__cursor__" in ck
                and "__vocab_size__" in ck
            ):
                output = ck.get("__results__", {}) or {}
                cursor = int(ck.get("__cursor__", -1))
    except Exception:
        pass

    key_to_index = model.key_to_index
    vocab = sorted(key_to_index.keys(), key=lambda w: key_to_index[w])
    total_vocab = len(vocab)

    try:
        for i in range(cursor + 1, total_vocab):
            wet = vocab[i]

            if i > 0 and i % checkpoint_every == 0:
                _checkpoint_save(output, i, total_vocab, checkpoint_file)

            similar_words: List[Tuple[str, float]] = [(wet, 1.0)]
            try:
                similar_words.extend(model.most_similar(wet, topn=top_n))
            except KeyError:
                pass

            matches_for_wet = []
            for wet2, sim_w in similar_words:
                if sim_w < word_similarity:
                    continue

                candidates = get_related_topics_name_only(wet2, concepts_norm, bucket4)
                wet2_norm = _norm(wet2)

                for topic_snake in candidates:
                    sim_t = Levenshtein.normalized_similarity(topic_snake, wet2_norm)
                    if sim_t >= min_similarity:
                        matches_for_wet.append(
                            {
                                "topic": topic_snake,
                                "sim_t": round(float(sim_t), 4),
                                "wet": wet2,
                                "sim_w": round(float(sim_w), 4),
                            }
                        )

            if matches_for_wet:
                output[wet] = matches_for_wet
            else:
                output.pop(wet, None)

        _checkpoint_save(output, total_vocab - 1, total_vocab, checkpoint_file)
        return output

    finally:
        try:
            last_i = i if "i" in locals() else (cursor if cursor >= 0 else -1)
            _checkpoint_save(output, last_i, total_vocab, checkpoint_file)
        except Exception:
            pass


def _checkpoint_save(results: Dict[str, list], cursor: int, vocab_size: int, filename: str):
    filtered_results = {k: v for k, v in results.items() if v}
    payload = {
        "__results__": filtered_results,
        "__cursor__": int(cursor),
        "__vocab_size__": int(vocab_size),
    }
    atomic_write_json(payload, filename)


def save_output(output: dict, filename: str):
    if isinstance(output, dict) and "__results__" in output:
        results = output.get("__results__", {}) or {}
        filtered = {k: v for k, v in results.items() if v}
        atomic_write_json({"__results__": filtered}, filename)
    else:
        filtered = {k: v for k, v in output.items() if v}
        atomic_write_json(filtered, filename)


# ----------------------------
# Main
# ----------------------------
def main():
    # >>> EDIT THESE PATHS <<<
    model_path = r'./w2v_model/264M_cp_32.bin'   # Word2Vec binary
    cso_csv_path = r'cso.csv'                  # One-column CSV with header "label"

    # Tuning
    TOP_N_SIMILAR = 10
    WORD2VEC_SIM_THRESHOLD = 0.70
    LEVENSHTEIN_THRESHOLD = 0.90
    CHECKPOINT_EVERY = 5000
    CHECKPOINT_FILE = "checkpoint_v2.json"
    OUTPUT_FILE = "cached-token-to-cso-combined_v2.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    concepts = load_concepts_csv_labels(cso_csv_path)
    if not concepts:
        raise RuntimeError("No concepts loaded from CSO CSV (check header 'label' and file content).")

    concepts_norm, norm_to_originals, bucket4 = build_topic_structures(concepts)

    results = match_terms(
        model,
        concepts_norm,
        bucket4,
        top_n=TOP_N_SIMILAR,
        word_similarity=WORD2VEC_SIM_THRESHOLD,
        min_similarity=LEVENSHTEIN_THRESHOLD,
        checkpoint_file=CHECKPOINT_FILE,
        checkpoint_every=CHECKPOINT_EVERY,
    )

    save_output(results, OUTPUT_FILE)
    print(f"Done. Results -> {OUTPUT_FILE} | Checkpoint -> {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
