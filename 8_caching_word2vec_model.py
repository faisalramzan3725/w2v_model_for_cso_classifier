import sys
import json
from rapidfuzz.distance import Levenshtein
from gensim.models import KeyedVectors

# Add ontology path
sys.path.append(r"C:\Users\Faisal Ramzan\Desktop\kmi_project_cso\cso-reader-main\cso_reader")
from ontology import Ontology

# loading w2v model
def load_word2vec_model(path: str)->KeyedVectors:
    """Load a pre-trained Word2Vec model from a binary file.

    This function loads a Word2Vec model that has been pre-trained and saved in binary format.
    The model contains word embeddings that can be used for semantic similarity calculations.

    Args:
        path (str): File path to the binary Word2Vec model file.

    Returns:
        KeyedVectors: A loaded Word2Vec model that provides access to word vectors and similarity operations.

    Raises:
        FileNotFoundError: If the model file cannot be found at the specified path.
    """
    try:
        model = KeyedVectors.load_word2vec_format(path, binary=True)
        print(f"Loaded Word2Vec model from: {path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {path}")

# loading cso ontology
def load_ontology()->Ontology:
    """Load the Computer Science Ontology (CSO).

    This function loads the Computer Science Ontology, which is a large-scale ontology 
    of research areas in computer science. The ontology provides a hierarchical 
    classification of research topics in computer science.

    Returns:
        Ontology: An instance of the CSO Ontology class containing the loaded ontology data.

    Raises:
        RuntimeError: If there is an error loading the ontology.

    Note:
        The CSO ontology is loaded using the Ontology class from the cso_reader package.
        The ontology contains topics, relationships, and other semantic information about
        computer science research areas.
    """
    try:
        ontology = Ontology()
        print("CSO Ontology loaded.")
        return ontology
    except Exception as e:
        raise RuntimeError(f"Failed to load ontology: {e}")

# get related topic from cso w r t to wet from w2v
def get_related_topics(wet2: str, ontology: Ontology)->list:
    """Get topics from the CSO ontology related to a given term.

    This function searches the Computer Science Ontology (CSO) for topics that are related
    to the input term in two ways:
    1. Topics that contain the input term as a substring (case-insensitive)
    2. All descendant topics of the input term if it exists in the ontology

    Args:
        wet2 (str): The term to find related topics for
        ontology (Ontology): The loaded CSO ontology object

    Returns:
        list: A deduplicated list of related topic strings found in the ontology

    Example:
        >>> ontology = Ontology()
        >>> get_related_topics("machine learning", ontology)
        ['machine learning', 'deep learning', 'reinforcement learning', ...]
    """
    related_by_name = [topic for topic in ontology.topics if wet2.lower() in topic.lower()]
    descendants = ontology.get_all_descendants_of_topics([wet2]) if wet2 in ontology.topics else []
    return list(set(related_by_name + descendants))

# searching for matching terms w r t similarity
def match_terms(model, ontology, top_n=10, word_similarity=0.7, min_similarity=0.90)->dict:
    """Match terms from a Word2Vec model vocabulary with topics in the CSO ontology.

    This function processes each word in the Word2Vec model's vocabulary and attempts to match it
    with related topics from the Computer Science Ontology (CSO). It uses both semantic similarity
    from Word2Vec and string similarity (Levenshtein) to find matches.

    Args:
        model (KeyedVectors): Loaded Word2Vec model containing word embeddings
        ontology (Ontology): Loaded CSO ontology object
        top_n (int, optional): Number of similar words to retrieve from Word2Vec. Defaults to 10.
        word_similarity (float, optional): Minimum Word2Vec similarity threshold. Defaults to 0.7.
        min_similarity (float, optional): Minimum Levenshtein similarity threshold. Defaults to 0.90.

    Returns:
        dict: A dictionary where:
            - Keys are words from the Word2Vec vocabulary
            - Values are lists of dictionaries containing matched topics with similarity scores:
                {
                    "topic": str,  # The matched CSO topic
                    "sim_t": float,  # Levenshtein similarity score with the topic
                    "wet": str,  # The similar word from Word2Vec that led to this match
                    "sim_w": float  # Word2Vec similarity score
                }

    Example:
        >>> model = load_word2vec_model("model.bin")
        >>> ontology = load_ontology()
        >>> matches = match_terms(model, ontology)
        >>> print(matches["machine_learning"])
        [
            {
                "topic": "machine learning",
                "sim_t": 1.0,
                "wet": "machine_learning",
                "sim_w": 1.0
            },
            ...
        ]
    """
    output = {}
    total_vocab = len(model.key_to_index)

    for i, wet in enumerate(model.key_to_index):
        if i % 1000 == 0:
            print(f"Processing {i}/{total_vocab} | Matches so far: {len(output)}")

        output[wet] = []
        similar_words = [(wet, 1.0)] # similarity of word with itself is 1.0

        try:
            similar_words.extend(model.most_similar(wet, topn=top_n)) # Finds semantically similar words from w2v voc
        except KeyError:
            continue

        for wet2, sim_w in similar_words: # Filters out similar words with a w2v similarity below word_similarity
            if sim_w < word_similarity: # similarity checking
                continue
            
            related_topics = get_related_topics(wet2, ontology) # get related topics from cso w r t to wet2
            # For each valid similar word (wet2)
            for topic in related_topics:
                sim_t = Levenshtein.normalized_similarity(topic, wet2) # measuring similarity between wet2 and topic from cso
                if sim_t >= min_similarity:
                    output[wet].append({ # key : wet / token / word
                        "topic": topic, # matched from cso concepts
                        "sim_t": round(sim_t, 4), # string similarity between wet2 and topic from cso
                        "wet": wet2, # related similar word or semantically word from w2v
                        "sim_w": round(sim_w, 4) # w2v similarity score between original token / word /key with wet
                    })
    return output

def save_output(output: dict, filename: str):
    """
    Save the matched terms dictionary to a JSON file, excluding entries with empty match lists.

    Args:
        output (dict): Dictionary containing matched terms and their related CSO topics.
        filename (str): Path where the JSON file should be saved.
    """
    # Filter out keys with empty lists
    filtered_output = {k: v for k, v in output.items() if v}

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_output, f, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Failed to save results: {e}")


def save_output_old(output: dict, filename: str):
    """Save the matched terms dictionary to a JSON file.

    This function saves the dictionary containing matched terms and their related CSO topics
    to a JSON file with proper formatting.

    Args:
        output (dict): Dictionary containing matched terms and their related CSO topics.
            The dictionary structure should be:
            {
                "word": [
                    {
                        "topic": str,  # The matched CSO topic
                        "sim_t": float,  # Levenshtein similarity score
                        "wet": str,  # Similar word from Word2Vec
                        "sim_w": float  # Word2Vec similarity score
                    },
                    ...
                ],
                ...
            }
        filename (str): Path where the JSON file should be saved.

    Raises:
        Exception: If there is an error while writing to the file.

    Example:
        >>> results = match_terms(model, ontology)
        >>> save_output(results, "matches.json")
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Failed to save results: {e}")

def main()->None:
    """Main execution function for caching Word2Vec model matches with CSO topics.

    This function orchestrates the main workflow:
    1. Loads a pre-trained Word2Vec model from a binary file
    2. Loads the Computer Science Ontology (CSO)
    3. Matches terms between the Word2Vec vocabulary and CSO topics
    4. Saves the results to a JSON file

    The matching process uses both semantic similarity from Word2Vec and string similarity
    (Levenshtein) to find related topics. Configuration parameters control the matching thresholds
    and number of similar words to consider.

    Global Configuration:
        model_path (str): Path to the binary Word2Vec model file
        top_n_similar_words (int): Number of similar words to retrieve from Word2Vec
        word2vec_threshold (float): Minimum Word2Vec similarity threshold
        levenshtein_threshold (float): Minimum Levenshtein similarity threshold
        output_file (str): Path where results will be saved as JSON

    Returns:
        None

    Example:
        >>> main()
        # Loads model and ontology
        # Processes matches
        # Saves results to JSON file
    """

    # === CONFIGURATION ===
    model_path = '9M[256-10]_sg.bin'
    top_n_similar_words = 10
    word2vec_threshold = 0.7
    levenshtein_threshold = 0.90
    output_file = 'cached-token-to-cso-combined.json'

    # === EXECUTION ===
    model = load_word2vec_model(model_path)
    ontology = load_ontology()
    results = match_terms(
        model,
        ontology,
        top_n=top_n_similar_words,
        word_similarity=word2vec_threshold,
        min_similarity=levenshtein_threshold
    )
    save_output(results, output_file)


if __name__ == "__main__":
    main()
