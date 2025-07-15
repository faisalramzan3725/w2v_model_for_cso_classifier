import logging
import os
import importlib.util
import sys
import time

# Setup logging with custom time format
logging.basicConfig(
    filename='myapp.log',
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Define the time format here
)
logger = logging.getLogger(__name__)

def import_module(module_name, file_path):
    """Import a Python module from a file path dynamically.

    This function imports a Python module from a specified file path using the importlib
    functionality. It allows for dynamic loading of modules at runtime.

    Args:
        module_name (str): The name to give to the imported module
        file_path (str): The file system path to the Python module to import

    Returns:
        module: The imported Python module object

    Raises:
        FileNotFoundError: If the specified module file path does not exist
    """
    if not os.path.exists(file_path):
        logger.error(f"Module file {file_path} not found")
        raise FileNotFoundError(f"Module file {file_path} not found")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_pipeline()->None:
    """Execute a multi-step data processing pipeline.

    This function orchestrates a sequence of data processing steps for analyzing academic papers:
    1. Extract CSO (Computer Science Ontology) concepts
    2. Process paper dataset
    3. Partition paper abstracts
    4. Clean the data
    5. Strip tokens
    6. Generate bigrams and trigrams
    7. Train Word2Vec model
    8. Cache Word2Vec similarities with CSO ontology

    The function:
    - Imports and executes each processing step module dynamically
    - Measures and logs execution time for each step
    - Handles errors for individual steps without stopping the pipeline
    - Identifies and logs the most time-consuming step
    - Maintains detailed logging throughout execution

    Returns:
        None

    Raises:
        No exceptions are raised as each step handles its own exceptions
    """
    logger.info("Starting pipeline execution")
    execution_times = {}  # Dictionary to store execution times for each step

    # Step 1: Extract CSO concepts
    logger.info("Step 1: Extracting CSO concepts")
    start_time = time.time()
    try:
        cso_module = import_module("cso_concept", "1_cso_concept.py")
        cso_module.main_cso_concepts()  # Assuming main function name
    except Exception as e:
        logger.error(f"Error in Step 1: {e}")
    execution_times["Step 1"] = time.time() - start_time
    logger.info(f"Step 1 completed in {execution_times['Step 1']:.2f} seconds")

    # Step 2: Process paper dataset
    logger.info("Step 2: Processing paper dataset")
    start_time = time.time()
    try:
        paper_module = import_module("paper_dataset", "2_paper_dataset.py")
        paper_module.main_load_papers()
    except Exception as e:
        logger.error(f"Error in Step 2: {e}")
    execution_times["Step 2"] = time.time() - start_time
    logger.info(f"Step 2 completed in {execution_times['Step 2']:.2f} seconds")

    # Step 3: Partition abstracts into parts
    logger.info("Step 3: Partitioning abstracts")
    start_time = time.time()
    try:
        partitions_module = import_module("dataset_partitions", "3_dataset_partitions.py")
        partitions_module.main_split()  # Assuming main function name
    except Exception as e:
        logger.error(f"Error in Step 3: {e}")
    execution_times["Step 3"] = time.time() - start_time
    logger.info(f"Step 3 completed in {execution_times['Step 3']:.2f} seconds")

    # Step 4: Clean data
    logger.info("Step 4: Cleaning data")
    start_time = time.time()
    try:
        clean_module = import_module("clean_data", "4_clean_data.py")
        clean_module.main_glue()
    except Exception as e:
        logger.error(f"Error in Step 4: {e}")
    execution_times["Step 4"] = time.time() - start_time
    logger.info(f"Step 4 completed in {execution_times['Step 4']:.2f} seconds")

    # Step 5: Strip token data
    logger.info("Step 5: Strip token data")
    start_time = time.time()
    try:
        tokens_module = import_module("strip_tokens", "5_strip_tokens.py")
        tokens_module.main_strip()  # Assuming main function name
    except Exception as e:
        logger.error(f"Error in Step 5: {e}")
    execution_times["Step 5"] = time.time() - start_time
    logger.info(f"Step 5 completed in {execution_times['Step 5']:.2f} seconds")

    # Step 6: Generate bigrams and trigrams
    logger.info("Step 6: Generating bigrams and trigrams")
    start_time = time.time()
    try:
        bigrams_module = import_module("bigrams_trigrams", "6_bigrams_trigrams.py")
        bigrams_module.main_bigrams_trigrams()
    except Exception as e:
        logger.error(f"Error in Step 6: {e}")
    execution_times["Step 6"] = time.time() - start_time
    logger.info(f"Step 6 completed in {execution_times['Step 6']:.2f} seconds")

    # Step 7: Train Word2Vec model
    logger.info("Step 7: Training Word2Vec model")
    start_time = time.time()
    try:
        w2v_module = import_module("w2v_model", "7_w2v_model.py")
        w2v_module.main_word2vec()
    except Exception as e:
        logger.error(f"Error in Step 7: {e}")
    execution_times["Step 7"] = time.time() - start_time
    logger.info(f"Step 7 completed in {execution_times['Step 7']:.2f} seconds")

        # Step 8: Cache Word2Vec similarities with CSO ontology
    logger.info("Step 8: Caching Word2Vec model with CSO ontology")
    start_time = time.time()
    try:
        cache_module = import_module("caching_word2vec_model", "8_caching_word2vec_model.py")
        cache_module.main()  # assuming you wrap your code in a main() function
    except Exception as e:
        logger.error(f"Error in Step 8: {e}")
    execution_times["Step 8"] = time.time() - start_time
    logger.info(f"Step 8 completed in {execution_times['Step 8']:.2f} seconds")


    # Find and log the step with maximum execution time
    max_step = max(execution_times, key=execution_times.get)
    max_time = execution_times[max_step]
    logger.info(f"Maximum execution time: {max_step} with {max_time:.2f} seconds")

    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    run_pipeline()