import logging

logging.basicConfig(
        filename='indexer.log',  # Log file name
        level=logging.INFO,      # Logging level
        format='%(asctime)s - %(levelname)s - %(message)s'
)

from Indexer import Indexer, process_json_files
import time

def main():
    start_time = time.time()
    indexer = Indexer()
    process_json_files("DEV", indexer)
    indexer.merge_indexes()
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()