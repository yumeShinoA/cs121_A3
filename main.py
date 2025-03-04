import logging
import os

# Define the log folder and filename
log_folder = "logs"
log_filename = "indexer.log"
log_path = os.path.join(log_folder, log_filename)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

logging.basicConfig(
        filename=log_path,  # Log file name
        level=logging.INFO,      # Logging level
        format='%(asctime)s - %(levelname)s - %(message)s'
)

from Indexer import Indexer, process_json_files
import time

def main():
    start_time = time.time()
    indexer = Indexer()

    # Get the directory where main.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev_path = os.path.join(script_dir, "DEV")  # guaranteed to point to DEV in project root
    
    logging.info(f"Searching in: {dev_path}")
    process_json_files(dev_path, indexer)
    indexer.flush()
    indexer.merge_indexes()
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()