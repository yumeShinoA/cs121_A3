# Simple Search Engine

This is a basic search engine that builds an inverted index and computes PageRank scores from a collection of JSON files. It uses TF-IDF and cosine similarity combined with PageRank to rank search results. A local GUI is provided to enter queries and view results.

## Setup

### 1. Install Dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```
This installs all the necessary libraries.

### 2. Build the Index
Run the main indexing script to build the following ```Output``` files:

- ```final_index.jsonl``` – The final inverted index.
- ```vocab.json``` – Vocabulary with file offsets.
- ```doc_id_map.json``` – Mapping from document IDs to URLs.
- ```page_rank.json``` – PageRank scores for each document.

## Running the Search Engine

### 1. Launch the GUI
In the terminal, run:
```bash
python searchGUI.py
```
This will open a local search interface.

### 2. Using the Search Interface
1. Type your search query into the search bar.
2. Press "Enter" or click the "Search" button.
3. Search results will appear below the search bar.
4. The interface shows the number of results and the time taken to process the query.

## How It Works
- Indexing:  
The program reads JSON files from the dataset. It extracts text from HTML, tokenizes and stems it, and builds an inverted index. It also computes PageRank from the links in the pages.

- Searching:  
When you type a query, the search engine uses TF-IDF and cosine similarity to score documents. It then combines these scores with PageRank values to rank the results.

- Output Files:  
The index files and PageRank scores are saved in the ```Output``` folder for use during search.

## Notes
- Make sure your JSON files are placed in the correct folder (```DEV```).
- If you update the dataset, re-run main.py to rebuild the index.
- The code uses simple Boolean AND for queries.

