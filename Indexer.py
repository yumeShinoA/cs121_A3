import json
import logging
import glob
import re
import os
import heapq
import time
from collections import Counter
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

def timer(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Runtime: O(N) to load the JSON file, where N is the number of entries in the file
def read_partial_index(file_path):
    """Generator: Yields (token, postings) from a partial index file without loading everything at once"""
    with open(file_path, "r") as f:
        # Load the JSON array once and yield items one by one
        for item in json.load(f):
            yield item

class Indexer:
    MEMORY_THRESHOLD = 200000  # Artificial memory limit for partial index (number of unique tokens)

    def __init__(self):
        self.inverted_index = {}  # In-memory index: token -> {doc_int: frequency}
        self.token_frequencies = Counter()  # Global token frequencies
        self.partial_index_count = 0  # Counter for partial index files

        # Map document URLs to integer IDs for space efficiency
        self.doc_id_map = {}
        self.next_doc_id = 0

    def _get_doc_int(self, doc_url):
        """Maps a document URL to an integer ID"""
        if doc_url not in self.doc_id_map:
            self.doc_id_map[doc_url] = self.next_doc_id
            self.next_doc_id += 1
        return self.doc_id_map[doc_url]

    # Runtime: O(m) where m is the number of tokens in the document.
    #          Additionally, checking the threshold is O(1)
    def add_document(self, doc_url, tokens):
        """
        Adds tokens from a document to the inverted index.
        Uses integer doc IDs instead of full URLs
        """
        doc_int = self._get_doc_int(doc_url)
        for token in tokens:
            self.token_frequencies[token] += 1
            if token not in self.inverted_index:
                self.inverted_index[token] = {}
            if doc_int not in self.inverted_index[token]:
                self.inverted_index[token][doc_int] = 0
            self.inverted_index[token][doc_int] += 1

        # Check if we've reached the memory threshold
        if len(self.inverted_index) >= self.MEMORY_THRESHOLD:
            self.write_partial_index()

    # Runtime: O(k log k), where k is the number of unique tokens in memory (due to sorting).
    #          JSON dumping is roughly O(k) depending on data size
    def write_partial_index(self):
        """
        Writes the current inverted index to disk as a sorted list of [token, postings] pairs.
        After writing, clears the in-memory index
        """
        sorted_index = sorted(self.inverted_index.items())
        filename = f"partial_index_{self.partial_index_count}.json"
        with open(filename, "w") as f:
            json.dump(sorted_index, f, indent=4)
        logging.info(f"Saved partial index: {filename}")
        self.inverted_index.clear()
        self.partial_index_count += 1

    # Runtime: O(1) check plus the cost of write_partial_index if data exists
    def flush(self):
        """
        Flushes any remaining in-memory index data to disk as a partial index
        """
        if self.inverted_index:
            logging.info("Flushing remaining in-memory index to disk.")
            self.write_partial_index()

    # Runtime: O(T log B) per batch, where T is the total number of tokens in the batch and
    #          B is the number of partial files in that batch. Then, merging batch results 
    #          into a global index adds additional overhead
    def merge_indexes(self, BATCH_SIZE=3):
        """
        Merges all partial index files into a final inverted index using a multi-way merge,
        processing files in batches to avoid loading everything into memory at once.
        Logs the number of unique tokens in the final index
        """
        partial_files = glob.glob("partial_index_*.json")
        final_index = {}

        # Process files in batches
        for i in range(0, len(partial_files), BATCH_SIZE):
            batch_files = partial_files[i: i + BATCH_SIZE]
            batch_final = {}
            iterators = {j: read_partial_index(file) for j, file in enumerate(batch_files)}
            heap = []
            # Initialize heap with first element from each iterator
            for j, it in iterators.items():
                try:
                    token, postings = next(it)
                    heapq.heappush(heap, (token, j, postings))
                except StopIteration:
                    pass

            while heap:
                current_token, idx, postings = heapq.heappop(heap)
                merged_postings = postings.copy()

                # Merge other entries with the same token from the heap
                while heap and heap[0][0] == current_token:
                    _, j, postings_j = heapq.heappop(heap)
                    for doc, freq in postings_j.items():
                        merged_postings[doc] = merged_postings.get(doc, 0) + freq
                    try:
                        next_token, next_postings = next(iterators[j])
                        heapq.heappush(heap, (next_token, j, next_postings))
                    except StopIteration:
                        pass

                # Merge into batch_final
                if current_token in batch_final:
                    for doc, freq in merged_postings.items():
                        batch_final[current_token][doc] = batch_final[current_token].get(doc, 0) + freq
                else:
                    batch_final[current_token] = merged_postings

                try:
                    next_token, next_postings = next(iterators[idx])
                    heapq.heappush(heap, (next_token, idx, next_postings))
                except StopIteration:
                    pass

            # Merge the batch result into the global final index
            for token, postings in batch_final.items():
                if token in final_index:
                    for doc, freq in postings.items():
                        final_index[token][doc] = final_index[token].get(doc, 0) + freq
                else:
                    final_index[token] = postings

        with open("final_index.json", "w") as f:
            json.dump(final_index, f, indent=4)
        logging.info(f"Merged indexes into final_index.json with {len(final_index)} unique tokens.")

        with open("doc_id_map.json", "w") as f:
            json.dump(self.doc_id_map, f, indent=4)
        logging.info("Saved document ID mapping into doc_id_map.json")

# Runtime: O(n), where n is the length of the HTML
def extract_text_from_html(html):
    """
    Parses HTML and extracts text, giving extra importance to headings, bold text, and titles
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string if soup.title else ""
    important_text = " ".join([elem.get_text() for elem in soup.find_all(["h1", "h2", "h3", "b", "strong"])])
    body_text = " ".join(soup.stripped_strings)
    return f"{title} {important_text} {body_text}"

# Runtime: O(n), where n is the length of the text
def tokenize(text):
    """
    Splits text into alphanumeric tokens
    """
    return re.findall(r'\b\w+\b', text.lower())

# Runtime: O(m), where m is the number of tokens
def stem_tokens(tokens):
    """
    Applies stemming to tokens using the PorterStemmer
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Runtime: O(D * (T + P)), where D is the number of documents, T is the time for tokenization
#          and stemming (linear in text size), and P is the overhead per document in add_document
def process_json_files(root_directory, indexer):
    """
    Recursively processes all JSON files under the given root directory
    """
    for subdir, _, _ in os.walk(root_directory):
        json_files = glob.glob(os.path.join(subdir, "*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                html = data.get("content", "")
                url = data.get("url", file)  # Use URL if available
            text = extract_text_from_html(html)
            tokens = tokenize(text)
            stemmed_tokens = stem_tokens(tokens)
            indexer.add_document(url, stemmed_tokens)
