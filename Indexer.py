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
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class Indexer:
    MEMORY_THRESHOLD = 200000  # Artificial memory limit for partial index (number of unique tokens)

    def __init__(self):
        self.inverted_index = {}  # In-memory index: token -> {doc_int: frequency}
        self.token_frequencies = Counter()  # Global token frequencies
        self.partial_index_count = 0  # Counter for partial index files

        # Map document URLs to integer IDs for space efficiency
        self.doc_id_map = {}
        self.next_doc_id = 0
    @timer
    def _get_doc_int(self, doc_url):
        """Maps a document URL to an integer ID."""
        if doc_url not in self.doc_id_map:
            self.doc_id_map[doc_url] = self.next_doc_id
            self.next_doc_id += 1
        return self.doc_id_map[doc_url]
    @timer
    def add_document(self, doc_url, tokens):
        """
        Adds tokens from a document to the inverted index.
        Uses integer doc IDs instead of full URLs.
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
    @timer
    def write_partial_index(self):
        """
        Writes the current inverted index to disk as a sorted list of [token, postings] pairs.
        After writing, clears the in-memory index.
        """
        # Produce a sorted list of (token, postings) pairs
        sorted_index = sorted(self.inverted_index.items())
        
        filename = f"partial_index_{self.partial_index_count}.json"
        with open(filename, "w") as f:
            json.dump(sorted_index, f, indent=4)
        logging.info(f"Saved partial index: {filename}")
        
        # Clear the in-memory index completely
        self.inverted_index.clear()
        
        self.partial_index_count += 1
    @timer
    def merge_indexes(self):
        """
        Merges all partial index files into a final inverted index using a multi-way merge.
        Writes the final index to 'final_index.json' and the doc ID mapping to 'doc_id_map.json'.
        """
        partial_files = glob.glob("partial_index_*.json")
        # Load each partial index as a list of (token, postings) pairs
        partial_lists = []
        for file in partial_files:
            with open(file, "r") as f:
                partial_lists.append(json.load(f))
        
        # Prepare iterators for each partial list and initialize heap
        iterators = [iter(lst) for lst in partial_lists]
        heap = []
        for i, it in enumerate(iterators):
            try:
                token, postings = next(it)
                heapq.heappush(heap, (token, i, postings))
            except StopIteration:
                pass
        
        final_index = {}
        # Multi-way merge: repeatedly take the smallest token from the heap
        while heap:
            current_token, idx, postings = heapq.heappop(heap)
            merged_postings = postings.copy()
            # Merge any other entries with the same token
            while heap and heap[0][0] == current_token:
                _, j, postings_j = heapq.heappop(heap)
                for doc, freq in postings_j.items():
                    merged_postings[doc] = merged_postings.get(doc, 0) + freq
                try:
                    next_token, next_postings = next(iterators[j])
                    heapq.heappush(heap, (next_token, j, next_postings))
                except StopIteration:
                    pass
            # Add merged postings for current_token to final index
            final_index[current_token] = merged_postings
            # Push the next element from the iterator idx, if available
            try:
                next_token, next_postings = next(iterators[idx])
                heapq.heappush(heap, (next_token, idx, next_postings))
            except StopIteration:
                pass
        
        # Write the final merged index to disk
        with open("final_index.json", "w") as f:
            json.dump(final_index, f, indent=4)
        logging.info(f"Merged partial indexes into final_index.json with {len(final_index)} unique tokens.")
        
        # Write the document ID mapping to disk
        with open("doc_id_map.json", "w") as f:
            json.dump(self.doc_id_map, f, indent=4)
        logging.info("Saved document ID mapping into doc_id_map.json")
@timer
# Text Processing Functions
def extract_text_from_html(html):
    """
    Parses HTML and extracts text, giving extra importance to headings, bold text, and titles.
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string if soup.title else ""
    important_text = " ".join([elem.get_text() for elem in soup.find_all(["h1", "h2", "h3", "b", "strong"])])
    body_text = " ".join(soup.stripped_strings)
    return f"{title} {important_text} {body_text}" 
@timer
def tokenize(text):
    """
    Splits text into alphanumeric tokens.
    """
    return re.findall(r'\b\w+\b', text.lower())
@timer
def stem_tokens(tokens):
    """
    Applies stemming to tokens using the PorterStemmer.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]
@timer
def process_json_files(root_directory, indexer):
    """
    Recursively processes all JSON files under the given root directory.
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