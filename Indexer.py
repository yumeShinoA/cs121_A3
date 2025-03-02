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

# Runtime: O(1) to load the JSON file line, O(N) for one merge process, where N is the number of entries in the file
def read_partial_index(file_path):
    """Generator: Yields (token, postings) from a partial index file line by line"""
    with open(file_path, "r") as f:
        for line in f:
            token, postings = json.loads(line)
            yield token, postings

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
    #          JSON dumping is roughly O(k) depending on data size. merging is done line by 
    #          line via the heap, which keep memory usage low
    @timer
    def write_partial_index(self):
        """
        Writes the current inverted index to disk as a sorted list of [token, postings] pairs.
        After writing, clears the in-memory index
        """
        sorted_index = sorted(self.inverted_index.items())
        filename = f"partial_index_{self.partial_index_count}.jsonl"
        with open(filename, "w") as f:
            for token, postings in sorted_index:
                json.dump([token, postings], f)  # Write as a JSON list
                f.write("\n")  # JSONL format: one entry per line
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

    def cleanup_partial_files(self, partial_files):
        """Deletes partial index files after successful merge."""
        for file in partial_files:
            try:
                os.remove(file)
                logging.info(f"Deleted partial index file: {file}")
            except OSError as e:
                logging.warning(f"Failed to delete {file}: {e}")

    # Runtime: O(T log P), where T is the total number of tokens across all partial files,
    #          and P is the number of partial files
    @timer
    def merge_indexes(self):
        """Merges all partial indexes into a final index using a global heap."""
        partial_files = glob.glob("partial_index_*.jsonl")
        vocab = {}
        heap = []
        iterators = []

        # Open all files and initialize the heap
        for file in partial_files:
            it = read_partial_index(file)
            try:
                token, postings = next(it)
                heapq.heappush(heap, (token, len(iterators), postings))
                iterators.append(it)
            except StopIteration:
                pass  # Skip empty files

        # Open final index file for writing
        with open("final_index.jsonl", "w") as final_file:
            current_token = None
            merged_postings = {}
            
            while heap:
                token, idx, postings = heapq.heappop(heap)

                if token != current_token:
                    if current_token is not None:
                        # Write the previous merged token
                        entry = [current_token, merged_postings]
                        position = final_file.tell()
                        json.dump(entry, final_file)
                        final_file.write("\n")
                        vocab[current_token] = position
                    current_token = token
                    merged_postings = postings.copy()
                else:
                    # Merge postings for the same token
                    for doc, freq in postings.items():
                        merged_postings[doc] = merged_postings.get(doc, 0) + freq

                # Fetch next token from the same iterator
                try:
                    next_token, next_postings = next(iterators[idx])
                    heapq.heappush(heap, (next_token, idx, next_postings))
                except StopIteration:
                    pass  # This iterator is exhausted

            # Write the last token
            if current_token is not None:
                entry = [current_token, merged_postings]
                position = final_file.tell()
                json.dump(entry, final_file)
                final_file.write("\n")
                vocab[current_token] = position

        # Save vocabulary and doc_id_map
        with open("doc_id_map.json", "w") as f:
            json.dump(self.doc_id_map, f, indent=4)

        with open("vocab.json", "w") as f:
            json.dump(vocab, f, indent=4)

        logging.info(f"Merged {len(partial_files)} files into final_index.jsonl")

        # Clean up partial index files
        self.cleanup_partial_files(partial_files)

# Runtime: O(n), where n is the length of the HTML
def extract_text_from_html(html):
    """
    Parses HTML and extracts text, giving extra importance to headings, bold text, and titles
    """
    soup = BeautifulSoup(html, "lxml")
    # Use get_text() to safely extract text, defaulting to empty string
    title = soup.title.get_text(strip=True) if soup.title else ""
    headers = " ".join(elem.get_text(strip=True) for elem in soup.find_all(["h1", "h2", "h3"]))
    bold_text = " ".join(elem.get_text(strip=True) for elem in soup.find_all(["b", "strong"]))
    body_text = " ".join(soup.stripped_strings)
    return {
        "title": title,
        "headers": headers,
        "bold": bold_text,
        "body": body_text if body_text else ""  # Ensure body is not None
    }

# Runtime: O(n), where n is the length of the text
def tokenize(text):
    """
    Splits text into alphanumeric tokens
    """
    if not text:  # Handles None, empty string, etc.
        return []
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
    logging.info(f"Processing JSON files in directory: {root_directory}")
    if not os.path.exists(root_directory):
        logging.error(f"Directory {root_directory} does not exist!")
        return
    
    for subdir, _, _ in os.walk(root_directory):
        json_files = glob.glob(os.path.join(subdir, "*.json"))
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                html = data.get("content", "")
                url = data.get("url", file)  # Use URL if available
            fields = extract_text_from_html(html)
            # Tokenize each field
            title_tokens = stem_tokens(tokenize(fields["title"]))
            header_tokens = stem_tokens(tokenize(fields["headers"]))
            bold_tokens = stem_tokens(tokenize(fields["bold"]))
            body_tokens = stem_tokens(tokenize(fields["body"]))
            # Apply multipliers to different fields
            weighted_tokens = (title_tokens * 3) + (header_tokens * 2) + (bold_tokens * 2) + body_tokens
            indexer.add_document(url, weighted_tokens)
