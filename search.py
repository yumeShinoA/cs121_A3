import logging
import json
from Indexer import tokenize, stem_tokens

class QueryProcessor:
    def __init__(self, vocab_path, index_path, doc_id_map_path):
        self.vocab_path = vocab_path
        self.index_path = index_path
        self.doc_id_map_path = doc_id_map_path
        self.vocab = None
        self.id_to_url = None
        self.load_resources()

    def load_resources(self):
        """Load vocabulary and document ID mapping."""
        try:
            with open(self.vocab_path, 'r') as f:
                self.vocab = json.load(f)
            
            with open(self.doc_id_map_path, 'r') as f:
                doc_id_map = json.load(f)
                self.id_to_url = {int(doc_id): url for url, doc_id in doc_id_map.items()}
                
        except FileNotFoundError as e:
            logging.error(f"Resource file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format: {e}")
            raise

    def get_postings(self, token):
        """Retrieve postings list for a token from the index."""
        if token not in self.vocab:
            return None
        offset = self.vocab[token]
        
        with open(self.index_path, 'r') as f:
            f.seek(offset)
            line = f.readline().strip()
            _, postings = json.loads(line)
            return {int(doc_id): freq for doc_id, freq in postings.items()}

    def process_query(self, query):
        """Process a Boolean AND query and return ranked URLs."""
        # Tokenize and stem query
        tokens = tokenize(query)
        stemmed_tokens = stem_tokens(tokens)
        if not stemmed_tokens:
            return []

        # Retrieve postings lists
        postings = []
        for token in stemmed_tokens:
            p = self.get_postings(token)
            if not p:
                return []  # Short-circuit if any token has no postings
            postings.append(p)

        # Sort postings by length for optimal intersection
        postings.sort(key=lambda x: len(x))

        # Compute intersection using sets
        doc_sets = [set(p.keys()) for p in postings]
        result = doc_sets[0]
        for s in doc_sets[1:]:
            result = result.intersection(s)
            if not result:
                return []

        # Score documents by sum of frequencies
        scored_docs = []
        for doc_id in result:
            score = sum(p[doc_id] for p in postings)
            scored_docs.append((doc_id, score))

        # Sort by score descending, then doc_id ascending
        scored_docs.sort(key=lambda x: (-x[1], x[0]))

        # Map to URLs
        results = []
        for doc_id, _ in scored_docs:
            url = self.id_to_url.get(doc_id)
            if url:
                results.append(url)
            else:
                logging.warning(f"Missing URL mapping for doc_id: {doc_id}")
        
        return results
