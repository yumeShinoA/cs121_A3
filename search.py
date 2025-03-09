import logging
import json
import math
from urllib.parse import urlparse, urlunparse
from Indexer import tokenize, stem_tokens

class QueryProcessor:
    def __init__(self, vocab_path, index_path, doc_id_map_path, page_rank_path):
        self.vocab_path = vocab_path
        self.index_path = index_path
        self.doc_id_map_path = doc_id_map_path
        self.page_rank_path = page_rank_path
        self.page_rank = None
        self.vocab = None
        self.id_to_url = None
        self.doc_count = 0
        self.load_resources()

    def load_resources(self):
        """Load vocabulary and document ID mapping."""
        try:
            with open(self.vocab_path, 'r') as f:
                self.vocab = json.load(f)
            
            with open(self.doc_id_map_path, 'r') as f:
                doc_id_map = json.load(f)
                self.id_to_url = {int(doc_id): url for url, doc_id in doc_id_map.items()}
                self.doc_count = len(doc_id_map)

            with open(self.page_rank_path, 'r') as f:
                self.page_rank = json.load(f)
                
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

    def compute_tf_idf(self, postings):
        """Compute the TF-IDF score for a postings list."""
        tf_idf_scores = {}
        for doc_id, freq in postings.items():
            tf = 1 + math.log(freq)
            idf = math.log(self.doc_count / len(postings))
            tf_idf_scores[doc_id] = tf * idf
        return tf_idf_scores

    @staticmethod
    def combine_scores(cosine_scores, pagerank_mapping, alpha=0.7, beta=0.3):
        """
        Combines cosine similarity scores with PageRank scores.
        Parameters:
        cosine_scores: dict mapping doc_id -> cosine similarity score.
        pagerank_mapping: dict mapping doc_id -> PageRank score.
        alpha: weight for cosine similarity.
        beta: weight for PageRank.
        Returns:
        A dictionary mapping doc_id to a combined score.
        """
        combined = {}
        for doc_id, cos_sim in cosine_scores.items():
            pr = pagerank_mapping.get(str(doc_id), 0)
            combined[doc_id] = alpha * cos_sim + beta * pr
        return combined

    def normalize_url(self, url):
        """Normalize URL by removing fragments and query parameters."""
        parsed_url = urlparse(url)
        normalized_url = urlunparse(parsed_url._replace(fragment='', query=''))
        return normalized_url

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
            tf_idf_scores = self.compute_tf_idf(p)
            postings.append(tf_idf_scores)

        # Sort postings by length for optimal intersection
        postings.sort(key=lambda x: len(x))

        # Compute intersection using sets
        doc_sets = [set(p.keys()) for p in postings]
        result = doc_sets[0]
        for s in doc_sets[1:]:
            result = result.intersection(s)
            if not result:
                return []

        # Compute cosine similarity
        query_vector = {token: 1 for token in stemmed_tokens}
        doc_vectors = {doc_id: {} for doc_id in result}
        for token, tf_idf_scores in zip(stemmed_tokens, postings):
            for doc_id, score in tf_idf_scores.items():
                if doc_id in doc_vectors:
                    doc_vectors[doc_id][token] = score

        cosine_scores = {}
        for doc_id, vector in doc_vectors.items():
            dot_product = sum(query_vector[token] * vector.get(token, 0) for token in query_vector)
            query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))
            doc_norm = math.sqrt(sum(val ** 2 for val in vector.values()))
            cosine_similarity = dot_product / (query_norm * doc_norm)
            cosine_scores[doc_id] = cosine_similarity

        # Now combine these with PageRank scores.
        combined = self.combine_scores(cosine_scores, self.page_rank, alpha=0.7, beta=0.3)

        # Map to URLs and normalize them
        results = []
        seen_urls = set()
        for doc_id, _ in sorted(combined.items(), key=lambda item: item[1], reverse=True):
            url = self.id_to_url.get(doc_id)
            if url:
                normalized_url = self.normalize_url(url)
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    results.append(normalized_url)
            else:
                logging.warning(f"Missing URL mapping for doc_id: {doc_id}")
        
        return results
