import os
import glob
import json
import logging
import networkx as nx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def extract_links_from_html(html):
    """
    Parses HTML content and extracts all hyperlinks (absolute URLs).
    Runtime: O(n) where n is the length of the HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Optionally filter out non-http links (e.g., mailto:, javascript:)
        if href.startswith("http"):
            links.append(href)
    return links

def build_link_graph(root_directory):
    """
    Walks through JSON files in the given root directory, extracts links,
    and builds a directed graph where each node is a URL and each edge represents a hyperlink.
    Runtime: O(D * L) where D is the number of documents and L is the average number of links per document.
    """
    G = nx.DiGraph()
    for subdir, _, _ in os.walk(root_directory):
        json_files = glob.glob(os.path.join(subdir, "*.json"))
        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                url = data.get("url", file)
                html = data.get("content", "")
                # Add the page as a node
                G.add_node(url)
                # Extract outlinks from the HTML content
                outlinks = extract_links_from_html(html)
                for link in outlinks:
                    G.add_node(link)
                    G.add_edge(url, link)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")
    return G

def compute_and_save_pagerank(root_directory, output_path="Output/page_rank.json", damping=0.85, tol=1.0e-6, max_iter=100):
    """
    Builds the link graph from the JSON files, computes PageRank using NetworkX, and saves the results.
    Runtime: Each iteration is O(N + E), where N is the number of nodes and E the number of edges.
    Convergence usually takes a modest number of iterations.
    """
    logging.info("Building link graph for PageRank computation...")
    G = build_link_graph(root_directory)
    logging.info(f"Link graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    logging.info("Computing PageRank...")
    pagerank_scores = nx.pagerank(G, alpha=damping, tol=tol, max_iter=max_iter)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pagerank_scores, f, indent=4)
    logging.info(f"PageRank scores saved to {output_path}")
    return pagerank_scores