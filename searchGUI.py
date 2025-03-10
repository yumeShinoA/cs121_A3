import logging
import time
import math
import os

log_folder = "logs"
log_filename = "query.log"
log_path = os.path.join(log_folder, log_filename)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QProgressBar
)
from PyQt5.QtGui import QKeySequence, QDesktopServices
from PyQt5.QtCore import Qt, QUrl, QTimer
from search import QueryProcessor

class CopyableListWidget(QListWidget):
    def keyPressEvent(self, event):
        # If Ctrl+C is pressed, copy selected items to clipboard
        if event.matches(QKeySequence.Copy):
            selected_items = self.selectedItems()
            clipboard_text = "\n".join(item.text() for item in selected_items)
            QApplication.clipboard().setText(clipboard_text)
        else:
            super().keyPressEvent(event)

class SearchEngineUI(QWidget):
    def __init__(self, query_processor):
        super().__init__()
        self.query_processor = query_processor
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Search Engine")
        
        # Horizontal layout: query input and search button
        h_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your search query...")
        self.query_input.returnPressed.connect(self.perform_search)  # Connect 'Enter' key to search
        h_layout.addWidget(self.query_input)
        
        self.search_button = QPushButton("Search")
        h_layout.addWidget(self.search_button)
        self.search_button.clicked.connect(self.perform_search)
        
        self.clear_button = QPushButton("Clear")
        h_layout.addWidget(self.clear_button)
        self.clear_button.clicked.connect(self.clear_results)
        
        self.results_list = CopyableListWidget()
        self.results_list.itemClicked.connect(self.open_url)
        
        self.result_count_label = QLabel()
        self.loading_indicator = QProgressBar()
        self.loading_indicator.setRange(0, 0)
        self.loading_indicator.setVisible(False)
        
        self.query_time_label = QLabel()
        
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.result_count_label)
        v_layout.addWidget(self.loading_indicator)
        v_layout.addWidget(self.query_time_label)
        v_layout.addWidget(self.results_list)
        
        self.setLayout(v_layout)
        self.resize(1000, 800)
    
    def perform_search(self):
        query = self.query_input.text().strip()
        if not query:
            self.results_list.clear()
            self.results_list.addItem("Please enter a query.")
            return
        
        self.loading_indicator.setVisible(True)
        QTimer.singleShot(100, self.search_query)  # Delay to show the loading indicator
    
    def search_query(self):
        query = self.query_input.text().strip()
        start_time = time.time()
        results = self.query_processor.process_query(query)
        end_time = time.time()
        query_time = (end_time - start_time) * 1000  # Convert to milliseconds
        query_time = math.ceil(query_time + 0.1)  # Round up to the nearest millisecond
        
        self.results_list.clear()
        self.loading_indicator.setVisible(False)
        self.query_time_label.setText(f"Query time: {query_time} ms")
        if results:
            self.result_count_label.setText(f"Results found: {len(results)}")
            for url in results:
                item = QListWidgetItem(url)
                self.results_list.addItem(item)
        else:
            self.result_count_label.setText("Results found: 0")
            self.results_list.addItem("No results found.")
    
    def clear_results(self):
        self.query_input.clear()
        self.results_list.clear()
        self.result_count_label.clear()
        self.query_time_label.clear()
    
    def open_url(self, item):
        """Open the clicked URL in the default web browser."""
        url = item.text()
        QDesktopServices.openUrl(QUrl(url))

def main():
    # Locate resource files
    script_dir = os.path.abspath(os.path.dirname(__file__))

    # Load paths from config.json
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    doc_id_map_path = os.path.join(script_dir, config["doc_id_map_path"])
    final_index_path = os.path.join(script_dir, config["final_index_path"])
    vocab_path = os.path.join(script_dir, config["vocab_path"])
    page_rank_path = os.path.join(script_dir, config["page_rank_path"])
    
    qp = QueryProcessor(vocab_path, final_index_path, doc_id_map_path, page_rank_path)
    
    app = QApplication(sys.argv)
    ui = SearchEngineUI(qp)
    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()