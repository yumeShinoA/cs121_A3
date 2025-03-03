import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLineEdit,
    QPushButton, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt
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
        h_layout.addWidget(self.query_input)
        
        self.search_button = QPushButton("Search")
        h_layout.addWidget(self.search_button)
        self.search_button.clicked.connect(self.perform_search)
        
        self.results_list = CopyableListWidget()
        
        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout)
        v_layout.addWidget(self.results_list)
        
        self.setLayout(v_layout)
        self.resize(600, 400)
    
    def perform_search(self):
        query = self.query_input.text().strip()
        if not query:
            self.results_list.clear()
            self.results_list.addItem("Please enter a query.")
            return
        
        results = self.query_processor.process_query(query)
        self.results_list.clear()
        if results:
            for url in results:
                item = QListWidgetItem(url)
                self.results_list.addItem(item)
        else:
            self.results_list.addItem("No results found.")

def main():
    # Locate resource files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    doc_id_map_path = os.path.join(parent_dir, "doc_id_map.json")
    final_index_path = os.path.join(parent_dir, "final_index.jsonl")
    vocab_path = os.path.join(parent_dir, "vocab.json")
    
    qp = QueryProcessor(vocab_path, final_index_path, doc_id_map_path)
    
    app = QApplication(sys.argv)
    ui = SearchEngineUI(qp)
    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()