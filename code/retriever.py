import os
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SupportRetriever:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Resolve data_dir relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
        
        self.data_dir = data_dir
        self.documents: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
        self.tfidf_matrix = None
        self.paths: List[str] = []
        self.corpus_loaded = False
        
        # Load and fit index
        self._load_corpus()

    def _load_corpus(self):
        """Walks the data_dir, loads all .md and .txt documents and builds TF-IDF index."""
        if not os.path.exists(self.data_dir):
            print(f"[WARNING] data directory {self.data_dir} does not exist.")
            return

        docs_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    abs_path = os.path.join(root, file)
                    # Get path relative to the repository root (e.g. data/visa/...)
                    # Assuming repo root is the parent of the data dir
                    repo_root = os.path.dirname(self.data_dir)
                    rel_path = os.path.relpath(abs_path, repo_root).replace('\\', '/')
                    
                    # Read content with multiple encoding fallbacks to ensure robustness
                    content = ""
                    for encoding in ('utf-8', 'latin-1', 'iso-8859-1', 'cp1252'):
                        try:
                            with open(abs_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content.strip():
                        docs_list.append({
                            "path": rel_path,
                            "content": content
                        })

        if not docs_list:
            print("[WARNING] No documents found to index.")
            return

        self.documents = docs_list
        self.paths = [d["path"] for d in docs_list]
        texts = [d["content"] for d in docs_list]
        
        # Fit vectorizer
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.corpus_loaded = True
        print(f"[OK] Indexed {len(self.documents)} corpus documents successfully.")

    def retrieve(self, query: str, company: str = None, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieves the top_k most relevant documents for the query.
        Optionally filters or weights based on company.
        """
        if not self.corpus_loaded or not query:
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Rank document indices by similarity
        ranked_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            score = similarities[idx]
            if score <= 0.001:  # ignore completely irrelevant documents
                continue
            
            doc_path = self.paths[idx]
            
            # Apply a boost or filter if company is specified
            # We check if the doc matches the company directory
            # For example, if company is 'Visa', the doc path should contain '/visa/'
            if company and company.lower() != 'none':
                comp_pattern = f"/{company.lower()}/"
                # If it doesn't match the specified company, penalize the score or filter it
                if comp_pattern not in f"/{doc_path.lower()}/":
                    # We slightly penalize cross-company documents but don't filter them out entirely,
                    # in case the 'company' in the ticket is incorrect/misleading!
                    score *= 0.1
            
            results.append((idx, score))
            
        # Re-sort results based on weighted score
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        top_docs = []
        for idx, score in results[:top_k]:
            doc = self.documents[idx]
            top_docs.append({
                "path": doc["path"],
                "content": doc["content"],
                "score": float(score)
            })
            
        return top_docs
