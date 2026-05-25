import os
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentProvider(ABC):
    """
    Abstract Port for document source ingestion.
    Abstracts filesystem directory walking from retriever logic.
    """
    @abstractmethod
    def get_documents(self) -> List[Dict[str, str]]:
        """Returns a list of dicts with 'path' and 'content' keys."""
        pass

class FileSystemDocumentProvider(DocumentProvider):
    """
    Production Adapter for loading documents from the local filesystem.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_documents(self) -> List[Dict[str, str]]:
        if not os.path.exists(self.data_dir):
            print(f"[WARNING] data directory {self.data_dir} does not exist.")
            return []

        docs_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.md', '.txt')):
                    abs_path = os.path.join(root, file)
                    # Relative to parent of data directory
                    repo_root = os.path.dirname(self.data_dir)
                    rel_path = os.path.relpath(abs_path, repo_root).replace('\\', '/')
                    
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
        return docs_list

class InMemoryDocumentProvider(DocumentProvider):
    """
    Test Adapter for loading mock documents directly in memory.
    """
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents

    def get_documents(self) -> List[Dict[str, str]]:
        return self.documents

class IRetriever(ABC):
    """
    Abstract Port representing the corpus retrieval seam.
    """
    @abstractmethod
    def retrieve(self, query: str, company: str = None, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieves top_k relevant documents relative to query and company."""
        pass

class SupportRetriever(IRetriever):
    """
    Corpus retrieval module implementing IRetriever seam.
    Uses TF-IDF vectorization over a DocumentProvider source.
    """
    def __init__(self, provider: Optional[DocumentProvider] = None):
        if provider is None:
            # Standard production filesystem resolver
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
            provider = FileSystemDocumentProvider(data_dir)
            
        self.provider = provider
        self.documents: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=1)
        self.tfidf_matrix = None
        self.paths: List[str] = []
        self.corpus_loaded = False
        
        # Load and fit index
        self._load_corpus()

    def _load_corpus(self):
        """Loads all documents from DocumentProvider and builds TF-IDF index."""
        docs_list = self.provider.get_documents()
        if not docs_list:
            print("[WARNING] No documents found to index.")
            return

        self.documents = docs_list
        self.paths = [d["path"] for d in docs_list]
        texts = [d["content"] for d in docs_list]
        
        # Fit vectorizer
        if len(texts) < 5:
            self.vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.corpus_loaded = True
        print(f"[OK] Indexed {len(self.documents)} corpus documents successfully.")

    def retrieve(self, query: str, company: str = None, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieves the top_k most relevant documents for the query.
        """
        if not self.corpus_loaded or not query:
            return []

        # Transform query
        try:
            query_vec = self.vectorizer.transform([query])
        except Exception:
            return []
            
        # Compute cosine similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Rank document indices by similarity
        ranked_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in ranked_indices:
            score = similarities[idx]
            if score <= 0.001:
                continue
            
            doc_path = self.paths[idx]
            
            # Weighted matching by company
            if company and company.lower() != 'none':
                comp_pattern = f"/{company.lower()}/"
                if comp_pattern not in f"/{doc_path.lower()}/":
                    score *= 0.1
            
            results.append((idx, score))
            
        # Re-sort
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
