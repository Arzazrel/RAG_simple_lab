"""
File for the retrieval component of RAG system. 
In this file there re different classes each one for a different level of information retrieval (from the simplest to the most difficult).

Author: Alessandro Diana
Version: November 17, 2026
"""
# imports for retrieval level 1
from pathlib import Path
import difflib

# imports for retrieval level 2
import numpy as np
from sentence_transformers import SentenceTransformer

# imports for retrieval level 3
import requests
import re
import html

# ------------------------------------ LEVEL 1: retrieval by name ------------------------------------


class NameRetriever:
    """
    The simplest retrieval system based on name file search.
    Create a list of document names, context fro the RAG.
    Search the filenames most similar to the query and return them in chunks (dictionary cntaining name and text).
    Use to demonstrates the limitation of lexical matching compared to semantic matching and show the simplet way.
    """
    
    def __init__(self, folder: str):
        """
        Initialize the files lists (one for the path and the other for the name).

        Args:
            folder: is a string for the path of the directory containing all the context files
        """
        
        self.files = list(Path(folder).glob("*.txt"))
        self.names = [f.stem for f in self.files]

    def retrieve(self, query: str, k: int = 1):
        """
        retriver the chunks from files..

        Args: 
            query: user's query
            k: the number of the file to retrieve (default is 1)
        
        Return: 
            a list of dictionary ("source": name, "text": content)
        """
        
        best = difflib.get_close_matches(query, self.names, n=k, cutoff=0.0)    # get best fit file
        chunks = []
        for name in best:
            fp = self.files[self.names.index(name)]
            chunks.append({"source": fp.name, "text": fp.read_text(encoding="utf-8")})
        return chunks           # one chunck for each retrieved file


# ------------------------------------ LEVEL 2: retrieval by semantic matching (chunking + embedding) ------------------------------------

# Scelta: sentence-transformers (basato su PyTorch) per embedding di qualita'
# in locale; coseno via prodotto scalare su vettori normalizzati.


class SemanticRetriever:
    """
    The usual retriever for RAG that use vectors embeddings for the query and the documents for the context.
    
    Using 'sentence-transformers' (PyTorch-based) to generate high-quality text embeddings locally. 
    Specifically, the 'all-MiniLM-L6-v2' model maps sentences and paragraphs to a 384-dimensional dense vector space, 
    optimized for semantic search, clustering, and Retrieval-Augmented Generation (RAG). 
    Similarity is computed efficiently via cosine similarity on normalized vectors.
    """
    
    def __init__(self, folder: str, model_name="all-MiniLM-L6-v2", chunk_size=400, overlap=50,
                 cache_dir: str | None = None):
        # cache_folder: save the embedding model locally so it is downloaded only once.
        self.embedder = SentenceTransformer(model_name, cache_folder=cache_dir)  # the embedder used
        self.chunks = self._load_chunks(folder, chunk_size, overlap)    # the chunks
        self.index = self._build_index()                                # the chunks embeddings

    def _load_chunks(self, folder, size, overlap):
        """
        Divide the context document in chunks.
        
        Args:
            folder: is a string for the path of the directory containing all the context files
            size: the size of the chunks
            overlap: the overlpped token between a chunks and the its neighbours
            
        Returns: the list of chunks
        """
        
        chunks = []
        for fp in Path(folder).glob("*.txt"):
            text = fp.read_text(encoding="utf-8")
            iteration = 0
            for i in range(0, len(text), size - overlap):
                piece = text[i:i + size].strip()
                if piece:
                    chunk_name = fp.name + "_" + str(iteration)
                    chunks.append({"source": chunk_name, "text": piece})
                iteration += 1
        return chunks

    def _build_index(self):
        """Metod to create the embeddings for the chunks. """
        vecs = self.embedder.encode([c["text"] for c in self.chunks],normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def retrieve(self, query: str, k: int = 3):
        """
        Method to compare the chunks with the query and return the most k relevant ones.
        
        Args:
            query: the user query
            k: the number of top 
            
        Returns: the chosen chunks
        """
        if not self.chunks or self.index.size == 0:
            return []  
        
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.index @ q                 # cosine similarity (normalized vectors)
        top = np.argsort(scores)[::-1][:k]      # get the top k
        return [self.chunks[i] for i in top]
        
    def min_similarity_retrieve(self, query: str, k: int = 3, min_similarity: float = 0.3):
        """
        Method to compare the chunks with the query and return the most k relevant ones.
        
        Args:
            query: the user query
            k: the number of top 
            min_similarity: the minimu similarity request for the best matching chunks from the context
            
        Returns: the chosen chunks or an empty list if the min similarity threshold is not met
        """
        if not self.chunks or self.index.size == 0:
            return []  
        
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.index @ q                 # cosine similarity (normalized vectors)
        
        best = scores[np.argmax(scores)] if scores.size > 0 else -1.0

        if best >= min_similarity: # local evidence is good enough
            top = np.argsort(scores)[::-1][:k]      # get the top k
            return [self.chunks[i] for i in top]
        else:
            print(f"[hybrid] best local score {best:.3f} < {min_similarity}: searching the web...")
            return []                   # local evidence isn't good enough


# ------------------------------------ LEVEL 3.0: retrieval by internet ------------------------------------

# base class
class WebRetriever:
    """
    Retrieval from the internet (Wikipedia). Use wikipedia because has public API, free and keyless.
    This class hasn't chunks, search from internet. Saving the context retrieved from the internet 
    for each individual question would take up a lot of memory and consume significant resources.

    Two-step MediaWiki call:
      1) list=search  -> find the most relevant article titles for the query;
      2) prop=extracts -> download the plain-text intro of those articles.
    The article intro is long enough to be a real context chunk, which is the
    main fix compared to using the search 'snippet' alone.
    """
    URL = "https://en.wikipedia.org/w/api.php"                                  # English wiki (this project use english text)
    HEADERS = {"User-Agent": "RAGRetrieverBot/1.0 (educational RAG project)"}   # Wikipedia asks for a descriptive User-Agent identifying the client.

    def __init__(self, lang: str = "en", sentences: int = 5):
        """
        Args:
            lang: Wikipedia language edition (e.g. 'en', 'it').
            sentences: how many sentences of each article intro to keep.
        """
        self.URL = f"https://{lang}.wikipedia.org/w/api.php"
        self.sentences = sentences
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def retrieve(self, query: str, k: int = 3):
        """
        Search Wikipedia and return up to k chunks with the article intros.

        Args:
            query: the user query
            k: the number of articles to retrieve

        Returns: a list of dict ("source": "wikipedia:<title>", "text": <intro>).
                 Returns [] on any network/parse error, so the caller can degrade
                 gracefully instead of crashing.
        """
        try:
            titles = self._search_titles(query, k)
            if not titles:
                return []
            return self._fetch_extracts(titles)
        except requests.RequestException as e:
            print(f"[web] network error: {e}")
            return []

    def _search_titles(self, query: str, k: int):
        """Step 1: get the k most relevant article titles for the query."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": k,
            "srprop": "",              # we only need titles here
            "format": "json",
        }
        data = self.session.get(self.URL, params=params, timeout=15).json()
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def _fetch_extracts(self, titles: list[str]):
        """Step 2: download the plain-text intro of each title."""
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,              # only the lead section
            "explaintext": 1,          # plain text, no HTML
            "exsentences": self.sentences,
            "titles": "|".join(titles),
            "redirects": 1,            # follow redirects to the real article
            "format": "json",
        }
        data = self.session.get(self.URL, params=params, timeout=15).json()
        pages = data.get("query", {}).get("pages", {})
        chunks = []
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:                # skip missing pages (no 'extract')
                text = " ".join(html.unescape(extract).split())   # normalize whitespace
                chunks.append({"source": "wikipedia:" + page.get("title", ""), "text": text})
        return chunks


# ------------------------------------ LEVEL 3.1: local semantic + web fallback(HYBRID) ------------------------------------

# advance class
class HybridRetriever:
    """
    Corrective-RAG-style retriever: try the local semantic retriever first and, 
    only if the BEST local match is weak, fall back to the web.

    This keeps the two sources cleanly separated (no re-indexing of the whole
    corpus, no permanent pollution of the local chunks) and decides the fallback
    on the strongest local hit..
    """
    
    def __init__(self, folder: str, model_name="all-MiniLM-L6-v2",chunk_size=400, overlap=50, min_similarity: float = 0.35,web_lang: str = "en", cache_dir: str | None = None):
        """
        Args:
            model_name: the model for generate the document embeddings
            chunk_size:size of the chunks
            overlap: the number of token share by a chunk and its neighbour
            min_similarity: if the top local chunk scores below this cosine
                            similarity, the web retriever is queried as a fallback.
            web_lang: the language of the wikipedia
        """
        
        self.local = SemanticRetriever(folder, model_name, chunk_size, overlap, cache_dir=cache_dir)
        self.web = WebRetriever(lang=web_lang)
        self.min_similarity = min_similarity

    def retrieve(self, query: str, k: int = 3):
        """
        Method to compare the chunks with the query and return the most k relevant ones.
        
        Args:
            query: the user query
            k: the number of top 
            
        Returns: up to k chunks. Local chunks carry a 'score'; web chunks are appended only when the local evidence is insufficient.
        """
        local_hits = self.local.min_similarity_retrieve(query, k,self.min_similarity)

        if local_hits:
            return local_hits          # local evidence is good enough

        # weak local match -> enrich with web results
        web_hits = self.web.retrieve(query, k=k)
        # keep the strongest local hit(s) plus the web hits, trimmed to k
        combined = local_hits + web_hits
        return combined[:k] if combined else []


# ------------------------------------ quick manual test ------------------------------------


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"

    print("== LEVEL 2: semantic ==")
    sem = SemanticRetriever(folder)
    for c in sem.retrieve("What measures acoustic pressure underwater?", k=2):
        print(f"  [{c['score']:.3f}] {c['source']}: {c['text'][:70]}...")

    print("\n== LEVEL 3: web (needs internet) ==")
    web = WebRetriever()
    for c in web.retrieve("magnetic loop antenna", k=2):
        print(f"  {c['source']}: {c['text'][:70]}...")
