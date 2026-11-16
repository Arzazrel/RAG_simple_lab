"""
File for the retrieval component of RAG system. 
In this file there re different classes each one for a different level of information retrieval (from the simplest to the most difficult).

Author: Alessandro Diana
Version: November 15, 2026
"""
# imports for retrieval level 1
from pathlib import Path
import difflib

# imports for retrieval level 2
import numpy as np
from sentence_transformers import SentenceTransformer

# imports for retrieval level 3

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
    def __init__(self, folder: str, model_name="all-MiniLM-L6-v2",chunk_size=400, overlap=50):
        self.embedder = SentenceTransformer(model_name)                 # the embedder used
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
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.index @ q                 # cosine similarity (normalized vectors)
        top = np.argsort(scores)[::-1][:k]      # get the top k
        return [self.chunks[i] for i in top]


# ------------------------------------ LEVEL : retrieval by  ------------------------------------