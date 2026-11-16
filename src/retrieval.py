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


