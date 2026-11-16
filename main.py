"""
Main file, user interface.

Author: Alessandro Diana
Version: November 15, 2026
"""

from pathlib import Path
from src.model import LocalLLM          
from src.retrieval import NameRetriever, SemanticRetriever#, WebRetriever

# ----------------------- costant -----------------------
SEPARATOR_CHAR = 100                # number of '=' in UI print
DOCS_FOLDER = "docs"
retriever_type = ("name", "embedding")

# ----------------------- UI utils methods -----------------------


def print_line_list(el_list):
    """Method to print a list in this way: a new line for each element and '-' at the beginning of the line."""
    
    for elem in el_list:
        print("- ", elem)

def print_UI_chapter_head(chapter_name: str):
    """
    Method to print chapter head in this format -> " === chapter_name === "
    
    Args
        chapter_name: the name of the chapter
    """
    print("=" * (SEPARATOR_CHAR // 3) + chapter_name + "=" * (SEPARATOR_CHAR // 3))


def get_user_settings():
    """
    Method to get the user settings for whych type of model or retrieval to use.
    
    Returns:
        retriever: the type of retriever.
    """
    print_UI_chapter_head(" MODEL SETTINGS ")
    print("Available retrievers:")
    print_line_list(retriever_type)
    user_retriever = input("Choose the type of retriever: ")
    
    if user_retriever not in retriever_type:
        print("Unknown retriever.")
        return None
    elif user_retriever == retriever_type[0]:   # NameRetriever
        retriever = NameRetriever(DOCS_FOLDER)  
    elif user_retriever == retriever_type[1]:   # SemanticRetriever
        retriever = SemanticRetriever(DOCS_FOLDER)  
    # WebRetriever
        
    return retriever


def print_document_context_name():
    """
    Method to print the list of the context documents name.
    """
    
    path_list = list(Path(DOCS_FOLDER).glob("*.txt"))
    name_list = [f.stem for f in path_list]
    print_UI_chapter_head(" Context documents ")
    print_line_list(name_list)
          

def print_document_context_text(query, retrieved):
    """
    Method to print the key searched in the document.
    
    Args:
        query: is the user's query
        retrieved: the retrieved context (doictionaries of name and text for the documents)
    """
    print ("The texts of the used context documents:")
    for c in retrieved:
        print("-------------------")
        print("Chunk/Document name: " + c["source"])
        print("Chunk/document text: \n" + c["text"])


def build_prompt(query, retrieved):
    """
    Method that combines the question and the context derived for it to form the prompt for the model (ONLY for RAG).
    
    Args:
        query: is the user's query
        retrieved: list of dictionaies cintaning the context (name and text of the retrieved documents)
        
    Returns:
        the formatted prompt for the model.
    """
    ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in retrieved)    
    return (f"Answer using only the following context\n\n"
            f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:")


def main():
    """
    Main method to get user prompt, manage them and return the model answers.
    """
    
    llm = LocalLLM()                        # load model structur and weigths
    retriever = get_user_settings()
    print_document_context_name()
    
    while True:
        print("=" * SEPARATOR_CHAR)
        query = input("\nQuestion (blank to exit): ").strip()
        if not query:
            break
        # --- confronto A/B ---
        baseline = llm.generate(f"Question: {query}\nAnswer:")
        retrieved = retriever.retrieve(query)                   # context from local document
        rag = llm.generate(build_prompt(query, retrieved))

        print("\n[WITHOUT RAG]\n", baseline)
        print("\n[WITH RAG]\n", rag)
        print("\nUSED DOCUMENT:", [c["source"] for c in retrieved])
        print_document_context_text(query, retrieved)

if __name__ == "__main__":
    main()

"""
TEST ANSWER
    For 'instruments' -> What instrument is Florbometer?
    For 'creatures' -> What creatures do you know?
"""