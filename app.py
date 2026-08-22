from src.data_loader import load_all_documents
from src.vector_store import Faiss_Vector_Store
from src.search import RAG_Search

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = Faiss_Vector_Store("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAG_Search()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)