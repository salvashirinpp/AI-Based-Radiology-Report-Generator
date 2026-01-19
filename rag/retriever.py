import faiss
import pickle
from sentence_transformers import SentenceTransformer

VECTOR_PATH = "rag/vector_store"

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(f"{VECTOR_PATH}/faiss.index")

with open(f"{VECTOR_PATH}/metadata.pkl", "rb") as f:
    documents, metadata = pickle.load(f)

def retrieve_docs(facts, top_k=3):
    """
    facts: output of confidence gate
    """
    queries = facts["findings"] + facts["uncertain"]

    if not queries:
        queries = ["normal chest x-ray"]

    query_text = " ".join(queries)
    query_embedding = model.encode([query_text])

    distances, indices = index.search(query_embedding, top_k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(documents[idx])

    return "\n".join(retrieved)

if __name__ == "__main__":
    test_facts = {
        "findings": ["cardiomegaly"],
        "uncertain": [],
        "normal": []
    }

    print(retrieve_docs(test_facts))
