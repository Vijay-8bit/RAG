import pandas as pd 
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import os 
from tqdm import tqdm
import requests
from docx import Document

load_dotenv()

Qclient = QdrantClient(os.getenv("QDRANT_URL"))


def embed_with_ollama(texts):
    try:
        url = os.getenv("OLLAMA_URL_EMBEDDING")
        response = requests.post(url, json={"model": os.getenv("OLLAMA_EMBEDDING_MODEL"), "prompt": texts})
        response.raise_for_status()
        return response.json()["embedding"]
    
    except Exception as e:
        print(f"Error embedding text with Ollama: {e}")
        return None

def create_collection(embedding_size):

    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    if not Qclient.collection_exists(collection_name):
        Qclient.recreate_collection(
            collection_name = collection_name,
            vectors_config = {"size": embedding_size, "distance": "Cosine"}
        )


def Store_Excel_to_Qdrant(Path = os.getenv("PATH_TO_EXCEL")):
    """
    Reads an Excel file, generates embeddings for question-response pairs using Ollama,
    and stores them in a Qdrant collection.
    """

    df = pd.read_excel(Path, header =1)

    pairs = [f"Q:{q} \nA:{a}" for q, a in zip(df["Questions"], df["Response"])]

    embeddings = [embed_with_ollama(pair) for pair in tqdm(pairs)]

    points = [
        PointStruct(
            id =i,
            vector = embeddings[i],
            payload = {"question": df["Questions"][i], "response": df["Response"][i]}
        )
        for i in range(len(pairs))
    ]

    create_collection(embedding_size=len(embeddings[0]))

    Qclient.upsert(
        collection_name = os.getenv("QDRANT_COLLECTION_NAME"),
        points = points
    )


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def Store_Doc_to_Qdrant(path = os.getenv("PATH_TO_DOC")):
    """
    Reads a document, generates embeddings for its content using Ollama,
    and stores them in a Qdrant collection.
    """
    if path.endswith(".docx"):
        doc = Document(path)
        content = "\n".join([para.text for para in doc.paragraphs])
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
    
    chunks = chunk_text(content)
    embeddings = [embed_with_ollama(chunk) for chunk in tqdm(chunks)]
    points = [ 
        PointStruct(
            id = f"doc_{i}",
            vector = embeddings[i],
            payload = {"content": chunks[i]}
        )
        for i in range(len(chunks))
    ]
    Qclient.upsert(
        collection_name = os.getenv("QDRANT_COLLECTION_NAME"),
        points = points
    )


def retrieve_from_qdrant(query, top_k = 5):
    """
    Retrieves the most similar response from Qdrant based on the query.
    """
    query_embedding = embed_with_ollama(query)

    if query_embedding is None:
        print("Failed to generate embedding for the query.")
        return None
    try:
        results = Qclient.search(
            collection_name = os.getenv("QDRANT_COLLECTION_NAME"),
            query_vector = query_embedding,
            limit = top_k
        )

        # for i, hit in enumerate(results):
        #     print(f"Result {i+1}:")
        #     print(f"Question: {hit.payload['question']}")
        #     print(f"Response: {hit.payload['response']}")
        #     print(f"Score: {hit.score}\n")
        #     print("-----")
        
        return results
    except Exception as e:
        print(f"Error retrieving from Qdrant: {e}")
        return None

# if __name__ == "__main__":
    # Store_Excel_to_Qdrant()
    # Store_Doc_to_Qdrant(os.getenv("PATH_TO_DOC"))
    # query = "what are the measures for the protection of data during storage."
    # retrieve_from_qdrant(query)
    # print("Data successfully stored in Qdrant.")

