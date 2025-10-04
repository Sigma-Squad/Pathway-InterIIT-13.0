import torch
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import os
from typing import override

PERSIST_PATH = "./chromadb"
COLLECTION_NAME = "DynamicRAG"


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ):
        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.batch_size = batch_size

    @override
    def embed_documents(self, texts: list[str]):
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            device=self.device,
        )

        print("Created document embeddings")

        return embeddings.cpu().numpy().tolist()

    @override
    def embed_query(self, text: str):
        emb = self.model.encode([text], convert_to_tensor=True, device=self.device)
        return emb.cpu().numpy()[0].tolist()


def create_vector_db(input_dir="./corrected_scraped_data/"):
    # load the data from the directory
    docs = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                if not text.strip():
                    continue
                docs.append(text.strip())

    print("Loaded data...")

    # create a chroma client
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)  # create a collection

    # get document embeddings
    embedder = SentenceTransformerEmbeddings()
    embeds = embedder.embed_documents(docs)
    num_embeds = len(embeds)

    # insert documents in batches
    batch_size = 2000
    for i in range(0, num_embeds, batch_size):
        j = min(i + batch_size, num_embeds)
        batch_embs = embeds[i:j]
        batch_docs = docs[i:j]
        ids = [str(id) for id in range(i, j)]

        collection.add(ids=ids, embeddings=batch_embs, documents=batch_docs)

    print(f"Inserted {num_embeds} vectors into {PERSIST_PATH}/{COLLECTION_NAME}")


if __name__ == "__main__":
    create_vector_db()
