from sentence_transformers import SentenceTransformer
import numpy as np


def encode_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode texts into vector embeddings using sentence transformers
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()
