from sentence_transformer import SentenceTransformer
from transformers import pipeline
from typing import List

def encode_texts(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Encode texts into vector embeddings using sentence transformers
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()