from .web_search import scrape_stackexchange, get_full_text, search
from .build_rag_database import SentenceTransformerEmbeddings
from transformers import pipeline
import numpy as np
import re
from ..states.PromptState import PromptState

embedder = SentenceTransformerEmbeddings()


def web_rag_retrieval(subtasks):
    """Retrieval relevant context from the internet"""
    methods = ["Google", "Wikipedia", "ConsumerFinance"]
    idx = 0
    result_dict = {
        "Google": [],
        "Wikipedia": [],
        "StackExchange": [],
        "ConsumerFinance": [],
    }
    query = subtasks[-1]
    stack_result = scrape_stackexchange(query, 3)
    for r in stack_result:
        try:
            if "link" in r:
                substring = r"stack exchange network consists of 183 q&a communities including stack overflow, the largest, most trusted online community for developers to learn, share their knowledge, and build their careers. now available on stack overflow for teams! ai features where you work: search, ide, and chat.\s+ask questions, find answers and collaborate at work with stack overflow for teams.\s+explore teams teams q&a for work connect and share knowledge within a single location that is structured and easy to search."
                text = (
                    get_full_text(r["link"]).lower().replace("\t", "").replace("\n", "")
                )
                text = re.sub(substring, "", text)
                result_dict["StackExchange"].append([r["title"], str(text)])
        except Exception:
            print("Error")
            result_dict["StackExchange"].append(["", ""])

    for task in subtasks:
        query = task
        result = search(query, 5)
        for kth_result in result:
            for r in kth_result:
                try:
                    if "link" in r:
                        text = (
                            get_full_text(r["link"])
                            .lower()
                            .replace("\t", "")
                            .replace("\n", "")
                        )
                    else:
                        text = r["content"]
                    if idx < 2:
                        result_dict[methods[idx]].append([r["link"], str(text)])
                    else:
                        result_dict[methods[idx]].append([r["title"], str(text)])
                except Exception:
                    print("Error")
                    result_dict[methods[idx]].append(["", ""])

            idx = (idx + 1) % len(methods)

    return result_dict


# below is the code to encode and rank the retrieved text
def filter_by_sentiment(texts: list[str], sentiment_threshold: float = 0.1):
    """
    Filter texts based on sentiment score
    Returns indices of texts with positive sentiment above the threshold
    """
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    positive_indices = []
    for idx, text in enumerate(texts):
        text = text[:512]
        result = sentiment_analyzer(text)
        if (
            result[0]["label"] == "POSITIVE"
            and result[0]["score"] > sentiment_threshold
        ):
            positive_indices.append(idx)

    return positive_indices


def prepare_search_data(results_dict, sentiment_threshold: float):
    """
    Prepare flattened lists of texts, URLs, and sources from the nested dictionary
    """
    texts, urls, sources = [], [], []

    for source, url_content_pairs in results_dict.items():
        for url, content in url_content_pairs:
            texts.append(content)
            urls.append(url)
            sources.append(source)

    positive_indices = filter_by_sentiment(texts, sentiment_threshold)

    filtered_texts = [texts[i] for i in positive_indices]
    filtered_urls = [urls[i] for i in positive_indices]
    filtered_sources = [sources[i] for i in positive_indices]

    return filtered_texts, filtered_urls, filtered_sources


def search_similar(query, embeddings, top_k):
    """
    Search for similar texts using cosine similarity
    """
    query_vector = np.array(embedder.embed_query(query))
    # cosine similarity: dot product since embeddings are normalized
    similarities = np.dot(embeddings, query_vector.T).squeeze()
    top_indices = similarities.argsort()[::-1][:top_k]
    top_scores = similarities[top_indices]
    return top_scores, top_indices


def format_results(
    scores,
    indices,
    texts,
    urls,
    sources,
):
    """
    Format search results into a list of dictionaries including URLs
    """
    results = []
    for score, idx in zip(scores, indices):
        results.append(
            {
                "source": sources[idx],
                "url": urls[idx],
                "content": texts[idx],
                "similarity_score": float(score),
            }
        )
    return results


def aggregate_and_rank_results(
    state: PromptState,
    top_k=5,
    sentiment_threshold=0.1,
):
    """
    Main function to process query and results using cosine similarity with sentiment filtering
    """
    query = state["input_prompt"]
    subtasks = state["subtasks"]

    results_dict = web_rag_retrieval(subtasks)

    texts, urls, sources = prepare_search_data(results_dict, sentiment_threshold)

    if not texts:
        print("No texts passed the sentiment threshold!")
        return []

    # Create embeddings
    embeddings = embedder.embed_documents(texts)

    # Perform search
    scores, indices = search_similar(query, embeddings, min(top_k, len(texts)))

    # Format results
    results = format_results(scores, indices, texts, urls, sources)

    return {"webrag_response": results}
