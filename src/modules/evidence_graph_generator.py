from ..states.PromptState import PromptState
from llama_index.core import Document, KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import torch
import os
from dotenv import load_dotenv

_ = load_dotenv()

# llm = HuggingFaceInferenceAPI(
#     model_name="mistralai/Mistral-7B-Instruct-v0.2", token=dotenv_values(".env")["HF"]
# )

llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ"))

embedder = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2",
    device="mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu",
)

Settings.llm = llm
Settings.embed_model = embedder


def generate_evidence_graph(rag_response, webrag_response):
    # start creating the evidence graph
    documents = [Document(text=rag_response), Document(text=webrag_response)]
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex.from_documents(
        documents=documents,
        max_triplets_per_chunk=3,
        storage_context=storage_context,
        embed_model=embedder,
        include_embeddings=True,
        llm=llm,
    )

    storage_context.persist()

    return index


def retrieve_nodes_from_evidence_graph(state: PromptState):
    # get responses from RAG and WebRAG
    rag_response = state["rag_response"]
    webrag_response = state["webrag_response"]

    # extract and concatenate all the content from the WebRAG response
    content_list_web_rag_retrieval = [
        webrag_response[i]["content"].strip().lower()
        for i in range(len(webrag_response))
    ]
    webrag_response = "".join(
        f"Content: {entry}\n" for entry in content_list_web_rag_retrieval
    )

    index = generate_evidence_graph(rag_response, webrag_response)
    query_engine = index.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid",
        similarity_top_k=5,
        llm=llm,
    )

    query = state["input_prompt"]
    response = query_engine.query(query)

    print(response)

    return {"eg_response": response}
