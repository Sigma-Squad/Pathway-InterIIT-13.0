from langchain_chroma import Chroma
from .build_rag_database import SentenceTransformerEmbeddings
from ..states.PromptState import PromptState

PERSIST_PATH = "./chromadb"
COLLECTION_NAME = "DynamicRAG"


# def debug_chroma_contents(vectordb):
#     col = (
#         vectordb._collection
#     )  # the underlying Chroma collection used by langchain_chroma
#     # get everything: documents, metadatas, ids
#     rec = col.get(include=["documents", "metadatas"])
#     docs = rec.get("documents", [])
#     metadatas = rec.get("metadatas", [])
#     ids = rec.get("ids", [])

#     print(docs)

#     # Chroma returns nested lists for batched requests; handle both shapes
#     for batch_i, batch in enumerate(docs):
#         for i, content in enumerate(batch):
#             doc_id = ids[batch_i][i] if ids and ids[batch_i] else f"{batch_i}-{i}"
#             meta = metadatas[batch_i][i] if metadatas and metadatas[batch_i] else {}
#             print(
#                 f"idx={batch_i}.{i} id={doc_id} type={type(content)} preview={repr(content)[:200]} meta={meta}"
#             )


def DBR(state: PromptState, llm, num_similar_docs=1):
    embedder = SentenceTransformerEmbeddings()

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_PATH,
        embedding_function=embedder,
    )

    # Maximum Marginal Relevance for better single document searches
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": num_similar_docs}
    )

    query = state["input_prompt"]
    # debug_chroma_contents(vectordb)

    # perform DB RAG
    retrieved_docs = retriever.invoke(query)

    prompt = f"""
    You are an AI assistant answering user queries based on some context. Use the following context to answer the user query

    Context:
        {retrieved_docs}
    User query:
        {query}

    If the retrieved context is irrelevant, do not answer the query and return an empty string.
    """
    response = llm.ask(prompt)

    print(response)

    return {"rag_response": response}
