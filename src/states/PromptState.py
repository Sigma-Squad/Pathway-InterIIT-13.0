from typing import TypedDict, Dict


# State the agent workflow needs to track
class PromptState(TypedDict):
    input_prompt: str
    subtasks: list[str]  # list of subtasks to do
    rag_response: str  # response from the database retriever
    webrag_response: list[Dict[str, float]]  # response from the web search retriever
    eg_response: str  # response from the evidence graph
    final_response: str
