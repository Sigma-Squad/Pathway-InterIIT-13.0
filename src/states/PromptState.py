from typing import TypedDict


# State the agent workflow needs to track
class PromptState(TypedDict):
    input_prompt: str
    subtasks: list[str]  # list of subtasks to do
    rag_response: str  # response from the database retriever
    webrag_response: str  # response from the web search retriever
