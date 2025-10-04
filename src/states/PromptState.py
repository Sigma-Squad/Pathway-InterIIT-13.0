from typing import TypedDict


# State the agent workflow needs to track
class PromptState(TypedDict):
    input_prompt: str
    subtasks: list[str]  # list of subtasks to do
    messages: list[str]  # track conversation with LLM
