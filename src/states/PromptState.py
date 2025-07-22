from typing import TypedDict, List

# State the agent workflow needs to track
class PromptState(TypedDict):
    input_prompt: str
    subtasks: List[str] # list of subtasks to do
    messages: List[str] # track conversation with LLM