from ..states.PromptState import PromptState


def final_response(state: PromptState, llm):
    # pass the Evidence Graph context to the LLM for the final response
    query = state["input_prompt"]
    context = state["eg_response"]

    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """

    response = llm.ask(prompt)

    return {"final_response": response}
