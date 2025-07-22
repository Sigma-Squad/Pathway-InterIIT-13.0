from model import Model
from langgraph.graph import StateGraph, START, END
from states.PromptState import PromptState
from modules.CoT import cot_tasks
from functools import partial
from time import perf_counter

llm = Model()

# pass the llm to the functions
cot = partial(cot_tasks, llm=llm)

# Create the graph
graph = StateGraph(PromptState)

# Add nodes
graph.add_node("CoT", cot)

# Add edges
graph.add_edge(START, "CoT")
graph.add_edge("CoT", END)

# Compile the graph
compiled_graph = graph.compile()

if __name__ == "__main__":
    input_prompt = "What information does Yahoo store?"
    
    start_time = perf_counter()
    
    output = compiled_graph.invoke({
        "input_prompt": input_prompt,
        "subtasks": [],
        "messages": []
    })
    
    end_time = perf_counter()
    print(f"{end_time - start_time}s")