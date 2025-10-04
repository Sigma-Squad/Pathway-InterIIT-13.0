from .model import Model
import torch
from langgraph.graph import StateGraph, START, END
from .states.PromptState import PromptState
from .modules.CoT import cot_tasks
from .modules.database_retriever import DBR
from functools import partial
from time import perf_counter

llm = Model()

# pass the llm to the functions
cot = partial(cot_tasks, llm=llm)
dbr = partial(DBR, llm=llm)

# Create the graph
graph = StateGraph(PromptState)

# Add nodes
_ = graph.add_node("CoT", cot)
_ = graph.add_node("DBR", dbr)

# Add edges
# _ = graph.add_edge(START, "CoT")
# _ = graph.add_edge("CoT", "DBR")
_ = graph.add_edge(START, "DBR")
_ = graph.add_edge("DBR", END)

# Compile the graph
compiled_graph = graph.compile()

if __name__ == "__main__":
    input_prompt = "What information does Buffalo Wild Wings store?"

    start_time = perf_counter()

    output = compiled_graph.invoke(
        {
            "input_prompt": input_prompt,
            "subtasks": [],
            "rag_response": "",
            "webrag_response": "",
        }
    )

    end_time = perf_counter()
    print(f"Response latency: {end_time - start_time}s")
