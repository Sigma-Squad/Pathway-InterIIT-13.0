from .model import Model
from langgraph.graph import StateGraph, START, END
from .states.PromptState import PromptState
from .modules.CoT import cot_tasks
from .modules.database_retriever import DBR
from .modules.web_retriever import aggregate_and_rank_results
from .modules.evidence_graph_generator import retrieve_nodes_from_evidence_graph
from .modules.final_response_generator import final_response
from functools import partial
from time import perf_counter

llm = Model()

# pass the llm to the functions
cot = partial(cot_tasks, llm=llm)
dbr = partial(DBR, llm=llm)
rg = partial(final_response, llm=llm)

# Create the graph
graph = StateGraph(PromptState)

# Add nodes
_ = graph.add_node("CoT", cot)
_ = graph.add_node("DBR", dbr)
_ = graph.add_node("WebRAG", aggregate_and_rank_results)
_ = graph.add_node("EG", retrieve_nodes_from_evidence_graph)
_ = graph.add_node("RG", rg)

# Add edges
_ = graph.add_edge(START, "CoT")
_ = graph.add_edge("CoT", "DBR")
_ = graph.add_edge("CoT", "WebRAG")
_ = graph.add_edge("DBR", "EG")
_ = graph.add_edge("WebRAG", "EG")
_ = graph.add_edge("EG", "RG")
_ = graph.add_edge("RG", END)

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
            "eg_response": "",
            "final_response": "",
        }
    )

    print("=" * 50)
    print(output["final_response"])
    print("=" * 50)

    end_time = perf_counter()
    print(f"Response latency: {end_time - start_time}s")
