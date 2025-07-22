from states.PromptState import PromptState

import time

def cot_tasks(state: PromptState, llm, max_steps=5, wait_time=1, list_length=0):
    prompt = state["input_prompt"]
    
    system_prompt = f"""
    System Prompts:
    Q:= Compare the privacy policy of 2 companies Company A and Company B.
    A:= 1) Fetch the privacy policy of Company A.
        2) Fetch the privacy policy of Company B.
        3) Compare the privacy policy of Company A and Company B.
    Q:= Give the Protection of Information from the privacy policy of Company A.
    A:= 1) Fetch the privacy policy of Company A.
        2) Give the Protection of Information from the privacy policy of Company A.
    Q:= List down the Information Required to Detect Violations in the privacy policy of Company A.
    A:= 1) Fetch the privacy policy of Company A.
        2) List down the Information Required to Detect Violations in the privacy policy of Company A.
    Just give the points and do not include question prompt and A:= tag.
    """
    initial_prompt = f"""Main Task: {prompt}\nBreak this task into a list of smaller subtasks. Give only the names of the smaller subtasks and not any description of that subtask. Just give the subtasks in as least number of points as possible. Give the answer following the pattern given in System Prompts.\n"""

    thought_process = system_prompt + "\n" + initial_prompt

    modified_prompt = thought_process
    modified_thought = llm.ask(modified_prompt)
    subtasks = modified_thought.strip().split("\n")
    subtasks = [subtask.strip().lower() for subtask in subtasks if subtask.strip()]
    list_length = len(subtasks)

    for _ in range(max_steps):
        modified_prompt = thought_process
        try:
            modified_thought = llm.ask(modified_prompt)
        except:
            time.sleep(wait_time)  # Wait for wait_time seconds before retrying
            modified_thought = llm.ask(modified_prompt)
        temp_list = modified_thought.strip().split("\n")
        temp_list = [t_list.strip().lower() for t_list in temp_list if t_list.strip()]
        temp_list_length = len(temp_list)
        if temp_list_length < list_length:
            subtasks = temp_list
            list_length = temp_list_length
        else:
            pass
        
    # update the subtasks in the PromptState
    all_subtasks = state.get("subtasks", []) + subtasks
    
    print(subtasks)

    return { "subtasks": all_subtasks }