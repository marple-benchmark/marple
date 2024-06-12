SYSTEM_MESSAGES = {
    "system_message_1": """You are an expert computer scientist and programmer, especially skilled at extracting information from scene graphs.""",
}


PROMPT_TEMPLATES = {
    "0shot_cot": """Take a deep breath. Your task is to analyze and determine which agent (target agent, other agent) is more likely to have performed specific actions leading to the final state of the environment. This decision should be based on comparing the changes between the initial and current states of both agents.

Remember, the states you are analyzing are select snapshots from a larger sequence. If the agents have gone through e.g., 100 states, you might only be seeing a fraction of these (like every 10th state for each agent), which means critical movements and decisions may have occurred in the unseen states.

Initial State of Target Agent:
{state_0_agent_target}

Current State of Target Agent:
{current_state_target}

Initial State of Other Agent:
{state_0_agent_other}

Current State of Other Agent:
{current_state_other}

Final State - Unknown Agent:
{final_state}

Your analysis should consider how the changes and progression from the initial to the current state for each agent might indicate their likely actions in the final state. Reflect on the sequence of events and decisions made by each agent and how the state changes relate to attributes of the agent in the final state. When an agent is in the exact same state as the agent in the final state, we can conclude that the mission has been completed.

Based on analyzing the changes between the initial and current states, and the final state, you must answer the following question about the final state: ***{question}***

Answer Options:
Provide an integer between 0 - 100 (where 0 = definitely Other Agent and 100 = definitely Target Agent).

Strictly follow this response format:
Reasoning: [your detailed 'Let's think step-by-step...' analysis of the changes between the initial and current states for each agent and how these changes might inform their behavior in the final state]
Answer: [insert your answer as an integer between 0 and 100 here]""",
}