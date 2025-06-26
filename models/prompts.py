from typing import List

def get_contrastive_trajectory_prompt(
    current_traj: List,
    env_description: str,
    strategy_context: str,
    level: str,
    num_trajs: int,
    traj_len: int,
    state_dim: int,
    action_dim: int
) -> str:
    """
    Generate a formatted prompt for OpenAI GPT API to produce contrastive trajectories.
    :param current_traj: Current trajectory as a list of steps (state, action, reward).
    :param env_description: Description of the RL environment.
    :param strategy_context: Context of the agent's current strategy.
    :param level: HRN level ('low' for 5-10 steps, 'mid' for 50-100, 'high' for ~500).
    :param num_trajs: Number of failure trajectories to generate.
    :param traj_len: Number of steps per trajectory.
    :param state_dim: Dimension of state vectors.
    :param action_dim: Dimension of action vectors.
    :return: Formatted prompt string for GPT API.
    """
    prompt = (
        f"Generate {num_trajs} failure trajectories for an agent in a multi-agent reinforcement learning environment.\n"
        f"**Environment**: {env_description}\n"
        f"**Level**: {level} (each trajectory should have {traj_len} steps)\n"
        f"**Current Trajectory**: {current_traj}\n"
        f"**Strategy Context**: {strategy_context}\n"
        f"**Requirements**:\n"
        f"- Each trajectory must have exactly {traj_len} steps.\n"
        f"- Each step must include:\n"
        f"  - 'state': a list of {state_dim} floats (agent's state, e.g., position, velocity).\n"
        f"  - 'action': a list of {action_dim} floats (agent's action, e.g., movement direction).\n"
        f"  - 'reward': a negative scalar float in [-10, 0] (indicating failure).\n"
        f"**Output Format**: Return a JSON object with a 'trajectories' key, containing a list of {num_trajs} trajectories.\n"
        f"Each trajectory is a list of {traj_len} step objects with 'state', 'action', and 'reward' keys.\n"
        f"**Example Output**:\n"
        f'{{\n'
        f'  "trajectories": [\n'
        f'    [\n'
        f'      {{"state": [0.1, 0.2], "action": [0.3], "reward": -1.0}},\n'
        f'      {{"state": [0.2, 0.3], "action": [0.4], "reward": -2.0}}\n'
        f'    ],\n'
        f'    [\n'
        f'      {{"state": [0.0, 0.1], "action": [0.2], "reward": -1.5}},\n'
        f'      {{"state": [0.1, 0.2], "action": [0.3], "reward": -3.0}}\n'
        f'    ]\n'
        f'  ]\n'
        f'}}'
    )
    return prompt

# # For action sequencing
# ACTION_SEQUENCING_PROMPT = """You are an expert in household task planning.
# Your goal is to decompose a high-level instruction into a sequence of executable low-level actions.
# The available actions are:
# - WALK <location> <id>
# - GRAB <object> <id>
# - OPEN <object> <id>
# - CLOSE <object> <id>
# - PUTIN <object1> <id1> <object2> <id2>
# - SWITCHON <object> <id>
# - SWITCHOFF <object> <id>
# - LOOKAT <object> <id>
# - TYPE <object> <id>
# - FIND <object> <id>
# - TOUCH <object> <id>
# - TURNTO <object> <id>
# - RINSE <object> <id>
# - WASH <object> <id>
# - DRINK <object> <id>
# - PUTON <object> <id>
# - STANDUP

# Instruction: {instruction}

# Decompose the instruction into a sequence of actions.
# Provide the output as a JSON object where each key is an action and the value is a list of arguments, like this:
# {{
#   "WALK": ["location", "id"],
#   "GRAB": ["object", "id"],
#   "SWITCHON": ["object", "id"]
# }}

# Do not add any explanation or comments. Return only the JSON object.
# """

# # For subgoal decomposition
# SUBGOAL_DECOMPOSITION_PROMPT = """You are an expert household task planner.
# Your goal is to decompose a high-level instruction into a sequence of high-level subgoals.
# Do not generate specific, low-level executable actions. Instead, provide a logical plan of intermediate steps.

# For example, if the instruction is "put the apple in the fridge", a good decomposition would be:
# ["find the apple", "go to the fridge", "open the fridge", "put the apple inside", "close the fridge"]

# Instruction: {instruction}

# Decompose the instruction into a sequence of subgoals.
# Provide the output as a Python list of strings, like this:
# ["subgoal 1", "subgoal 2", ...]
# Do not add any explanation or comments.
# """

# # REFLECTION_HEAD_PROMPT = """You are a critic for a robot's plans.
# # You will be given a high-level instruction and a sequence of sub-goals or actions to achieve it.
# # Your task is to identify if the plan is flawed and, if so, pinpoint the exact step where the plan deviates from a logical and successful path.
# # A deviation point is the first step in the plan that is illogical, inefficient, impossible, or misses a crucial prerequisite.

# # Consider the following:
# # - Are the steps in a logical order? (e.g., you must 'open' the fridge before you can 'put' something in it)
# # - Does the plan miss any obvious necessary steps?
# # - Are any steps impossible or nonsensical?
# # - Is the plan overly complicated or inefficient?

# # Instruction: {instruction}
# # Plan: {plan}

# # Analyze the plan. If the plan is perfect, respond with -1.
# # If the plan is flawed, respond with a single integer: the 0-indexed number of the step where the deviation begins.
# # Do not provide any explanation, only the integer.
# # """