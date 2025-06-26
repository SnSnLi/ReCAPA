import torch
import openai
from openai import OpenAI
from typing import List
from collections import namedtuple
import json
import concurrent.futures
import time
from .prompts import get_contrastive_trajectory_prompt
from .encoder import NomicEncoder
import os

MultiTransition = namedtuple('MultiTransition', ['agent_id', 'state', 'action', 'reward'])

# # For Embodied-Agent-Interface evaluation
# # The original prompt is saved here for reference.
# _SYSTEM_PROMPT_EAI_ORIGINAL = """You are an expert agent controlling a robot in a virtual home. Your task is to generate a sequence of actions to accomplish a high-level goal.

# You MUST provide your response as a single, continuous string of actions separated by " then ".
# Do NOT use JSON, Markdown, or any other formatting.

# CRITICAL: The actions in the sequence MUST follow the format `ACTION_NAME object_name`.
# - `ACTION_NAME` must be one of the allowed uppercase actions.
# - `object_name` should be the object or location.
# - Do NOT include IDs, numbers, or parentheses like `(plate,1)`.

# Allowed Actions:
# - CLOSE
# - CUT
# - DRINK
# - EAT
# - FIND
# - GRAB
# - LIE
# - LOOKAT
# - MOVE
# - OPEN
# - PULL
# - PUSH
# - PUTIN
# - PUTON
# - READ
# - RINSE
# - SCRUB
# - SIT
# - SLEEP
# - SQUEEZE
# - STANDUP
# - SWITCHOFF
# - SWITCHON
# - TOUCH
# - TURNTO
# - TYPE
# - WALK
# - WASH
# - WAKEUP

# Example for the goal "wash a plate":
# FIND plate then WALK sink then GRAB plate then WASH plate then RINSE plate

# Now, generate the action sequence for the given high-level task.
# """

# NEW PROMPT - More explicit about allowed actions to reduce hallucination
# SYSTEM_PROMPT_EAI = """You are an expert agent controlling a robot in a virtual home. Your task is to generate a sequence of actions to accomplish a high-level goal.
#
# You MUST provide your response as a single, continuous string of actions separated by " then ".
# Do NOT use JSON, Markdown, or any other formatting.
#
# VERY IMPORTANT: You may ONLY use actions from the 'Allowed Actions' list below. Do NOT invent new actions like 'TAKE' or 'GOTO'. Using any action not on this list will result in failure.
#
# CRITICAL: The actions in the sequence MUST follow the format `ACTION_NAME object_name`.
# - `ACTION_NAME` must be one of the allowed uppercase actions.
# - `object_name` should be the object or location.
# - Do NOT include IDs, numbers, or parentheses like `(plate,1)`.
#
# Allowed Actions:
# - CLOSE
# - CUT
# - DRINK
# - EAT
# - FIND
# - GRAB
# - LIE
# - LOOKAT
# - MOVE
# - OPEN
# - PULL
# - PUSH
# - PUTIN
# - PUTON
# - READ
# - RINSE
# - SCRUB
# - SIT
# - SLEEP
# - SQUEEZE
# - STANDUP
# - SWITCHOFF
# - SWITCHON
# - TOUCH
# - TURNTO
# - TYPE
# - WALK
# - WASH
# - WAKEUP
#
# Example for the goal "wash a plate":
# FIND plate then WALK sink then GRAB plate then WASH plate then RINSE plate
#
# Now, generate the action sequence for the given high-level task.
# """

SYSTEM_PROMPT_GOAL_INTERPRETATION = """
Your task is to understand natural language goals for a household robot, reason about the object states and relationships, and turn natural language goals into symbolic goals in the given format. The goals include: node goals describing object states, edge goals describing object relationships and action goals describing must-to-do actions in this goal. The input will be the goal's name, the goal's description, relevant objects as well as their current and all possible states, and all possible relationships between objects. The output should be the symbolic version of the goals.


Relevant objects in the scene indicates those objects involved in the action execution initially. It will include the object name, the object initial states, and the object all possible states. It follows the format: object name, id: ...(object id), states: ...(object states), possible states: ...(all possible states). Your proposed object states should be within the following set: CLOSED, OPEN, ON, OFF, SITTING, DIRTY, CLEAN, LYING, PLUGGED_IN, PLUGGED_OUT.


All possible relationships are the keys of the following dictionary, and the corresponding values are their descriptions:
{'ON': 'An object rests atop another, like a book on a table.', 'FACING': 'One object is oriented towards another, as in a person facing a wall.', 'HOLDS_LH': 'An object is held or supported by the left hand, like a left hand holding a ball.', 'INSIDE': 'An object is contained within another, like coins inside a jar.', 'BETWEEN': 'An object is situated spatially between two entities, like a park between two buildings.', 'HOLDS_RH': 'An object is grasped or carried by the right hand, such as a right hand holding a pen.', 'CLOSE': 'Objects are near each other without touching, like two close-standing trees.'}


Symbolic goals format:
Node goals should be a list indicating the desired ending states of objects. Each goal in the list should be a dictionary with two keys 'name' and 'state'. The value of 'name' is the name of the object, and the value of 'state' is the desired ending state of the target object. For example, [{'name': 'washing_machine', 'state': 'PLUGGED_IN'}, {'name': 'washing_machine', 'state': 'CLOSED'}, {'name': 'washing_machine', 'state': 'ON'}] requires the washing_machine to be PLUGGED_IN, CLOSED, and ON. It can be a valid interpretation of natural language goal: 
Task name: Wash clothes. 
Task description: Washing pants with washing machine
This is because if one wants to wash clothes, the washing machine should be functioning, and thus should be PLUGGED_IN, CLOSED, and ON.

Edge goals is a list of dictionaries indicating the desired relationships between objects. Each goal in the list is a dictionary with three keys 'from_name', and 'relation' and 'to_name'. The value of 'relation' is desired relationship between 'from_name' object to 'to_name' object. The value of 'from_name' and 'to_name' should be an object name. The value of 'relation' should be an relationship. All relations should only be within the following set: ON, INSIDE, BETWEEN, CLOSE, FACING, HOLDS_RH, HOLDS_LH.

Each relation has a fixed set of objects to be its 'to_name' target. Here is a dictionary where keys are 'relation' and corresponding values is its possible set of 'to_name' objects:
{'ON': {'couch', 'washing_machine', 'bed', 'coffe_maker', 'table', 'dishwasher', 'toilet', 'oven', 'character'}, 'HOLDS_LH': {'keyboard', 'tooth_paste', 'water_glass', 'toothbrush', 'spectacles', 'novel'}, 'HOLDS_RH': {'address_book', 'phone', 'tooth_paste', 'drinking_glass', 'water_glass', 'cup', 'toothbrush', 'mouse', 'remote_control', 'novel'}, 'INSIDE': {'dining_room', 'home_office', 'freezer', 'hands_both', 'bathroom'}, 'FACING': {'phone', 'television', 'laptop', 'computer', 'remote_control', 'toilet'}, 'CLOSE': {'cat', 'shower'}}

Action goals is a list of actions that must be completed in the goals. The number of actions is less than three. If node goals and edge goals are not enough to fully describe the goal, add action goals to describe the goal. Below is a dictionary of possible actions, whose keys are all possible actions and values are corresponding descriptions. When output actions goal list, each action goal should be a dictionary with keys 'action' and 'description'.
{'CLOSE': 'as opposed to open sth, CLOSE sth means changing the state from OPEN to CLOSE, not get close to!', 'DRINK': 'drink up sth', 'FIND': 'find and get near to sth', 'WALK': 'walk towards sth, get near to sth', 'GRAB': 'graph sth', 'LOOKAT': 'look at sth, face sth', 'LOOKAT_SHORT': 'shortly look at sth', 'LOOKAT_LONG': 'look at sth for long', 'OPEN': 'open sth, as opposed to close sth', 'POINTAT': 'point at sth', 'PUTBACK': 'put object A back to object B', 'PUTIN': 'put object A into object B', 'PUTOBJBACK': 'put object back to its original place', 'RUN': 'run towards sth, get close to sth', 'SIT': 'sit on sth', 'STANDUP': 'stand up', 'SWITCHOFF': 'switch sth off (normally lamp/light)', 'SWITCHON': 'switch sth on (normally lamp/light)', 'TOUCH': 'touch sth', 'TURNTO': 'turn and face sth', 'WATCH': 'watch sth', 'WIPE': 'wipe sth out', 'PUTON': 'put on clothes, need to hold the clothes first', 'PUTOFF': 'put off clothes', 'GREET': 'greet to somebody', 'DROP': "drop something in robot's current room, need to hold the thing first", 'READ': 'read something, need to hold the thing first', 'LIE': 'lie on something, need to get close the thing first', 'POUR': 'pour object A into object B', 'TYPE': 'type on keyboard', 'PUSH': 'move sth', 'PULL': 'move sth', 'MOVE': 'move sth', 'WASH': 'wash sth', 'RINSE': 'rinse sth', 'SCRUB': 'scrub sth', 'SQUEEZE': 'squeeze the clothes', 'PLUGIN': 'plug in the plug', 'PLUGOUT': 'plug out the plug', 'CUT': 'cut some food', 'EAT': 'eat some food', 'RELEASE': 'drop sth inside the current room'}

Now output the symbolic version of the goal. Output in json format, whose keys are 'node goals', 'edge goals', and 'action goals', and values are your output of symbolic node goals, symbolic edge goals, and symbolic action goals, respectively. That is, {'node goals': SYMBOLIC NODE GOALS, 'edge goals': SYOBOLIC EDGE GOALS, 'action goals': SYMBOLIC ACTION GOALS}. Please strictly follow the symbolic goal format.
"""

SYSTEM_PROMPT_SUBGOAL = """
You are an expert planner for a household robot. Your task is to decompose a high-level goal into a sequence of logical intermediate states, or "subgoals".

You will be given the target task, a list of relevant objects, the initial state of the world, and the final goal states.

You must generate a JSON object with a single key "subgoals". The value of "subgoals" must be a list of strings. Each string represents a single, achievable intermediate state (a subgoal) in a logical order. The sequence of subgoals should form a coherent plan to get from the initial state to the goal state.

Example Input (simplified for clarity):
# Target Task: Make coffee
## Initial States
...
## Goal States
[States]
FILLED(coffeemaker)
ON(coffeemaker)

Example Output:
{
  "subgoals": [
    "ONTOP(mug, coffeemaker)",
    "FILLED(coffeemaker)",
    "ON(coffeemaker)"
  ]
}

Now, generate the subgoal plan for the following task.
"""

# A new, stricter prompt for goal interpretation to enforce JSON output.
SYSTEM_PROMPT_GOAL_INTERPRETATION_EAI = """
You are a task-oriented system that converts natural language goals for a robot into a structured JSON format.
You will be provided with a `Task name`, `Task description`, `Relevant objects` with their states, and a list of `All possible relationships`.

You MUST output a single JSON object. This JSON object must have three keys: "node_goals", "edge_goals", and "action_goals".
- "node_goals": A list of dictionaries, where each dictionary has "name" and "state" keys.
- "edge_goals": A list of dictionaries, where each dictionary has "from_name", "relation", and "to_name" keys.
- "action_goals": A list of dictionaries, where each dictionary has "action" and "description" keys.

Do NOT include any text, explanations, or markdown formatting before or after the JSON object.

Example Input:
- Task name: Wash clothes.
- Task description: Washing pants with washing machine

Example Output:
{
  "node_goals": [
    {"name": "washing_machine", "state": "PLUGGED_IN"},
    {"name": "washing_machine", "state": "CLOSED"},
    {"name": "washing_machine", "state": "ON"}
  ],
  "edge_goals": [],
  "action_goals": []
}
"""

SYSTEM_PROMPT_SUBGOAL_DECOMPOSITION_EAI = """
You are a task-oriented system that decomposes a high-level goal for a robot into a sequence of subgoals.
You will be provided with a `Target Task`, `Initial States`, and `Goal States`.

You MUST output a single JSON object. This JSON object must have a single key: "subgoals".
The value of "subgoals" must be a list of strings. Each string is a subgoal representing an intermediate state.
Do NOT include any text, explanations, or markdown formatting before or after the JSON object.

Example Input:
- Target Task: Make coffee
- Initial States: ...
- Goal States: [FILLED(coffeemaker), ON(coffeemaker)]

Example Output:
{
  "subgoals": [
    "ONTOP(mug, coffeemaker)",
    "FILLED(coffeemaker)",
    "ON(coffeemaker)"
  ]
}
"""

class LLMTrajectory:
    """
    This class serves as a client for interacting with OpenAI's GPT models
    to generate reinforcement learning trajectories. It handles the API calls,
    manages prompts, and formats the responses into usable data structures.

    Key functionalities include:
    - Initializing the OpenAI client with necessary credentials.
    - Generating a single action sequence from a high-level task description.
    - Generating multiple "contrastive" trajectories for more complex learning scenarios.
    - Formatting the raw API output into tensors and transition objects.
    - Robust retry mechanisms to handle transient API errors.
    - Parallel API calls for efficient generation of multiple trajectories.

    The class is designed to be a modular component in a larger RL framework,
    abstracting away the complexities of direct API interaction.
    """
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        api_key: str = None,
        model_name: str = "gpt-4-1106-preview",
        api_base: str = None,
        num_agents: int = 1,
        hidden_dim: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initializes the LLMTrajectory client.

        :param state_dim: Dimensionality of the state space.
        :param action_dim: Dimensionality of the action space.
        :param api_key: OpenAI API key.
        :param model_name: The name of the GPT model to use.
        :param api_base: The base URL for the OpenAI API, for custom deployments.
        :param num_agents: The number of agents in the environment.
        :param hidden_dim: The hidden dimension size for internal representations.
        :param device: The computing device ('cpu' or 'cuda').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim + 1
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.client = None
        self._initialize_client()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.device = device
        self.traj_lengths = {
            'low': (2, 4),
            'mid': (5, 8),
            'high': (9, 12)
        }

    def _initialize_client(self):
        """Initializes and returns the OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)

    def generate_trajectory_from_prompt(
        self, 
        prompt_text: str, 
        task_type: str = "action_sequencing"
    ) -> str:
        """
        Generates a trajectory from a given text prompt, handling API calls and retries.
        
        Args:
            prompt_text: The full prompt to send to the model.
            task_type: The type of task, one of 'action_sequencing', 
                       'goal_interpretation', or 'subgoal_decomposition'.
                       This determines the system prompt used.

        Returns:
            The raw string output from the language model. If generation or 
            JSON parsing fails for EAI tasks, returns an empty dict string '{}'.
        """
        system_prompt = ""
        if task_type == "action_sequencing":
            system_prompt = SYSTEM_PROMPT_EAI
        elif task_type == "goal_interpretation":
            system_prompt = SYSTEM_PROMPT_GOAL_INTERPRETATION_EAI
        elif task_type == "subgoal_decomposition":
            system_prompt = SYSTEM_PROMPT_SUBGOAL_DECOMPOSITION_EAI
        if task_type == "action_sequencing":
            system_prompt = SYSTEM_PROMPT_EAI
        elif task_type == "goal_interpretation":
            system_prompt = SYSTEM_PROMPT_GOAL_INTERPRETATION_EAI
        elif task_type == "subgoal_decomposition":
            system_prompt = SYSTEM_PROMPT_SUBGOAL_DECOMPOSITION_EAI

        max_retries = 5
        retry_delay = 10  # seconds
        for attempt in range(max_retries):
            try:
                # --- DEBUGGING: Print API call details ---
                print("\n" + "="*50)
                print(f"Attempt {attempt + 1}/{max_retries} - Calling DeepSeek API with details:")
                print(f"  - Model Name: {self.model_name}")
                print(f"  - API Base URL: {self.api_base}")
                print(f"  - API Key (last 4 chars): ...{self.api_key[-4:] if self.api_key else 'None'}")
                
                user_message = f"High-level task: {prompt_text}"
                messages_to_send = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
                print("  - Messages Payload:")
                print(json.dumps(messages_to_send, indent=2))
                print("="*50 + "\n")
                # --- END DEBUGGING ---

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_to_send,
                    temperature=0,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                response_text = response.choices[0].message.content
                
                if not response_text or not response_text.strip():
                    raise ValueError("API returned empty or whitespace-only content.")

                # For EAI tasks, the output should be a valid JSON object.
                # We'll validate it here. If it's not valid, we'll return an empty
                # dict string to signify an error to the calling code.
                if task_type in ['goal_interpretation', 'subgoal_decomposition']:
                    try:
                        # Attempt to parse the JSON to see if it's valid.
                        json.loads(response_text)
                    except json.JSONDecodeError:
                        print(f"Warning: LLM output is not valid JSON for task {task_type}. Content: '{response_text}'")
                        return "{}" # Return empty dict string on failure
                
                return response_text
            except (openai.APIError, openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError) as e:
                print(f"OpenAI API error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        return "{}" if task_type in ['goal_interpretation', 'subgoal_decomposition'] else ""

    def _simulate_llm_call(
        self,
        current_traj: torch.Tensor,
        env_description: str,
        strategy_context: str,
        level: str,
    ) -> tuple:
        """
        Generates a single trajectory and its logprobs by calling the API.
        The prompt is modified to request a single trajectory to ensure logprobs
        correspond to a single logical unit.
        Includes a retry mechanism for robustness.

        :return: A tuple containing (raw_trajectory, logprobs_content)
        """
        if level not in self.traj_lengths:
            raise ValueError(f"Invalid level: {level}. Expected 'low', 'mid', or 'high'.")

        min_len, max_len = self.traj_lengths[level]
        traj_len = torch.randint(min_len, max_len + 1, (1,)).item()

        prompt = get_contrastive_trajectory_prompt(
            current_traj=current_traj.tolist(),
            env_description=env_description,
            strategy_context=strategy_context,
            level=level,
            num_trajs=1,
            traj_len=traj_len,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

        retries = 3
        delay = 5  # seconds
        last_exception = None

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert in generating RL trajectories in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    logprobs=True,
                    max_tokens=2000,
                    temperature=0.7
                )

                response_content = response.choices[0].message.content
                if not response_content or response_content.strip() == "":
                    raise ValueError("API returned empty or whitespace-only content.")

                response_data = json.loads(response_content)
                
                raw_trajs = response_data.get('trajectories', [])
                if not isinstance(raw_trajs, list) or len(raw_trajs) != 1:
                    raise ValueError(f"API did not return exactly one trajectory. Found: {len(raw_trajs)}")

                logprobs_content = response.choices[0].logprobs.content
                return raw_trajs[0], logprobs_content

            except (openai.APIError, ValueError, json.JSONDecodeError) as e:
                last_exception = e
                print(f"Warning: API call or parsing failed on attempt {attempt + 1}/{retries}. Error: {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Error: All retry attempts failed.")
                    print(f"--- Begin Failed Prompt ---")
                    print(prompt)
                    print(f"--- End Failed Prompt ---")

        raise RuntimeError(f"API call or parsing failed after {retries} retries.") from last_exception

    def format_trajectory(
        self,
        raw_traj: List[dict],
        agent_id: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> tuple:
        """
        Format raw LLM trajectory into MultiTransition and tensor.
        :param raw_traj: List of dicts (state, action, reward).
        :param agent_id: Agent ID.
        :param device: Device for tensors.
        :return: (List[MultiTransition], torch.Tensor) for RCE.
        """
        transitions = []
        tensor_list = []
        for step in raw_traj:
            state = torch.tensor(step['state'], dtype=torch.float32, device=device)
            action = torch.tensor(step['action'], dtype=torch.float32, device=device)
            reward = torch.tensor(step['reward'], dtype=torch.float32, device=device)
            transitions.append(MultiTransition(
                agent_id=agent_id,
                state=state,
                action=action,
                reward=reward
            ))
            tensor_list.append(torch.cat([state, action, reward.view(1)]))

        traj_tensor = torch.stack(tensor_list) if tensor_list else torch.zeros((0, self.input_dim), device=device)
        return transitions, traj_tensor

    def generate_contrastive_trajectories(
        self,
        current_traj: torch.Tensor,
        agent_id: int,
        num_trajs: int = 3,
        level: str = 'low',
        env_description: str = "Multi-agent RL environment",
        strategy_context: str = "Current strategy for agent"
    ) -> List[tuple]:
        """
        Generate contrastive trajectories and their logprobs in parallel using OpenAI GPT API.
        This now returns a list of tuples, where each tuple is (trajectory_tensor, logprobs).
        """
        device = current_traj.device
        trajectories_with_logprobs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_trajs) as executor:
            future_to_traj = {
                executor.submit(
                    self._simulate_llm_call,
                    current_traj=current_traj,
                    env_description=env_description,
                    strategy_context=strategy_context,
                    level=level
                ): i for i in range(num_trajs)
            }

            for future in concurrent.futures.as_completed(future_to_traj):
                try:
                    raw_traj, logprobs = future.result()

                    _, traj_tensor = self.format_trajectory(raw_traj, agent_id, device)
                    if traj_tensor.numel() > 0:
                        trajectories_with_logprobs.append((traj_tensor, logprobs))
                except Exception as exc:
                    print(f'Warning: A trajectory generation task failed with an exception: {exc}')

        return trajectories_with_logprobs