import torch
import openai
from openai import OpenAI
from typing import List, Union
from collections import namedtuple
import json
import concurrent.futures
import time
from .prompts import get_contrastive_trajectory_prompt
from .encoder import NomicEncoder
import os
from .vectorized_prompt import VectorizedPromptGenerator, VectorMetadata, TaskContext, PromptEnhancer

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
You are a task-oriented system that converts natural language goals for a robot into a structured format.
You will be provided with a `Task name`, `Task description`, `Relevant objects` with their states, and a list of `All possible relationships`.

**CRITICAL: You MUST output ONLY a JSON object with a single key "structured_goal". DO NOT use any other keys like "node_goals", "edge_goals", "action_goals", "node goals", etc.**

The value of "structured_goal" must be a list of strings, where each string represents a predicate in the format:

- State predicates: "predicate(object.id)" (e.g., "cleaned(tray.1)", "on(light.2)")
- Spatial relation predicates: "relation(object1.id, object2.id)" (e.g., "next_to(rag.0, sink.2)", "inside(plate.1, dishwasher.3)")
- Action predicates: "action(object.id)" (e.g., "washed(plate.1)", "turned_on(light.2)")

**FORBIDDEN: Do NOT output "node_goals", "edge_goals", "action_goals", "node goals", "edge goals", or "action goals" fields. These will cause evaluation failure.**

Do NOT include any text, explanations, or markdown formatting before or after the JSON object.

Example Input:
- Task name: Wash clothes
- Task description: Washing pants with washing machine

Example Output:
{
  "structured_goal": [
    "plugged_in(washing_machine.1)",
    "closed(washing_machine.1)", 
    "on(washing_machine.1)",
    "inside(clothes.2, washing_machine.1)"
  ]
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

SYSTEM_PROMPT_TRANSITION_MODELING_EAI = """
You are an expert in PDDL (Planning Domain Definition Language) action modeling. Given a domain definition, problem file with initial and goal states, and incomplete action definitions, your task is to complete the PDDL actions by writing their preconditions and effects.

IMPORTANT OUTPUT FORMAT:
- Your response must be valid PDDL action definitions
- Complete each action's :precondition and :effect sections
- Use proper PDDL syntax with logical operators (and, or, not, when, etc.)
- Return the result as JSON with key "output" and value as the complete PDDL string
- Concatenate all actions into a single string

Example format:
{
  "output": "(:action action1\n  :parameters (?char - character ?obj - object)\n  :precondition (and (predicate1 ?char) (predicate2 ?obj))\n  :effect (and (not (predicate1 ?char)) (predicate3 ?obj))\n)\n(:action action2\n  :parameters (?char - character)\n  :precondition (predicate4 ?char)\n  :effect (predicate5 ?char)\n)"
}

Your task is to analyze the problem file (initial state and goals) and write preconditions and effects that allow the actions to transform the initial state into the goal state.
"""

SYSTEM_PROMPT_EAI = """You are an expert agent controlling a robot in a virtual home. Your task is to generate a sequence of actions to accomplish a high-level goal.

You MUST provide your response as a single, continuous string of actions separated by " then ".
Do NOT use JSON, Markdown, or any other formatting.

VERY IMPORTANT: You may ONLY use actions from the 'Allowed Actions' list below. All actions MUST be in UPPERCASE. Using any action not on this list will result in failure.

CRITICAL: The actions in the sequence MUST follow the format `[ACTION] <object> (id)`.
- `ACTION` must be one of the allowed uppercase actions wrapped in square brackets.
- `<object>` should be the object or location.
- `(id)` is the object's identifier.

Allowed Actions:
- [WALK]
- [RUN]
- [SIT]
- [STANDUP]
- [GRAB]
- [OPEN]
- [CLOSE]
- [PUTIN]
- [PUTBACK]
- [SWITCHON]
- [SWITCHOFF]
- [DRINK]
- [EAT]
- [FIND]
- [LIE]
- [LOOKAT]
- [PULL]
- [PUSH]
- [PUTON]
- [READ]
- [RINSE]
- [SCRUB]
- [SQUEEZE]
- [TOUCH]
- [TURNTO]
- [TYPE]
- [WASH]
- [WAKEUP]

DO NOT USE the following forbidden actions:
- 'move', 'move_to', 'goto', 'navigate' (use [WALK] or [RUN] instead)
- 'take', 'get' (use [GRAB] instead)
- 'place', 'put', 'put on', 'place on' (use [PUTIN] or [PUTBACK] instead)
- 'find_object' (use [FIND] instead)
- Any lowercase actions (e.g., 'walk', 'grab'). ALL actions must be UPPERCASE and in brackets.

Example for the goal "wash a plate":
[FIND] <char> (plate) then [WALK] <char> (sink) then [GRAB] <char> (plate) then [WASH] <char> (plate) then [RINSE] <char> (plate)

Now, generate the action sequence for the given high-level task.
"""

class LLMTrajectory:
    """
    This class serves as a client for interacting with OpenAI's GPT models
    to generate reinforcement learning trajectories. It handles the API calls,
    manages prompts, and formats the responses into usable data structures.

    Enhanced with vectorized prompt support for better context integration.
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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        enable_vectorized_prompts: bool = True
    ):
        """
        Initializes the LLMTrajectory client with optional vectorized prompt support.
        """
        self.state_dim = state_dim or 64  # 默认值避免None相加问题
        self.action_dim = action_dim or 8
        self.input_dim = self.state_dim + self.action_dim + 1
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.client = None
        if api_key:  # 只有提供API key时才初始化客户端
            self._initialize_client()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.device = device
        self.enable_vectorized_prompts = enable_vectorized_prompts
        
        # 初始化向量化提示支持
        if self.enable_vectorized_prompts:
            self.vector_metadata = VectorMetadata(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                state_dtype="float32",
                action_dtype="float32"
            )
            self.task_context = TaskContext(
                task_type="transition_modeling",
                environment="virtualhome",
                agent_count=self.num_agents
            )
            self.prompt_generator = VectorizedPromptGenerator(
                self.vector_metadata, 
                self.task_context
            )
        
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

    def generate_vectorized_transition_response(
        self,
        current_state: Union[torch.Tensor, List, str],
        action: Union[torch.Tensor, List, str],
        additional_context: str = ""
    ) -> str:
        """
        使用向量化提示生成transition modeling响应
        
        Args:
            current_state: 当前状态（可以是向量或文本描述）
            action: 执行的动作（可以是向量或文本描述）
            additional_context: 额外的上下文信息
            
        Returns:
            LLM生成的响应
        """
        if not self.enable_vectorized_prompts:
            raise ValueError("Vectorized prompts are not enabled")
            
        if not self.client:
            raise ValueError("API client not initialized. Please provide api_key.")
        
        # 如果输入是文本，使用增强型提示
        if isinstance(current_state, str) or isinstance(action, str):
            # 文本输入的情况，增强现有提示
            text_prompt = f"Current state: {current_state}\nAction: {action}\n{additional_context}"
            enhanced_prompt = PromptEnhancer.enhance_with_metadata(
                text_prompt, 
                self.vector_metadata, 
                self.task_context
            )
            prompt = enhanced_prompt
        else:
            # 向量输入的情况，使用专门的向量化提示
            prompt = self.prompt_generator.create_transition_prompt(
                current_state, 
                action, 
                additional_context
            )
        
        # 使用transition modeling系统提示
        system_prompt = SYSTEM_PROMPT_TRANSITION_MODELING_EAI
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1024,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in vectorized transition modeling: {e}")
            return "[]"  # 返回空数组作为fallback

    def generate_trajectory_from_prompt(
        self, 
        prompt_text: str, 
        task_type: str = "action_sequencing",
        use_vectorized_enhancement: bool = None
    ) -> str:
        """
        Enhanced version with optional vectorized prompt support
        """
        if use_vectorized_enhancement is None:
            use_vectorized_enhancement = self.enable_vectorized_prompts and task_type == "transition_modeling"
        
        # 如果启用向量化增强且是transition modeling任务
        if use_vectorized_enhancement and self.enable_vectorized_prompts:
            enhanced_prompt = PromptEnhancer.enhance_with_metadata(
                prompt_text,
                self.vector_metadata,
                TaskContext(task_type, "virtualhome", self.num_agents)
            )
            prompt_to_use = enhanced_prompt
        else:
            prompt_to_use = prompt_text
        
        # 选择系统提示
        system_prompt = ""
        if task_type == "action_sequencing":
            system_prompt = SYSTEM_PROMPT_EAI
        elif task_type == "goal_interpretation":
            system_prompt = SYSTEM_PROMPT_GOAL_INTERPRETATION_EAI
        elif task_type == "subgoal_decomposition":
            system_prompt = SYSTEM_PROMPT_SUBGOAL_DECOMPOSITION_EAI
        elif task_type == "transition_modeling":
            system_prompt = SYSTEM_PROMPT_TRANSITION_MODELING_EAI

        if not self.client:
            raise ValueError("API client not initialized. Please provide api_key.")

        max_retries = 5
        retry_delay = 10  # seconds
        for attempt in range(max_retries):
            try:
                user_message = prompt_to_use if use_vectorized_enhancement else f"High-level task: {prompt_to_use}"
                messages_to_send = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]

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

                # JSON验证 - 对transition_modeling任务，不强制要求严格的JSON格式
                if task_type in ['goal_interpretation', 'subgoal_decomposition']:
                    try:
                        # 处理可能的markdown代码块
                        cleaned_response = response_text.strip()
                        if cleaned_response.startswith('```json'):
                            cleaned_response = cleaned_response[7:]  # 移除 ```json
                        if cleaned_response.endswith('```'):
                            cleaned_response = cleaned_response[:-3]  # 移除 ```
                        cleaned_response = cleaned_response.strip()
                        
                        parsed_json = json.loads(cleaned_response)
                        
                        # 对于goal_interpretation，转换旧格式到新格式
                        if task_type == 'goal_interpretation' and 'node_goals' in parsed_json:
                            structured_goal = []
                            
                            # 转换node_goals
                            for node_goal in parsed_json.get('node_goals', []):
                                if 'name' in node_goal and 'state' in node_goal:
                                    structured_goal.append(f"{node_goal['state'].lower()}({node_goal['name']}.1)")
                            
                            # 转换edge_goals
                            for edge_goal in parsed_json.get('edge_goals', []):
                                if 'from_name' in edge_goal and 'relation' in edge_goal and 'to_name' in edge_goal:
                                    structured_goal.append(f"{edge_goal['relation'].lower()}({edge_goal['from_name']}.1, {edge_goal['to_name']}.2)")
                            
                            # 转换action_goals
                            for action_goal in parsed_json.get('action_goals', []):
                                if 'action' in action_goal:
                                    structured_goal.append(f"{action_goal['action'].lower()}({action_goal.get('object', 'object')}.1)")
                            
                            # 返回新格式
                            return json.dumps({"structured_goal": structured_goal})
                        
                        return response_text
                    except json.JSONDecodeError:
                        print(f"Warning: LLM output is not valid JSON for task {task_type}. Content: '{response_text}'")
                        return "{}"
                
                return response_text
            except Exception as e:
                print(f"API error: {e}. Retrying in {retry_delay} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return "{}" if task_type in ['goal_interpretation', 'subgoal_decomposition'] else "[]"

    def update_vector_metadata(self, **kwargs):
        """更新向量元数据"""
        if self.enable_vectorized_prompts:
            for key, value in kwargs.items():
                if hasattr(self.vector_metadata, key):
                    setattr(self.vector_metadata, key, value)
            # 重新创建prompt generator
            self.prompt_generator = VectorizedPromptGenerator(
                self.vector_metadata, 
                self.task_context
            )

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