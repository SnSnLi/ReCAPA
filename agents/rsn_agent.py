# rsn_agent.py

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List

# Corrected imports
from models.encoder import NomicEncoder
from models.rce import RceModule as GEMRModule # Alias RceModule as GEMRModule for compatibility
from models.forward_policy_network import PolicyNetwork
from models.llm_trajectory import LLMTrajectory, MultiTransition

class RSNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_agents: int,
                 device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1) Encoders
        # Text encoder for task descriptions and trajectory scoring
        self.text_encoder = NomicEncoder(device=self.device)
        
        # 2) Policy Network
        self.policy = PolicyNetwork(
            state_dim, action_dim, num_agents=num_agents
        ).to(self.device)
        
        # 3) RCE/GEMR Module + ReflectionHead
        self.gemr = GEMRModule(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128, # Example value, adjust as needed
            prompt_embedding_dim=self.text_encoder.text_model.config.hidden_size
        ).to(self.device)

        # 4) Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) +
            list(self.gemr.encoders.parameters()) + # Updated to correct module
            list(self.gemr.embedding_heads.parameters()), # Updated to correct module
            lr=1e-4
        )
        # 5) Trajectory Buffer
        self.episode_buf = defaultdict(list)

    def select_best_trajectory(
        self,
        task_description: str,
        candidates: List[str]
    ) -> str:
        """
        Selects the best trajectory from a list of candidates based on semantic
        similarity to the task description.

        :param task_description: The high-level task instruction.
        :param candidates: A list of candidate trajectory strings.
        :return: The candidate string with the highest similarity score.
        """
        if not candidates:
            return ""

        # Encode the task description and all candidate trajectories
        task_embedding = self.text_encoder.encode_text(task_description, task_type="search_document")
        candidate_embeddings = self.text_encoder.encode_text(candidates, task_type="search_query")

        # Compute cosine similarity between the task and each candidate
        similarities = F.cosine_similarity(task_embedding, candidate_embeddings, dim=1)

        # Find the index of the candidate with the highest similarity
        best_candidate_idx = torch.argmax(similarities).item()

        return candidates[best_candidate_idx]

    def select_action(self, agent_id: int, raw_obs) -> torch.Tensor:
        # This is a placeholder for image encoding, which is not used in the EAI script
        # In a real scenario, you would have a proper image encoder here.
        state_emb = torch.randn(self.policy.state_dim, device=self.device) # Placeholder
        # Policy network sampling
        action, logp = self.policy.sample_action(state_emb, agent_id)
        return action, logp, state_emb
    
    # --- The following methods are for training and are not used in the current eval script ---

    def store_transition(self,
                         agent_id: int,
                         state_emb: torch.Tensor,
                         action: torch.Tensor,
                         reward: float):
        # Store as MultiTransition
        mt = MultiTransition(agent_id, state_emb, action, torch.tensor(reward))
        self.episode_buf[agent_id].append(mt)

    def finish_episode(self, sub_len: int):
        # Slice and store full agent trajectories in GEMR
        for aid, traj in self.episode_buf.items():
            self.gemr.store_episode(aid, traj, sub_len) # This method needs to be added to RceModule if used
        # Clear buffer
        self.episode_buf.clear()

    def reflect_and_update(self, agent_id: int, sub_len: int):
        # 从缓存中取最新 sub_len+1 段
        traj = self.episode_buf[agent_id][- (sub_len+1):]
        # 反思更新
        loss = self.gemr.reflect(agent_id, traj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def reflect_all_and_update(self, sub_len: int):
        # 针对所有 agent 一次性反思
        all_traj = sum(self.episode_buf.values(), [])  # flatten
        loss = self.gemr.reflect_all(all_traj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
