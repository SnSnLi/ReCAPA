import torch
import torch.optim as optim
from collections import defaultdict
import sys
import os
import torch.nn.functional as F
import numpy as np
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder import NomicEncoder
from models.forward_policy_network import PolicyNetwork
from models.llm_trajectory import LLMTrajectoryGenerator, MultiTransition
from models.reflection_head import ReflectionHead
from models.hpcr_reflection_head import HPCRReflectionHead
from models.rce import RceModule
from models.cgf import CGFModule
# from environments.env_wrapper import MuMAToMGymEnv   
from environments.env_wrapper import VirtualHomeWrapper  

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # --- Instantiate Modules ---
        self.encoder = NomicEncoder(device=self.device)
        
        self.policy_net = PolicyNetwork(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            num_agents=config["num_agents"]
        ).to(self.device)

        self.rce_module = RceModule(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"],
            temperature=config.get("rce_temperature", 0.1),
            # Pass alignment-specific params
            sinkhorn_epsilon=config["sinkhorn_eps"],
            score_field_hidden_dim=config["score_hidden"],
            prompt_embedding_dim=config["state_dim"], # Assuming prompt emb dim matches state dim
            num_heads=config["num_heads"]
        ).to(self.device)

        self.cgf_module = CGFModule(
            hidden_dim=config["hidden_dim"],
            policy_net=self.policy_net
        ).to(self.device)

        # 根据配置选择使用传统ReflectionHead还是HPCR版本
        if config.get("enable_hpcr", False):
            self.reflection_head = HPCRReflectionHead(
                config=config,
                rce_module=self.rce_module,
                cgf_module=self.cgf_module,
                gamma=config["gamma"]
            ).to(self.device)
            print("使用HPCR增强版反思头")
        else:
            self.reflection_head = ReflectionHead(
                rce_module=self.rce_module,
                cgf_module=self.cgf_module,
                gamma=config["gamma"]
            ).to(self.device)
            print("使用传统反思头")

        self.llm_traj_gen = LLMTrajectoryGenerator(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            api_key=config["llm_api_key"],
            model_name=config["llm_model_name"],
            api_base=config["llm_api_base"],
            num_agents=config["num_agents"],
            hidden_dim=config["hidden_dim"],
            device=self.device
        )

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.reflection_head.parameters()),
            lr=config["lr"]
    )

        self.env = VirtualHomeWrapper(config["executable_path"], no_graphics=True)
        self.best_eval_success_rate = -1.0

    def train(self):
        for ep in range(1, self.config["num_episodes"] + 1):
            # Dynamic weight scheduling
            warmup_episodes = self.config.get("warmup_episodes", self.config["num_episodes"] / 10)
            progress = min(ep / warmup_episodes, 1.0)
            
            lambda_sinkhorn = self.config.get("lambda_sinkhorn_start", 0.0) + \
                              (self.config.get("lambda_sinkhorn_end", 0.1) - self.config.get("lambda_sinkhorn_start", 0.0)) * progress
            
            lambda_score = self.config.get("lambda_score_start", 0.0) + \
                           (self.config.get("lambda_score_end", 0.1) - self.config.get("lambda_score_start", 0.0)) * progress

            total_loss, loss_breakdown = self.train_one_episode(lambda_sinkhorn, lambda_score)
            
            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                loss_info = (f"Episode {ep}/{self.config['num_episodes']} | "
                            f"Total Loss: {loss_breakdown['total_loss']:.2f} | "
                            f"Contrastive: {loss_breakdown['contrastive_loss']:.2f} | "
                            f"Sinkhorn: {loss_breakdown['sinkhorn_loss']:.2f} | "
                            f"Score: {loss_breakdown['score_field_loss']:.2f}")
                
                # 如果使用HPCR，添加HPCR损失信息
                if self.config.get("enable_hpcr", False):
                    hpcr_loss = loss_breakdown.get('total_hpcr_loss', 0.0)
                    mine_loss = loss_breakdown.get('total_mine_loss', 0.0)
                    loss_info += f" | HPCR: {hpcr_loss:.2f} | MINE: {mine_loss:.2f}"
                    
                    # 显示各层级HPCR损失
                    if 'hpcr_low_to_mid' in loss_breakdown:
                        loss_info += f" | L->M: {loss_breakdown['hpcr_low_to_mid']:.2f}"
                    if 'hpcr_mid_to_high' in loss_breakdown:
                        loss_info += f" | M->H: {loss_breakdown['hpcr_mid_to_high']:.2f}"
                
                print(loss_info)
            else:
                 print(f"Episode {ep}/{self.config['num_episodes']} | No valid data for training.")

            # --- Evaluation Step ---
            if ep % self.config.get("eval_interval", 20) == 0:
                eval_metrics = self.evaluate()
                print(f"\n--- Evaluation at Episode {ep} ---")
                print(f"Success Rate: {eval_metrics['success_rate']:.2f}")
                print(f"Avg Episode Length: {eval_metrics['avg_ep_length']:.2f}")
                print(f"Avg Reward: {eval_metrics['avg_reward']:.2f}")
                print("------------------------------------\n")

                # Save the model if it has the best success rate so far
                if eval_metrics['success_rate'] > self.best_eval_success_rate:
                    self.best_eval_success_rate = eval_metrics['success_rate']
                    save_path = os.path.join(self.config.get("save_dir", "results/models"), "best_policy.pth")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.policy_net.state_dict(), save_path)
                    print(f"New best policy saved to {save_path} with success rate {self.best_eval_success_rate:.2f}")

        self.env.close()

    def evaluate(self, num_eval_episodes: int = 10):
        self.policy_net.eval()
        
        total_successes = 0
        total_ep_length = 0
        total_reward = 0

        with torch.no_grad():
            for _ in range(num_eval_episodes):
                obs = self.env.reset()
                done_flags = {aid: False for aid in range(self.config["num_agents"])}
                ep_reward = 0
                ep_length = 0
                
                for step in range(self.config["max_steps"]):
                    for agent_name in self.env.agent_iter():
                        if self.env.dones.get(agent_name, True):
                            self.env.step(None)
                            continue
                        
                        agent_id = int(agent_name.split('_')[-1])
                        raw_obs = obs
                        
                        img = raw_obs.get('obs_0')
                        state_emb = self.encoder.encode_image([img]).squeeze(0).to(self.device)
                        
                        action, _ = self.policy_net.sample_action(state_emb, agent_id)
                        
                        next_obs, reward, done, info = self.env.step(action.cpu().numpy())

                        ep_reward += reward
                        done_flags[agent_id] = done
                        obs = next_obs
                        
                        if all(done_flags.values()):
                            break
                    
                    ep_length += 1
                    if all(done_flags.values()):
                        break
                
                if all(done_flags.values()):
                    total_successes += 1
                
                total_ep_length += ep_length
                total_reward += ep_reward

        self.policy_net.train()

        return {
            "success_rate": total_successes / num_eval_episodes,
            "avg_ep_length": total_ep_length / num_eval_episodes,
            "avg_reward": total_reward / num_eval_episodes
        }

    def train_one_episode(self, lambda_sinkhorn: float, lambda_score: float):
        # --- Setup for the episode ---
        episode_buf = defaultdict(list)
        obs = self.env.reset()
        done_flags = {aid: False for aid in range(self.config["num_agents"])}

        # Define and encode a fixed prompt for the episode
        env_description = "A virtual home environment where agents perform household tasks."
        strategy_context = "The agent should try to complete the task efficiently while avoiding mistakes."
        full_prompt_text = f"{env_description} {strategy_context}"
        
        # This is the global prompt embedding
        prompt_global_embedding = self.encoder.encode_text(full_prompt_text).squeeze()

        # For Sinkhorn loss, we need token-level embeddings. 
        # Here we simulate them by splitting the text and encoding, or simply replicating the global one.
        # A simple placeholder: repeat global embedding N times. Let's say N=10.
        prompt_token_embeddings = prompt_global_embedding.unsqueeze(0).repeat(10, 1)

        # --- Data Collection Loop ---
        for _ in range(self.config["max_steps"]):
            for agent_name in self.env.agent_iter():
                if self.env.dones.get(agent_name, True):
                    self.env.step(None)
                    continue

                agent_id = int(agent_name.split('_')[-1])
                raw_obs = obs

                img = raw_obs.get('obs_0')
                state_emb = self.encoder.encode_image([img]).squeeze(0).to(self.device)

                action, _ = self.policy_net.sample_action(state_emb, agent_id)
                
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())

                episode_buf[agent_id].append(MultiTransition(
                    agent_id=agent_id, state=state_emb, action=action, 
                    reward=torch.tensor(float(reward), device=self.device)
                ))

                done_flags[agent_id] = done
                obs = next_obs
                if all(done_flags.values()): break
            if all(done_flags.values()): break

        # --- Reflection and Update Step ---
        total_loss_for_ep = torch.tensor(0.0, device=self.device)
        loss_breakdown_for_ep = defaultdict(float)
        num_valid_agents = 0

        for aid in range(self.config["num_agents"]):
            buf = episode_buf[aid]
            if len(buf) < self.config["sub_len"]:
                continue

            # --- Hierarchical Reflection ---
            # A helper function to perform reflection for a given level
            def _perform_reflection(level: str, segment_tensor: torch.Tensor, segment_len: int, episode_step: int = 0):
                if len(buf) < segment_len: return None, None

                trajs_with_logprobs = self.llm_traj_gen.generate_contrastive_trajectories(
                    current_traj=segment_tensor, agent_id=aid, level=level, num_trajs=5,
                    env_description=env_description, strategy_context=strategy_context
                )
                if not trajs_with_logprobs: return None, None

                candidate_trajs = [item[0] for item in trajs_with_logprobs]
                logprobs_list = [item[1] for item in trajs_with_logprobs]
                
                pos_trajs = [segment_tensor]

                deviation_points = [
                    next((i for i, lp in enumerate(lps) if lp.logprob < self.config["logprob_threshold"]), -1)
                    for lps in logprobs_list
                ]
                neg_weights = torch.tensor([
                    1.0 / (p + 2.0) if p != -1 else 1.0 for p in deviation_points
                ], device=self.device)

                # 如果使用HPCR，需要准备层级数据
                hierarchical_data = None
                if self.config.get("enable_hpcr", False) and hasattr(self.reflection_head, 'prepare_hierarchical_data'):
                    # 将episode buffer转换为trajectory格式
                    full_trajectory = []
                    for transition in buf:
                        step_dict = {
                            'state': transition.state,
                            'action': transition.action,
                            'reward': transition.reward
                        }
                        full_trajectory.append(step_dict)
                    
                    # 准备层级数据
                    hierarchical_data = self.reflection_head.prepare_hierarchical_data(
                        full_trajectory, episode_step
                    )

                level_loss, level_loss_breakdown = self.reflection_head(
                    anchor_traj=segment_tensor,
                    pos_trajs=pos_trajs,
                    neg_trajs=candidate_trajs,
                    prompt_token_embeddings=prompt_token_embeddings,
                    prompt_global_embedding=prompt_global_embedding,
                    sinkhorn_weight=lambda_sinkhorn,
                    score_field_weight=lambda_score,
                    neg_weights=neg_weights,
                    level=level,
                    hierarchical_data=hierarchical_data
                )
                return level_loss, level_loss_breakdown

            agent_total_loss = torch.tensor(0.0, device=self.device)
            
            # 1. Low-level reflection (always)
            low_len = self.config["sub_len"]
            low_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-low_len:]])
            low_loss, low_breakdown = _perform_reflection('low', low_seg_tensor, low_len, len(buf))
            if low_loss is not None:
                weighted_loss = low_loss * self.config.get('lambda_contrastive_low', 1.0)
                agent_total_loss += weighted_loss
                for k, v in low_breakdown.items():
                    loss_breakdown_for_ep[f"low_{k}"] += v

            # 2. Mid-level reflection (periodic)
            if ep % self.config.get("mid_level_interval", 5) == 0:
                mid_len = self.config.get("mid_len", 50)
                if len(buf) >= mid_len:
                    mid_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf[-mid_len:]])
                    mid_loss, mid_breakdown = _perform_reflection('mid', mid_seg_tensor, mid_len, len(buf))
                    if mid_loss is not None:
                        weighted_loss = mid_loss * self.config.get('lambda_contrastive_mid', 0.5)
                        agent_total_loss += weighted_loss
                        for k, v in mid_breakdown.items():
                            loss_breakdown_for_ep[f"mid_{k}"] += v

            # 3. High-level reflection (periodic)
            if ep % self.config.get("high_level_interval", 10) == 0:
                high_len = len(buf) # Full episode
                high_seg_tensor = torch.stack([torch.cat([s.state, s.action, s.reward.view(1)]) for s in buf])
                high_loss, high_breakdown = _perform_reflection('high', high_seg_tensor, high_len, len(buf))
                if high_loss is not None:
                    weighted_loss = high_loss * self.config.get('lambda_contrastive_high', 0.2)
                    agent_total_loss += weighted_loss
                    for k, v in high_breakdown.items():
                        loss_breakdown_for_ep[f"high_{k}"] += v

            if agent_total_loss.item() > 0:
                total_loss_for_ep += agent_total_loss
                num_valid_agents += 1

        if num_valid_agents > 0:
            avg_total_loss = total_loss_for_ep / num_valid_agents
            for k in loss_breakdown_for_ep:
                loss_breakdown_for_ep[k] /= num_valid_agents
            return avg_total_loss, loss_breakdown_for_ep
        else:
            return None, None


def main():
    import yaml
    
    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "rsn_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override API key from environment if available
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        config["llm_api_key"] = api_key
    elif not config.get("llm_api_key"):
        print("Warning: No API key found in config or environment. Using placeholder.")
        config["llm_api_key"] = "YOUR_FALLBACK_API_KEY_HERE"
    
    # Ensure device is set
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"配置加载完成:")
    print(f"- 使用设备: {config['device']}")
    print(f"- HPCR模式: {'启用' if config.get('enable_hpcr', False) else '禁用'}")
    print(f"- 训练轮数: {config['num_episodes']}")
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()