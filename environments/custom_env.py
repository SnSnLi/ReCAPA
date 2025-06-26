
import json
import random
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ai2thor.controller import Controller


class AlfredEnv:
    def __init__(self):
        self.controller = Controller()
        self.controller.reset("FloorPlan1")  # 选择场景
        self.current_state = None
        self.current_instruction = None  # 存储当前指令

    def reset(self):
        event = self.controller.reset("FloorPlan1")
        self.current_state = self._get_state(event)
        return self.current_state

    def step(self, action):
        event = self.controller.step(action)
        next_state = self._get_state(event)
        reward = self._get_reward(event)
        done = self._is_done(event)
        return next_state, reward, done, {}

    def _get_state(self, event):
        image = event.frame  # RGB 图像
        depth = event.depth_frame  # 深度图
        return {"image": image, "depth": depth, "language": self.current_instruction}

    def _get_reward(self, event):
        return 0  # 根据任务定义奖励

    def _is_done(self, event):
        return False
        


class CustomVirtualHomeEnv:
    """
    封装 VirtualHome Unity SDK。
    reset()/step() 返回 dict keyed by (behavior, agent_id)。
    """
    def __init__(self, executable_path: str, no_graphics=True, time_scale=1.0, seed=1):
        ch = EngineConfigurationChannel()
        ch.set_configuration_parameters(time_scale=time_scale)
        self.env = UnityEnvironment(
            file_name=executable_path,
            timeout_wait=60,
            side_channels=[ch],
            no_graphics=True,
            worker_id=0,       
            base_port=5005,
            seed=seed
        )
        self.env.reset()
        self.behaviors = list(self.env.behavior_specs.keys())

    def reset(self):
        self.env.reset()
        out = {}
        for b in self.behaviors:
            dec, _ = self.env.get_steps(b)
            spec = self.env.behavior_specs[b]
            for aid in dec.agent_id_to_index:
                idx = dec.agent_id_to_index[aid]
                obs = {f"obs_{i}": dec.obs[i][idx] for i in range(len(spec.observation_shapes))}
                out[(b, aid)] = obs
        return out

    def step(self, action_dict):
        # action_dict: {(behavior, agent_id): action_array, ...}
        for b in self.behaviors:
            dec, _ = self.env.get_steps(b)
            spec = self.env.behavior_specs[b]
            acts = [action_dict.get((b, aid), np.zeros(spec.action_spec.discrete_branches[0]))
                    for aid in dec.agent_id_to_index]
            if acts:
                self.env.set_actions(b, np.stack(acts))
        self.env.step()
        obs, rew, done, info = {}, {}, {}, {}
        for b in self.behaviors:
            dec, term = self.env.get_steps(b)
            spec = self.env.behavior_specs[b]
            for aid in dec.agent_id_to_index:
                idx = dec.agent_id_to_index[aid]
                obs[(b, aid)] = {f"obs_{i}": dec.obs[i][idx] for i in range(len(spec.observation_shapes))}
                rew[(b, aid)] = dec.reward[idx]
                done[(b, aid)] = False
                info[(b, aid)] = {}
            for aid in term.agent_id_to_index:
                idx = term.agent_id_to_index[aid]
                obs[(b, aid)] = None
                rew[(b, aid)] = term.reward[idx]
                done[(b, aid)] = True
                info[(b, aid)] = {}
        return obs, rew, done, info

    def close(self):
        self.env.close()




