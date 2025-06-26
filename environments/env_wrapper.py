import gymnasium as gym
import numpy as np
from gymnasium import spaces


from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from environments.custom_env import CustomVirtualHomeEnv
# from alfred.gen import Tasks        
# from alfred.vis.env.thor_env import ThorEnv 



# class AlfredWrapper(AECEnv):
#     metadata = {'render.modes': ['human']}
#     def __init__(self, data_root, split='train', no_graphics=True, time_scale=1.0, seed=1):
#         super().__init__()
#         # 1) 加载所有 Alfred 任务
#         tasks = Tasks(data_root, split=split).get_tasks()
#         # 2) 创建 AI2-THOR 环境
#         self.envs = [ThorEnv(t, no_graphics=no_graphics, time_scale=time_scale, seed=seed+i)
#                      for i,t in enumerate(tasks)]
#         self.cur = 0
#         # 3) 初始化 agent_selector
#         init_obs = self.envs[self.cur].reset()
#         self.possible_agents = ['alfred']
#         self.agents = self.possible_agents[:]
#         self.agent_selection = agent_selector(self.agents).next()
#         # 4) 定义 spaces（以第一个任务的 obs/action 为准）
#         obs = init_obs[self.agent_selection]
#         self.observation_spaces = {
#             'alfred': spaces.Dict({
#                 'rgb': spaces.Box(0,255,obs['rgb'].shape, dtype=np.uint8),
#                 'depth': spaces.Box(0,1,obs['depth'].shape, dtype=np.float32),
#                 'instruction': spaces.Box(0,1,(1,),dtype=object),
#             })
#         }
#         # Alfred 默认动作集
#         self.action_spaces = {
#             'alfred': spaces.Discrete(len(self.envs[self.cur].action_names))
#         }
#         self.rewards = {a:0 for a in self.agents}
#         self.dones   = {a:False for a in self.agents}
#         self.infos   = {a:{} for a in self.agents}
#         self._agent_sel = agent_selector(self.agents)

#     def reset(self):
#         obs_dict = self.envs[self.cur].reset()
#         self.agents = self.possible_agents[:]
#         self.agent_selection = self._agent_sel.reinit(self.agents).next()
#         return obs_dict[self.agent_selection]

#     def step(self, action):
#         # 执行动作
#         obs_dict, rew_dict, done_dict, info_dict = self.envs[self.cur].step({action})
#         a = self.agent_selection
#         self.rewards[a] = rew_dict[a]
#         self.dones[a]   = done_dict[a]
#         self.infos[a]   = info_dict[a]
#         if self.dones[a]:
#             self.agents.remove(a)
#         reward = self.rewards[a]
#         done   = self.dones[a]
#         info   = self.infos[a]
#         self.agent_selection = self._agent_sel.next()
#         return obs_dict[a], reward, done, info

#     def close(self):
#         for e in self.envs:
#             e.close()


class VirtualHomeWrapper(AECEnv):
    metadata = {'render.modes': ['human']}
    def __init__(self, executable_path, no_graphics=True, time_scale=1.0, seed=1):
        super().__init__()
        self.custom = CustomVirtualHomeEnv(executable_path, no_graphics, time_scale, seed)
        self.behaviors = self.custom.behaviors
        # 首次 reset 得到全体 agent keys
        init = self.custom.reset()
        self.possible_agents = [f"{b}_{aid}" for (b,aid) in init.keys()]
        self.agents = self.possible_agents[:]
        self.agent_selection = agent_selector(self.agents).next()
        # 构建 spaces
        spec = next(iter(init.values()))
        self.observation_spaces = {a: spaces.Dict({k: spaces.Box(-np.inf, np.inf, v.shape, v.dtype)
                                                   for k,v in obs.items()})
                                   for a, obs in zip(self.agents, init.values())}
        act_branch = self.custom.env.behavior_specs[self.behaviors[0]].action_spec.discrete_branches[0]
        self.action_spaces = {a: spaces.Discrete(act_branch) for a in self.agents}
        self.rewards = {a:0 for a in self.agents}
        self.dones   = {a:False for a in self.agents}
        self.infos   = {a:{} for a in self.agents}
        self._agent_sel = agent_selector(self.agents)

    def reset(self):
        obs = self.custom.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_sel.reinit(self.agents).next()
        return obs[next(iter(obs.keys()))]

    def step(self, action):
        b, aid = self.agent_selection.split('_')
        key = (b, int(aid))
        obs, rew, dones, infos = self.custom.step({key: action})
        for a in self.agents:
            b2, i2 = a.split('_')
            k2 = (b2, int(i2))
            self.rewards[a] = rew.get(k2, 0)
            self.dones[a]   = dones.get(k2, False)
            self.infos[a]   = infos.get(k2, {})
        if self.dones[self.agent_selection]:
            self.agents.remove(self.agent_selection)
        reward = self.rewards[self.agent_selection]
        done   = self.dones[self.agent_selection]
        info   = self.infos[self.agent_selection]
        self.agent_selection = self._agent_sel.next()
        return obs[key], reward, done, info


    def close(self):
        self.custom.close()

