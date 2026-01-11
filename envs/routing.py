"""
TSP (Traveling Salesman Problem) and CVRP (Capacitated Vehicle Routing Problem) environments.
"""
import gymnasium as gym
import numpy as np


class TSPEnv(gym.Env):
    """
    TSP 환경: 모든 노드를 최단 경로로 방문하고 시작점으로 복귀.
    
    Observation:
        - coords: (num_nodes, 2) 노드 좌표
        - current_node: (1,) 현재 위치
        - mask: (num_nodes,) 방문 가능 여부 (True = 방문 가능)
    
    Action:
        - Discrete(num_nodes): 다음 방문할 노드 인덱스
    
    Reward:
        - 이동 거리의 음수값 (dense reward)
    """
    metadata = {}
    
    def __init__(self, num_nodes=20, seed=0):
        super().__init__()
        self.num_nodes = num_nodes
        self._random = np.random.RandomState(seed)
        
        # Spaces
        self.action_space = gym.spaces.Discrete(num_nodes)
        self.action_space.discrete = True
        
        self._coords = None
        self._current_node = None
        self._visited = None
        self._step_count = None
    
    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'coords': gym.spaces.Box(-np.inf, np.inf, (self.num_nodes * 2,), dtype=np.float32),
            'current_node': gym.spaces.Box(0, 1, (self.num_nodes,), dtype=np.float32),
            'current_pos': gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            'mask': gym.spaces.Box(0, 1, (self.num_nodes,), dtype=bool),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })
    
    def _get_obs(self, is_first=False, is_last=False, is_terminal=False):
        # Mask: True = can visit, False = already visited
        mask = ~self._visited.copy()
        # Depot (node 0) checks
        if np.all(self._visited[1:]):
            # All other nodes visited -> Depot is the ONLY valid target
            mask[0] = True
        else:
            # Not all visited -> Depot is invalid (cannot return yet)
            mask[0] = False
        
        return {
            'coords': self._coords.flatten().astype(np.float32),
            'current_node': np.eye(self.num_nodes)[self._current_node].astype(np.float32),
            'current_pos': self._coords[self._current_node].astype(np.float32),
            'mask': mask,
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._random = np.random.RandomState(seed)
        
        # Random coordinates in [0, 1]
        self._coords = self._random.rand(self.num_nodes, 2)
        self._current_node = 0  # Start at depot
        self._visited = np.zeros(self.num_nodes, dtype=bool)
        self._visited[0] = True
        self._step_count = 0
        
        return self._get_obs(is_first=True)
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.flat[0])
        
        self._step_count += 1
        prev_node = self._current_node
        
        # Check if action is valid (not already visited, or depot when all visited)
        is_valid = not self._visited[action]
        if action == 0:  # Depot
            is_valid = np.all(self._visited[1:])  # Valid only if all others visited
        
        if is_valid:
            self._current_node = action
            self._visited[action] = True
            # Reward: negative distance
            distance = np.linalg.norm(self._coords[prev_node] - self._coords[action])
            reward = -distance
        else:
            # Invalid action: heavy penalty, no state change
            reward = -2.0  # Penalty larger than max possible distance (sqrt(2) ≈ 1.41)
        
        # Done when returned to depot after visiting all
        done = (action == 0 and np.all(self._visited))
        
        # Truncate if too many steps (including invalid ones)
        truncated = self._step_count >= self.num_nodes + 10
        
        obs = self._get_obs(is_last=done or truncated, is_terminal=done)
        return obs, float(reward), done or truncated, {}

    
    def render(self):
        pass


class CVRPEnv(gym.Env):
    """
    CVRP 환경: 차량 용량 제약이 있는 TSP.
    
    추가 Observation:
        - demands: (num_nodes,) 각 노드의 수요
        - current_capacity: (1,) 현재 남은 용량
    
    Mask 로직:
        - 용량 초과 노드는 방문 불가
        - 용량 부족시 depot으로 복귀해야 함
    """
    metadata = {}
    
    def __init__(self, num_nodes=20, capacity=1.0, seed=0):
        super().__init__()
        self.num_nodes = num_nodes
        self.capacity = capacity
        self._random = np.random.RandomState(seed)
        
        self.action_space = gym.spaces.Discrete(num_nodes)
        self.action_space.discrete = True
        
        self._coords = None
        self._demands = None
        self._current_node = None
        self._current_capacity = None
        self._visited = None
        self._step_count = None
    
    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'coords': gym.spaces.Box(-np.inf, np.inf, (self.num_nodes * 2,), dtype=np.float32),
            'demands': gym.spaces.Box(0, np.inf, (self.num_nodes,), dtype=np.float32),
            'current_node': gym.spaces.Box(0, 1, (self.num_nodes,), dtype=np.float32),
            'current_capacity': gym.spaces.Box(0, np.inf, (1,), dtype=np.float32),
            'mask': gym.spaces.Box(0, 1, (self.num_nodes,), dtype=bool),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })
    
    def _get_obs(self, is_first=False, is_last=False, is_terminal=False):
        # Mask: True = can visit
        mask = ~self._visited.copy()
        # 용량 초과 노드 제외
        mask = mask & (self._demands <= self._current_capacity)
        # Depot은 항상 방문 가능 (용량 충전)
        mask[0] = True
        # 모든 노드 방문 후 depot만 방문 가능
        if np.all(self._visited[1:]):
            mask[:] = False
            mask[0] = True
        
        return {
            'coords': self._coords.flatten().astype(np.float32),
            'demands': self._demands.astype(np.float32),
            'current_node': np.eye(self.num_nodes)[self._current_node].astype(np.float32),
            'current_capacity': np.array([self._current_capacity], dtype=np.float32),
            'mask': mask,
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._random = np.random.RandomState(seed)
        
        self._coords = self._random.rand(self.num_nodes, 2)
        # Demands: random in [0.1, 0.3], depot has 0 demand
        self._demands = self._random.uniform(0.1, 0.3, self.num_nodes)
        self._demands[0] = 0.0
        
        self._current_node = 0
        self._current_capacity = self.capacity
        self._visited = np.zeros(self.num_nodes, dtype=bool)
        self._visited[0] = True
        self._step_count = 0
        
        return self._get_obs(is_first=True)
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.flat[0])
        
        self._step_count += 1
        prev_node = self._current_node
        
        # Check validity: not visited, has enough capacity, or returning to depot
        is_valid = False
        if action == 0:
            # Depot always valid (refill)
            is_valid = True
        elif not self._visited[action] and self._demands[action] <= self._current_capacity:
            # Valid customer: not visited and enough capacity
            is_valid = True
        
        if is_valid:
            distance = np.linalg.norm(self._coords[prev_node] - self._coords[action])
            reward = -distance
            self._current_node = action
            
            if action == 0:
                self._current_capacity = self.capacity
            else:
                self._current_capacity -= self._demands[action]
                self._visited[action] = True
        else:
            # Invalid action: heavy penalty
            reward = -2.0
        
        # Done when all visited and at depot
        done = (action == 0 and np.all(self._visited))
        truncated = self._step_count >= self.num_nodes * 2
        
        obs = self._get_obs(is_last=done or truncated, is_terminal=done)
        return obs, float(reward), done or truncated, {}
    
    def render(self):
        pass
