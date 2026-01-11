import gymnasium as gym
import numpy as np
import cv2


class Gym(gym.Env):
    def __init__(self, task, obs_key="state", act_key="action", size=(96, 96), seed=0):
        self._env = gym.make(task, render_mode="rgb_array")  
        self._obs_is_dict = hasattr(self._env.observation_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._size = size
        self._random = np.random.RandomState(seed)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def observation_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return gym.spaces.Dict(
            {
                **spaces,
                # "image": gym.spaces.Box(0, 255, self._size + (3,), np.uint8),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            }
        )

    @property
    def action_space(self):
        space = self._env.action_space
        space.discrete = True
        return space

    def _get_image(self):
        frame = self.env.render()
        if frame is None:
            return np.zeros((*self._size, 3), dtype=np.uint8)
            
        resized = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        return resized
    
    def step(self, action, *arg, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(action, *arg, **kwargs)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        # obs["image"] = self._get_image()
        obs["is_first"] = False
        obs["is_last"] = terminated or truncated
        obs["is_terminal"] = terminated
        return obs, reward, terminated or truncated, info

    def reset(self, *arg, **kwargs):
        obs, info = self._env.reset(*arg, **kwargs)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        # obs["image"] = self._get_image()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs
