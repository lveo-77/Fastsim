"""TODO:写一个控制器,在 A-D 任意一条赛道上跑出最快的一圈。

评分只调用这里的 Controller:
  · Controller(track, car_params)  开跑前调用一次,拿到整条赛道,限时 60 秒;
  · controller(state)              之后每 0.02 秒调用一次,返回 (目标车速 m/s, 目标前轮转角 rad)。
方法不限(仅可更改homework/ 下的文件)。

下面是默认示例:沿中心线、全程 1 m/s,用 fastsim 自带的纯追踪 PathFollower 跟踪。
它一定能跑完,但很慢。最省事的改法是只改 plan(),让路线和速度更好。
"""
import numpy as np

from fastsim import CarParams, PathFollower, State, Track


def plan(track: Track, car: CarParams) -> tuple[np.ndarray, np.ndarray]:
    """返回 (path, speed):path 形状 (n,2) 路线点,speed 形状 (n,) 每点车速。"""
    path = track.centerline.copy()
    speed = np.full(len(path), 1.0)
    return path, speed


class Controller:
    def __init__(self, track: Track, car_params: CarParams):
        path, speed = plan(track, car_params)
        self.follower = PathFollower(path, speed, car_params)

    def __call__(self, state: State) -> tuple[float, float]:
        return self.follower(state)
