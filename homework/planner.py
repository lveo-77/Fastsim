"""作业:你只需要改这个文件。

给你整条赛道(左右边界、中心线),返回一条路线和每个点上的车速。
评分脚本会用 fastsim.PathFollower 沿你的路线开一圈,用时越短越好,出界算 0 分。

现在的写法是:沿中心线、全程 1.0 m/s。它一定能跑完,但很慢。
改进方向(由易到难):
  1. 直道快、弯道慢:按路线的弯曲程度(曲率)算每个点能跑多快;
  2. 考虑加减速:车不能瞬间刹下来,入弯前要提前减速;
  3. 不走中心线:外-内-外切弯,让弯道半径变大,就能跑得更快;
  4. 自己写控制器替换 PathFollower(进阶,见 README)。
"""
import numpy as np

from fastsim import CarParams, Track


def plan(track: Track, car: CarParams) -> tuple[np.ndarray, np.ndarray]:
    """返回 (path, speed):path 形状 (n,2),speed 形状 (n,),单位 m 和 m/s。"""
    path = track.centerline.copy()
    speed = np.full(len(path), 1.0)
    return path, speed
