"""跑一圈:从起跑线静止出发,车体任何一角压出边界就算出界,回到起跑线算完赛。"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastsim.car import Car
from fastsim.params import CarParams
from fastsim.track import Track, resample_closed

DT = 0.02           # 仿真步长 s(50Hz,和实车控制频率一致)


@dataclass
class LapResult:
    finished: bool
    time: float                 # 完赛用时 s;没完赛就是停下时的时间
    reason: str                 # 'finished' / 'off_track' / 'timeout'
    log: np.ndarray             # (n,5) 每步 [t, x, y, yaw, v]

    def __str__(self):
        if self.finished:
            return f'完赛 {self.time:.2f}s  最高 {self.log[:, 4].max():.2f}m/s'
        return f'{self.reason} @ {self.time:.2f}s'


def run_lap(track: Track, controller, car_params: CarParams = CarParams(),
            time_limit: float = 120.0) -> LapResult:
    """controller(car) -> (目标车速 m/s, 目标前轮转角 rad),每 DT 秒调用一次。

    car 可读的量:car.x / car.y / car.yaw / car.v / car.steer,以及 car.p(车参数)。
    """
    car = Car(car_params)
    start, nxt = track.centerline[0], track.centerline[1]
    car.reset(start[0], start[1], math.atan2(nxt[1] - start[1], nxt[0] - start[0]))
    length = track.length
    s_prev, progress, t = 0.0, 0.0, 0.0
    log = []
    while True:
        log.append((t, car.x, car.y, car.yaw, car.v))
        if t >= time_limit:
            return LapResult(False, t, 'timeout', np.array(log))
        v_cmd, steer_cmd = controller(car)
        car.step(float(v_cmd), float(steer_cmd), DT)
        t += DT
        if not track.on_track(car.corners()).all():
            log.append((t, car.x, car.y, car.yaw, car.v))
            return LapResult(False, t, 'off_track', np.array(log))
        s, _ = track.project((car.x, car.y), near_s=s_prev)
        progress += (s - s_prev + length / 2) % length - length / 2
        s_prev = s
        if progress >= length:
            log.append((t, car.x, car.y, car.yaw, car.v))
            return LapResult(True, t, 'finished', np.array(log))


class PathFollower:
    """沿给定路线开车:纯追踪算转角,速度照速度表给。

    path  (n,2) 闭合路线点(首尾不用重复),顺序要和行驶方向一致
    speed (n,)  每个路线点上想要的车速 m/s
    """

    def __init__(self, path, speed, lookahead: float = 0.35, lookahead_gain: float = 0.25,
                 speed_preview: float = 0.35):
        path, speed = np.asarray(path, float), np.asarray(speed, float)
        closed = np.vstack([path, path[:1]])
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
        self.path = resample_closed(path, 0.05)
        s_new = np.linspace(0, s[-1], len(self.path), endpoint=False)
        self.speed = np.interp(s_new, s, np.append(speed, speed[0]))
        self.ds = s[-1] / len(self.path)
        self.lookahead, self.lookahead_gain, self.speed_preview = lookahead, lookahead_gain, speed_preview
        self.idx = None

    def __call__(self, car) -> tuple[float, float]:
        n = len(self.path)
        if self.idx is None:        # 第一次:全局找最近点
            self.idx = int(np.argmin(np.hypot(*(self.path - (car.x, car.y)).T)))
        else:                       # 之后只往前小范围找,防止跳到隔壁那段
            window = (self.idx + np.arange(-10, 60)) % n
            d = np.hypot(*(self.path[window] - (car.x, car.y)).T)
            self.idx = int(window[np.argmin(d)])
        # 速度:看稍前方的速度表(车追速度有惯性,提前一点给)
        v_cmd = self.speed[(self.idx + int(car.v * self.speed_preview / self.ds)) % n]
        # 转角:纯追踪,瞄准前方 ld 处的路线点
        ld = self.lookahead + self.lookahead_gain * car.v
        target = self.path[(self.idx + max(1, int(ld / self.ds))) % n]
        dx, dy = target[0] - car.x, target[1] - car.y
        alpha = math.atan2(dy, dx) - car.yaw
        steer = math.atan2(2 * car.p.wheelbase * math.sin(alpha), math.hypot(dx, dy))
        return v_cmd, steer
