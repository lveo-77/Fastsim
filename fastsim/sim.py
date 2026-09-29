"""跑一圈:从起跑线静止出发,车身压到边界就算出界,回到起跑线算完赛。

自己写训练环境时,直接用 car_off_track() 和 Progress 判定出界和完赛,就和评分完全一致。
"""
from __future__ import annotations

import math
import time
from dataclasses import astuple, dataclass, replace

import numpy as np

from fastsim.car import Car
from fastsim.params import CarParams
from fastsim.track import Track, resample_closed

DT = 0.02           # 仿真步长 s(50Hz,和实车控制频率一致)


@dataclass(frozen=True)
class State:
    """每步交给控制器的车辆状态(原版为真值,扰动版带观测噪声;只读)。"""
    t: float            # 本圈已用时间 s
    x: float            # 后轴中心位置 m
    y: float
    yaw: float          # 车头朝向 rad,逆时针为正
    v: float            # 车速 m/s
    yaw_rate: float     # 横摆角速度 rad/s
    accel: float        # 纵向加速度 m/s²
    steer: float        # 当前前轮实际转角 rad,左正


# 扰动版的观测噪声(高斯标准差),和 State 字段一一对应;t 不加
NOISE = State(t=0.0, x=0.02, y=0.02, yaw=0.01, v=0.02, yaw_rate=0.02, accel=0.1, steer=0.005)


def noisy_params(p: CarParams, rng: np.random.Generator) -> CarParams:
    """扰动版的真实车参数:摩擦系数和侧向能力上限 ±10%,电机响应、舵机速度 ±20%。控制器拿到的仍是标称值。"""
    return replace(p, mu=p.mu * rng.uniform(0.9, 1.1), speed_tau=p.speed_tau * rng.uniform(0.8, 1.2),
                   steer_rate=p.steer_rate * rng.uniform(0.8, 1.2),
                   lateral_g=p.lateral_g * rng.uniform(0.9, 1.1))


def noisy_state(s: State, rng: np.random.Generator) -> State:
    """给状态加观测噪声(定位、IMU、编码器的误差)。"""
    return State(*(v + rng.normal(0.0, sd) if sd else v for v, sd in zip(astuple(s), astuple(NOISE))))


@dataclass
class LapResult:
    finished: bool
    time: float                 # 完赛用时 s(冲线时刻插值);没完赛就是停下时的时间
    reason: str                 # 'finished' / 'off_track' / 'timeout' / 'bad_output'
    log: np.ndarray             # (n,5) 每步 [t, x, y, yaw, v](真值)
    ctrl_time: float = 0.0      # 控制器每步平均耗时 s
    progress: float = 0.0       # 停下时跑完了一圈的多少(0~1)

    def __str__(self):
        if self.finished:
            return f'完赛 {self.time:.3f}s  最高 {self.log[:, 4].max():.2f}m/s'
        return f'{self.reason} @ {self.time:.2f}s(一圈的 {self.progress:.0%})'


def start_pose(track: Track) -> tuple[float, float, float]:
    """起跑位置:后轴在中心线首点,车头沿中心线。"""
    (x0, y0), (x1, y1) = track.centerline[0], track.centerline[1]
    return float(x0), float(y0), math.atan2(y1 - y0, x1 - x0)


def _cross(p0, p1, q0, q1) -> bool:
    """线段组 p0->p1 (n,2) 与 q0->q1 (m,2) 有没有任何一对相交或接触(含共线重叠)。"""
    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    P0, P1, Q0, Q1 = p0[:, None], p1[:, None], q0[None], q1[None]
    eps = 1e-10
    a, b = orient(P0, P1, Q0), orient(P0, P1, Q1)
    c, d = orient(Q0, Q1, P0), orient(Q0, Q1, P1)
    straddle = (((a <= eps) & (b >= -eps)) | ((b <= eps) & (a >= -eps)))
    straddle &= (((c <= eps) & (d >= -eps)) | ((d <= eps) & (c >= -eps)))
    overlap = (np.maximum(np.minimum(P0, P1), np.minimum(Q0, Q1)) <=
               np.minimum(np.maximum(P0, P1), np.maximum(Q0, Q1)) + eps).all(axis=-1)
    return bool((straddle & overlap).any())


def car_off_track(track: Track, car: Car) -> bool:
    """车身在上一步的运动过程中有没有碰到边界(检查 car.trace 里每个位姿)。

    每个位姿查三样:车身四个角都在赛道上、没有边界折点进入车身、车身的边和边界线段不相交。
    包括边与边的接触和共线重叠。步内使用离散位姿采样,不是严格的连续碰撞检测。
    """
    corners = car.corners(car.trace)                                  # (k,4,2)
    if not track.on_track(corners.reshape(-1, 2)).all():
        return True
    p = car.p
    for x, y, yaw in car.trace:
        rel = track.vertices - (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        lx = rel[:, 0] * c + rel[:, 1] * s
        ly = -rel[:, 0] * s + rel[:, 1] * c
        if ((lx > -p.rear_overhang) & (lx < p.length - p.rear_overhang) & (np.abs(ly) < p.width / 2)).any():
            return True
    body0 = corners.reshape(-1, 2)
    body1 = np.roll(corners, -1, axis=1).reshape(-1, 2)
    return _cross(body0, body1, *track.segments)


class Progress:
    """沿中心线累计走了多远,满一圈长度就是完赛。倒着开会减。"""

    def __init__(self, track: Track):
        self.track, self.length = track, track.length
        self.s, self.total = 0.0, 0.0

    def update(self, x: float, y: float) -> float:
        s, _ = self.track.project((x, y), near_s=self.s)
        self.total += (s - self.s + self.length / 2) % self.length - self.length / 2
        self.s = s
        return self.total


def run_lap(track: Track, controller, car_params: CarParams = CarParams(),
            time_limit: float = 120.0, noise_seed: int | None = None) -> LapResult:
    """controller(state: State) -> (目标车速 m/s, 目标前轮转角 rad),每 DT 秒调用一次。

    noise_seed 不为 None 时:真实车参数按 noisy_params 扰动,控制器看到的状态加 noisy_state 噪声
    (判定、计时、log 都用真值)。控制器应按标称的 car_params 构造。
    """
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    car = Car(noisy_params(car_params, rng) if rng else car_params)
    car.reset(*start_pose(track))
    progress = Progress(track)
    t, done, ctrl = 0.0, 0.0, 0.0
    log = [(t, car.x, car.y, car.yaw, car.v)]

    def result(finished, t_end, reason):
        return LapResult(finished, t_end, reason, np.array(log), ctrl / max(1, len(log) - 1),
                         1.0 if finished else max(0.0, done / progress.length))

    while t < time_limit:
        state = State(t, car.x, car.y, car.yaw, car.v, car.yaw_rate, car.accel, car.steer)
        if rng:
            state = noisy_state(state, rng)
        t0 = time.perf_counter()
        out = controller(state)
        ctrl += time.perf_counter() - t0
        try:
            v_cmd, steer_cmd = (float(u) for u in out)
        except (TypeError, ValueError):
            return result(False, t, 'bad_output')
        if not (math.isfinite(v_cmd) and math.isfinite(steer_cmd)):
            return result(False, t, 'bad_output')
        car.step(v_cmd, steer_cmd, DT)
        t += DT
        log.append((t, car.x, car.y, car.yaw, car.v))
        if car_off_track(track, car):
            return result(False, t, 'off_track')
        prev, done = done, progress.update(car.x, car.y)
        if done >= progress.length:         # 冲线时刻按这一步走过的比例插值
            return result(True, t - DT * (done - progress.length) / (done - prev), 'finished')
    return result(False, t, 'timeout')


class PathFollower:
    """沿给定路线开车:纯追踪算转角,速度照速度表给。

    path  (n,2) 闭合路线点(首尾不用重复),顺序要和行驶方向一致
    speed (n,)  每个路线点上想要的车速 m/s
    """

    def __init__(self, path, speed, car_params: CarParams = CarParams(), lookahead: float = 0.35,
                 lookahead_gain: float = 0.25, speed_preview: float = 0.35):
        path, speed = np.asarray(path, float), np.asarray(speed, float)
        closed = np.vstack([path, path[:1]])
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
        self.path = resample_closed(path, 0.05)
        s_new = np.linspace(0, s[-1], len(self.path), endpoint=False)
        self.speed = np.interp(s_new, s, np.append(speed, speed[0]))
        self.ds = s[-1] / len(self.path)
        self.wheelbase = car_params.wheelbase
        self.lookahead, self.lookahead_gain, self.speed_preview = lookahead, lookahead_gain, speed_preview
        self.idx = None

    def __call__(self, state: State) -> tuple[float, float]:
        n = len(self.path)
        if self.idx is None:        # 第一次:全局找最近点
            self.idx = int(np.argmin(np.hypot(*(self.path - (state.x, state.y)).T)))
        else:                       # 之后只往前小范围找,防止跳到隔壁那段
            window = (self.idx + np.arange(-10, 60)) % n
            d = np.hypot(*(self.path[window] - (state.x, state.y)).T)
            self.idx = int(window[np.argmin(d)])
        # 速度:看稍前方的速度表(车追速度有惯性,提前一点给)
        v_cmd = self.speed[(self.idx + int(state.v * self.speed_preview / self.ds)) % n]
        # 转角:纯追踪,瞄准前方 ld 处的路线点
        ld = self.lookahead + self.lookahead_gain * state.v
        target = self.path[(self.idx + max(1, int(ld / self.ds))) % n]
        dx, dy = target[0] - state.x, target[1] - state.y
        alpha = math.atan2(dy, dx) - state.yaw
        steer = math.atan2(2 * self.wheelbase * math.sin(alpha), math.hypot(dx, dy))
        return v_cmd, steer
