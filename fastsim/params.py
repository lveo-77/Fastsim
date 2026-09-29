"""车参数。数值来自队里实车(实测/规则的标了出处,其余是合理假设)。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarParams:
    wheelbase: float = 0.33         # 轴距 m [实测]
    length: float = 0.56            # 车长 m
    width: float = 0.35             # 车宽 m
    rear_overhang: float = 0.13     # 后轴到车尾 m
    max_steer: float = 0.379        # 前轮最大转角 rad(21.7°)[实测]
    steer_rate: float = 4.0         # 舵机转角速率 rad/s
    v_max: float = 5.0              # 最高车速 m/s(实车极速没测;给 5 是为了让弯道成为约束,3m/s 在 A-D 上满油门就能过所有弯)
    speed_tau: float = 0.35         # 电机一阶响应时间常数 s:给了目标速度,车要慢慢追上去
    accel_max: float = 3.0          # 最大加速度 m/s²
    decel_max: float = 3.0          # 最大减速度 m/s²(实车刹车能力还没测,先给乐观值)
    a_lat_max: float = 7.85         # 横向加速度上限 m/s²(0.8g),超过就推头

