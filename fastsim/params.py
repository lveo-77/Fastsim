"""车参数。推导和出处见 docs/车辆参数.md。

尺寸、轴距、转角参照队里实车;性能上限(极速、抓地)是按物理估的假设值,不追求和实车一致。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CarParams:
    # --- 尺寸(参照实车)---
    wheelbase: float = 0.33         # 轴距 m
    length: float = 0.56            # 车长 m
    width: float = 0.35             # 车宽 m
    rear_overhang: float = 0.13     # 后轴到车尾 m
    max_steer: float = 0.379        # 前轮最大转角 rad(21.7°)
    steer_rate: float = 4.0         # 舵机转角速率 rad/s

    # --- 性能上限(假设)---
    mass: float = 3.0               # 整车质量 kg
    mu: float = 0.8                 # 轮胎与地面摩擦系数
    g: float = 9.81
    rear_load: float = 0.5          # 后轴承担的重量比例(后驱:只有后轮能出力、刹车)
    v_max: float = 6.0              # 极速 m/s
    speed_tau: float = 0.35         # 电机响应时间常数 s:给了目标速度,车要慢慢追上去

    @property
    def a_lat_max(self) -> float:
        """横向加速度上限 = μg,四个轮胎一起提供侧向力。"""
        return self.mu * self.g

    @property
    def accel_max(self) -> float:
        """最大加速度 = μg × 后轴载荷比,只有后轮驱动。"""
        return self.mu * self.g * self.rear_load

    @property
    def decel_max(self) -> float:
        """最大减速度:电机刹车同样只作用在后轮。"""
        return self.mu * self.g * self.rear_load
