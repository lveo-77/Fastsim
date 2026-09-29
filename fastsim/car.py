"""阿克曼小车:运动学自行车模型 + 横向抓地力上限。参考点是后轴中心。

输入:目标车速 m/s、目标前轮转角 rad(左正)。车不会瞬间做到:
  · 速度按一阶惯性追目标(speed_tau),且加减速度有上限;
  · 舵机按 steer_rate 转过去;
  · 转弯需要的横向加速度 v²·tan(δ)/L 超过 a_lat_max 时抓不住,车少转(推头)。
"""
from __future__ import annotations

import math

import numpy as np

from fastsim.params import CarParams


class Car:
    def __init__(self, params: CarParams = CarParams()):
        self.p = params
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, y, yaw):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.v = 0.0
        self.steer = 0.0
        self.yaw_rate = 0.0
        self.accel = 0.0
        self.slip = 0.0         # 0=抓得住,接近 1=完全打滑

    def step(self, v_cmd: float, steer_cmd: float, dt: float):
        p = self.p
        target_v = min(max(v_cmd, 0.0), p.v_max)
        dv = (target_v - self.v) * min(1.0, dt / p.speed_tau)
        dv = min(max(dv, -p.decel_max * dt), p.accel_max * dt)
        self.accel = accel = dv / dt
        self.v += dv

        target_steer = min(max(steer_cmd, -p.max_steer), p.max_steer)
        self.steer += min(max(target_steer - self.steer, -p.steer_rate * dt), p.steer_rate * dt)

        # 摩擦圆:纵向加减速也占抓地力,横向可用的先扣掉那一份
        yaw_rate = self.v * math.tan(self.steer) / p.wheelbase
        a_lat = abs(self.v * yaw_rate)
        budget = math.sqrt(max(p.a_lat_max ** 2 - min(accel ** 2, p.a_lat_max ** 2), 0.0))
        self.slip = 0.0
        if a_lat > budget > 0.0:
            self.slip = 1.0 - budget / a_lat
            yaw_rate *= budget / a_lat
        self.yaw_rate = yaw_rate
        yaw_mid = self.yaw + 0.5 * yaw_rate * dt
        self.x += self.v * math.cos(yaw_mid) * dt
        self.y += self.v * math.sin(yaw_mid) * dt
        self.yaw = (self.yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi

    def corners(self) -> np.ndarray:
        """车体矩形四个角的世界坐标 (4,2)。"""
        p = self.p
        lx = np.array([-p.rear_overhang, p.length - p.rear_overhang] * 2)
        ly = np.array([p.width / 2] * 2 + [-p.width / 2] * 2)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack([self.x + lx * c - ly * s, self.y + lx * s + ly * c], axis=1)
