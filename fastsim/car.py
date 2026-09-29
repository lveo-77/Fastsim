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

SWEEP = 0.03        # 出界检查的位姿间隔 m:一步里每走这么远记一个中间位姿,防止两帧之间擦过边界


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
        self.trace = np.array([[self.x, self.y, self.yaw]])     # 上一步经过的位姿 (k,3),含终点

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
        # 中点积分;中间位姿按同一公式取走过 f 比例时的位置,终点就是 f=1
        f = np.arange(1, max(1, math.ceil(self.v * dt / SWEEP)) + 1) / max(1, math.ceil(self.v * dt / SWEEP))
        yaw_mid = self.yaw + 0.5 * yaw_rate * dt * f
        xs = self.x + self.v * np.cos(yaw_mid) * dt * f
        ys = self.y + self.v * np.sin(yaw_mid) * dt * f
        yaws = (self.yaw + yaw_rate * dt * f + math.pi) % (2 * math.pi) - math.pi
        self.trace = np.stack([xs, ys, yaws], axis=1)
        self.x, self.y, self.yaw = float(xs[-1]), float(ys[-1]), float(yaws[-1])

    def corners(self, poses=None) -> np.ndarray:
        """车体矩形四个角的世界坐标,按环顺序(后左、前左、前右、后右)。

        不给 poses 就是当前位姿,返回 (4,2);给 (k,3) 的位姿就返回 (k,4,2)。
        """
        p = self.p
        lx = np.array([-p.rear_overhang, p.length - p.rear_overhang, p.length - p.rear_overhang, -p.rear_overhang])
        ly = np.array([p.width / 2, p.width / 2, -p.width / 2, -p.width / 2])
        pose = np.array([[self.x, self.y, self.yaw]]) if poses is None else np.asarray(poses, float)
        c, s = np.cos(pose[:, 2:3]), np.sin(pose[:, 2:3])
        out = np.stack([pose[:, 0:1] + lx * c - ly * s, pose[:, 1:2] + lx * s + ly * c], axis=2)
        return out[0] if poses is None else out
