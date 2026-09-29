"""比赛赛道 A-D:左右两条闭合边界线 + 一条中心线。

数据来自规则图 9 估读(fastsim/tracks/X.json),不是实地测量,和真赛道会有几厘米到十几厘米的出入。
车逆时针行驶:left 是内边界,right 是外边界。centerline 首点是起跑线(START 框中间)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACKS = ('A', 'B', 'C', 'D')
_DIR = Path(__file__).resolve().parent / 'tracks'


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """把闭合折线按弧长等间距重采样。"""
    closed = np.vstack([points, points[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(8, int(s[-1] / spacing))
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """射线法:每个点是否在闭合多边形内。"""
    x, y = pts[:, 0:1], pts[:, 1:2]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cross = (y0 > y) != (y1 > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return ((cross & (x < xi)).sum(axis=1) % 2) == 1


@dataclass
class Track:
    name: str
    centerline: np.ndarray      # (n,2) 逆时针,约 5cm 一个点,首点是起跑线
    left: np.ndarray            # (m,2) 左(内)边界,闭合折线
    right: np.ndarray           # (k,2) 右(外)边界,闭合折线

    @property
    def length(self) -> float:
        return float(np.hypot(*np.diff(np.vstack([self.centerline, self.centerline[:1]]), axis=0).T).sum())

    def bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """中心线每个点的 (左法向单位向量 (n,2), 到左边界距离 (n,), 到右边界距离 (n,))。

        centerline[i] + t * normal[i] 在 -d_right[i] < t < d_left[i] 之间都在赛道上。
        """
        c = self.centerline
        tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
        tangent /= np.hypot(*tangent.T)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        dist = [np.hypot(*(c[:, None, :] - resample_closed(edge, 0.02)[None]).transpose(2, 0, 1)).min(axis=1)
                for edge in (self.left, self.right)]
        return normal, dist[0], dist[1]

    @property
    def vertices(self) -> np.ndarray:
        """两条边界的全部折点。"""
        return np.vstack([self.left, self.right])

    def perturbed(self, seed: int, jitter: float = 0.03) -> 'Track':
        """同一条赛道的扰动版:边界折点各自沿法向抖 ±jitter 米,再整体随机旋转、平移。

        验收时用它防止"认出是哪条赛道、调出背好的答案";估读数据本来也有这个量级的误差。
        """
        rng = np.random.default_rng(seed)

        def shake(edge):
            tangent = np.roll(edge, -1, axis=0) - np.roll(edge, 1, axis=0)
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / np.hypot(*tangent.T)[:, None]
            return edge + normal * rng.uniform(-jitter, jitter, len(edge))[:, None]

        th = rng.uniform(-math.pi, math.pi)
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        shift = rng.uniform(-5, 5, 2)
        move = lambda pts: pts @ rot.T + shift
        return Track(f'{self.name}~{seed}', move(self.centerline), move(shake(self.left)), move(shake(self.right)))

    def on_track(self, pts) -> np.ndarray:
        """每个点是否在赛道上(外边界之内、内边界之外)。"""
        pts = np.asarray(pts, float).reshape(-1, 2)
        return _inside(self.right, pts) & ~_inside(self.left, pts)

    def project(self, xy, near_s: float | None = None, window: float = 4.0) -> tuple[float, float]:
        """点投影到中心线,返回 (弧长 s, 横向偏移,左正)。

        near_s 给上一帧的弧长时只在它前后 window 米里找:赛道折回来的两段可能
        只隔一个赛道宽,全局找最近点会认到隔壁那段去。
        """
        pts = self.centerline
        seg = np.roll(pts, -1, axis=0) - pts
        seg_len2 = (seg ** 2).sum(axis=1)
        cum = np.concatenate([[0], np.cumsum(np.sqrt(seg_len2))])
        t = np.clip(((np.asarray(xy) - pts) * seg).sum(axis=1) / seg_len2, 0, 1)
        proj = pts + seg * t[:, None]
        dist = np.hypot(*(proj - xy).T)
        if near_s is not None:
            along = np.abs(cum[:-1] - near_s)
            dist = np.where(np.minimum(along, cum[-1] - along) <= window, dist, np.inf)
        i = int(np.argmin(dist))
        s = cum[i] + t[i] * math.sqrt(seg_len2[i])
        d = np.asarray(xy) - proj[i]
        lateral = (-d[0] * seg[i, 1] + d[1] * seg[i, 0]) / math.sqrt(seg_len2[i])
        return float(s), float(lateral)


def load_track(name: str) -> Track:
    d = json.loads((_DIR / f'{name}.json').read_text(encoding='utf-8'))
    return Track(name, resample_closed(np.array(d['centerline']), 0.05),
                 np.array(d['left']), np.array(d['right']))
