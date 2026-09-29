"""比赛赛道 A-D:左右两条闭合边界线 + 一条中心线。

数据来自规则图 9 估读(fastsim/tracks/X.json),不是实地测量,和真赛道会有几厘米到十几厘米的出入。
车逆时针行驶:left 是内边界,right 是外边界。centerline 首点是起跑线(START 框中间)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
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


def _dist_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """每个点到闭合折线的最近距离(点到线段,精确)。"""
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    len2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)                  # 数据里有首尾重复的零长线段
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(axis=2) / len2[None], 0, 1)
    near = a[None] + t[..., None] * ab[None]
    return np.hypot(*(pts[:, None, :] - near).transpose(2, 0, 1)).min(axis=1)


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
        """中心线每个点的 (左法向单位向量 (n,2), 到左边界最近距离 (n,), 到右边界最近距离 (n,))。

        距离是点到边界折线的**最近**距离(精确值),不是沿法向量到边界的距离:
        在弯道、斜边处最近距离更小,所以它是保守的 ——
        以 centerline[i] 为圆心、d_left/d_right 中较小者为半径的圆内全在赛道上,
        单侧距离不保证沿该侧法向移动时不会遇到另一条边界;偏移路线仍需检查 on_track 和车身。
        """
        c = self.centerline
        tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
        tangent /= np.hypot(*tangent.T)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        return normal, _dist_to_polyline(c, self.left), _dist_to_polyline(c, self.right)

    @cached_property
    def vertices(self) -> np.ndarray:
        """两条边界的全部折点。"""
        return np.vstack([self.left, self.right])

    @cached_property
    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """两条边界的全部线段 (起点 (m,2), 终点 (m,2))。"""
        return (np.vstack([self.left, self.right]),
                np.vstack([np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)]))

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
