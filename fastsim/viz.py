"""画赛道和行驶轨迹(按车速上色)。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fastsim.sim import LapResult
from fastsim.track import Track


def plot(track: Track, result: LapResult | None = None, path=None, title: str = '', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    for edge in (track.left, track.right):
        closed = np.vstack([edge, edge[:1]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', lw=1.5)
    ax.plot(*track.centerline.T, ':', color='gray', lw=0.8)
    x, y = track.centerline[0]
    ax.plot([x, x], [y - 0.75, y + 0.75], 'g-', lw=3, label='起终点')
    if path is not None:
        path = np.asarray(path)
        ax.plot(*np.vstack([path, path[:1]]).T, '--', color='tab:orange', lw=1, label='规划路线')
    if result is not None:
        log = result.log
        sc = ax.scatter(log[:, 1], log[:, 2], c=log[:, 4], cmap='viridis', s=4)
        plt.colorbar(sc, ax=ax, label='车速 m/s')
        if not result.finished:
            ax.plot(log[-1, 1], log[-1, 2], 'rx', ms=12, mew=3)
        title = f'{title}  {result}'.strip()
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    return ax


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
