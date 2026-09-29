"""画赛道和行驶轨迹(按车速上色)。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from fastsim.sim import LapResult
from fastsim.track import Track

# 有中文字体就用中文标签;没有(Linux/Mac 默认环境常见)就用英文,免得显示成方框
_CJK = [f for f in ('Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC', 'Noto Sans CJK SC',
                    'WenQuanYi Micro Hei') if any(x.name == f for x in font_manager.fontManager.ttflist)]
if _CJK:
    plt.rcParams['font.sans-serif'] = _CJK + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
_T = (dict(start='起终点', path='规划路线', speed='车速 m/s') if _CJK else
      dict(start='start/finish', path='planned path', speed='speed m/s'))


def plot(track: Track, result: LapResult | None = None, path=None, title: str = '', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    for edge in (track.left, track.right):
        closed = np.vstack([edge, edge[:1]])
        ax.plot(closed[:, 0], closed[:, 1], 'k-', lw=1.5)
    ax.plot(*track.centerline.T, ':', color='gray', lw=0.8)
    (x0, y0), (x1, y1) = track.centerline[0], track.centerline[1]
    n = np.array([y0 - y1, x1 - x0]) / np.hypot(x1 - x0, y1 - y0) * 0.75      # 起跑线垂直于中心线
    ax.plot([x0 - n[0], x0 + n[0]], [y0 - n[1], y0 + n[1]], 'g-', lw=3, label=_T['start'])
    if path is not None:
        path = np.asarray(path)
        ax.plot(*np.vstack([path, path[:1]]).T, '--', color='tab:orange', lw=1, label=_T['path'])
    if result is not None:
        log = result.log
        sc = ax.scatter(log[:, 1], log[:, 2], c=log[:, 4], cmap='viridis', s=4)
        plt.colorbar(sc, ax=ax, label=_T['speed'])
        if not result.finished:
            ax.plot(log[-1, 1], log[-1, 2], 'rx', ms=12, mew=3)
        brief = str(result) if _CJK else (f'{result.time:.3f}s' if result.finished else result.reason)
        title = f'{title}  {brief}'.strip()
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    return ax
