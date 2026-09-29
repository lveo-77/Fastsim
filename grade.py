"""本地自测:在 A-D 上跑 homework/controller.py 的 Controller,打印每条赛道的圈速。

    python grade.py                 # A B C D 四条都跑
    python grade.py --tracks B      # 只跑 B
    python grade.py --perturb 3     # 再加每条赛道 3 个扰动版(地图扰动 + 车参数扰动 + 观测噪声)
    python grade.py --plot          # 每条赛道出一张图,存到 out/

正式验收用 accept.py,判定和这里完全一样,只是把你的代码放在单独进程里跑。
"""
import argparse
import importlib
import os
import traceback

import numpy as np

from fastsim import TRACKS, CarParams, load_track, run_lap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', nargs='+', default=list(TRACKS))
    ap.add_argument('--perturb', type=int, default=0, help='每条赛道再跑几个扰动版')
    ap.add_argument('--module', default='homework.controller', help='Controller 所在的模块')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    Controller = importlib.import_module(args.module).Controller
    car = CarParams()
    tracks = []
    for name in args.tracks:
        base = load_track(name)
        tracks += [(base, None)] + [(base.perturbed(seed), seed) for seed in range(1, args.perturb + 1)]
    times = []
    for track, seed in tracks:
        try:
            result = run_lap(track, Controller(track, car), car, noise_seed=seed)
        except Exception:
            traceback.print_exc()
            print(f'赛道 {track.name:5s}  你的代码报错')
            times.append(np.nan)
            continue
        print(f'赛道 {track.name:5s}  长 {track.length:5.1f}m  {result}  每步 {result.ctrl_time * 1e3:.2f}ms')
        times.append(result.time if result.finished else np.nan)
        if args.plot:
            import matplotlib.pyplot as plt
            from fastsim import viz
            os.makedirs('out', exist_ok=True)
            viz.plot(track, result, title=track.name)
            plt.savefig(f'out/track_{track.name}.png', dpi=120, bbox_inches='tight')
            plt.close()
    done = ~np.isnan(times)
    print(f'\n完赛 {done.sum()}/{len(times)}', end='')
    if done.all():
        print(f'   总用时 {np.sum(times):.3f}s')
    else:
        print('   有赛道没完赛,总用时不计')


if __name__ == '__main__':
    main()
