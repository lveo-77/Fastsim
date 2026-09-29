"""评分:在比赛赛道 A-D 上跑 homework/planner.py 的 plan(),打印每条赛道的圈速。

    python grade.py                 # A B C D 四条都跑
    python grade.py --tracks B      # 只跑 B
    python grade.py --plot          # 每条赛道出一张图,存到 out/
"""
import argparse
import importlib
import os

import numpy as np

from fastsim import TRACKS, CarParams, PathFollower, load_track, run_lap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', nargs='+', default=list(TRACKS))
    ap.add_argument('--planner', default='homework.planner', help='plan() 所在的模块')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    plan = importlib.import_module(args.planner).plan
    car = CarParams()
    times = []
    for name in args.tracks:
        track = load_track(name)
        path, speed = plan(track, car)
        result = run_lap(track, PathFollower(path, speed), car)
        print(f'赛道 {name}  长 {track.length:5.1f}m  {result}')
        times.append(result.time if result.finished else np.nan)
        if args.plot:
            import matplotlib.pyplot as plt
            from fastsim import viz
            os.makedirs('out', exist_ok=True)
            viz.plot(track, result, path, title=f'赛道 {name}')
            plt.savefig(f'out/track_{name}.png', dpi=120, bbox_inches='tight')
            plt.close()
    done = ~np.isnan(times)
    print(f'\n完赛 {done.sum()}/{len(times)}', end='')
    if done.all():
        print(f'   总用时 {np.sum(times):.2f}s')
    else:
        print('   有赛道出界或超时,总用时不计')


if __name__ == '__main__':
    main()
