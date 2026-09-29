"""正式验收(助教用)。在本仓库最新 main 的干净 clone 里运行:

    python accept.py 提交1.zip 提交2.zip ... [--perturb 3] [--seed <当天定>] [--csv 结果.csv]

学生只交 homework/ 目录打的压缩包(学号_姓名.zip);也可以给解压后的目录。
包里 homework/ 在第一层、套了一层文件夹、或者直接就是 homework 的内容,都能认出来。

做的事:
  1. 只取学生的 homework/,和本仓库的 fastsim/ 拼成临时工作区;
  2. 每条赛道(A-D 原版 + 每条 --perturb 个扰动版)单独起一个子进程运行学生的 Controller,
     仿真和判定在本进程里做,本进程从不导入学生代码 —— 学生改不了车模型和计时;
  3. 限制:Controller 初始化 60 秒;控制器每步平均 ≤ 20ms,单步 ≤ 1 秒;
  4. 报告每条赛道是否完赛(圈速仅供参考,不排名),以及有没有交 思路.md。

学生代码会在本机执行,来源不可信的话请在虚拟机或容器里跑。
学生用到的第三方库(homework/requirements.txt)需要提前装好。
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INIT_LIMIT = 60.0       # Controller 初始化限时 s
STEP_MEAN_LIMIT = 0.020 # 每步平均耗时上限 s
STEP_HARD_LIMIT = 1.0   # 单步最长 s


# ---------------- 子进程:只跑学生代码 ----------------

def _worker(conn, workdir: str):
    os.chdir(workdir)
    sys.path.insert(0, workdir)
    try:
        from fastsim import State
        from homework.controller import Controller
        track, car = conn.recv()
        ctrl = Controller(track, car)
        conn.send(('ok', None))
        import time
        while True:
            msg = conn.recv()
            if msg is None:
                return
            t0 = time.perf_counter()
            out = ctrl(State(*msg))
            dt = time.perf_counter() - t0
            try:
                v, steer = (float(u) for u in out)
            except (TypeError, ValueError):
                conn.send(('bad_output', repr(out)[:200]))
                return
            conn.send(('ok', (v, steer, dt)))
    except Exception:
        conn.send(('error', traceback.format_exc()[-2000:]))


class _Abort(Exception):
    def __init__(self, reason, detail=''):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class _Remote:
    """本进程里的控制器替身:把状态发给子进程,取回指令。"""

    def __init__(self, workdir: str, track, car):
        ctx = mp.get_context('spawn')
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, workdir), daemon=True)
        self.proc.start()
        self.conn.send((replace(track, name='?'), car))       # 不告诉学生是哪条赛道
        self.step_times = []
        self._expect(INIT_LIMIT, 'init_timeout')

    def _expect(self, limit, timeout_reason):
        if not self.conn.poll(limit):
            raise _Abort(timeout_reason)
        try:
            kind, payload = self.conn.recv()
        except EOFError:
            raise _Abort('error', '子进程退出')
        if kind != 'ok':
            raise _Abort(kind, payload)
        return payload

    def __call__(self, state):
        self.conn.send(astuple(state))
        v, steer, dt = self._expect(STEP_HARD_LIMIT, 'too_slow')
        self.step_times.append(dt)
        return v, steer

    def close(self):
        try:
            self.conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.proc.join(2)
        if self.proc.is_alive():
            self.proc.kill()


# ---------------- 主进程:判定 ----------------

def find_homework(root: Path) -> Path | None:
    """在学生提交里找 homework 目录:含 controller.py 的最浅一层。"""
    hits = sorted((p.parent for p in root.rglob('controller.py') if '__MACOSX' not in p.parts),
                  key=lambda d: (d.name != 'homework', len(d.parts)))
    return hits[0] if hits else None


def workspace(homework: Path) -> str:
    tmp = Path(tempfile.mkdtemp(prefix='fastsim_accept_'))
    shutil.copytree(ROOT / 'fastsim', tmp / 'fastsim', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(homework, tmp / 'homework', ignore=shutil.ignore_patterns('__pycache__'))
    return str(tmp)


def run_one(workdir: str, track, car):
    from fastsim import run_lap
    remote = None
    try:
        remote = _Remote(workdir, track, car)
        r = run_lap(track, remote, car)
        mean = float(np.mean(remote.step_times)) if remote.step_times else 0.0
        if r.reason == 'bad_output':
            return None, 'bad_output', 'NaN/inf'
        if mean > STEP_MEAN_LIMIT:
            return None, 'too_slow', f'每步平均 {mean * 1e3:.1f}ms'
        return (r.time if r.finished else None), r.reason, f'每步 {mean * 1e3:.2f}ms'
    except _Abort as e:
        return None, e.reason, str(e.detail).strip().splitlines()[-1] if e.detail else ''
    finally:
        if remote:
            remote.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('students', nargs='+', type=Path, help='学生提交的 zip 或目录')
    ap.add_argument('--perturb', type=int, default=3, help='每条赛道的扰动版数量')
    ap.add_argument('--seed', type=int, default=20261001, help='扰动种子,验收当天再定,别提前公开')
    ap.add_argument('--csv', type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from fastsim import TRACKS, CarParams, load_track
    car = CarParams()
    tracks = []
    for name in TRACKS:
        base = load_track(name)
        tracks += [base] + [base.perturbed(args.seed + k) for k in range(args.perturb)]

    rows = []
    for sub in args.students:
        sub = sub.resolve()
        name = sub.stem if sub.suffix == '.zip' else sub.name
        print(f'\n===== {name} =====')
        unpacked = None
        if sub.suffix == '.zip':
            unpacked = Path(tempfile.mkdtemp(prefix='fastsim_zip_'))
            with zipfile.ZipFile(sub) as z:
                z.extractall(unpacked)
            sub = unpacked
        homework = find_homework(sub)
        row = {'student': name}
        if homework is None:
            print('  找不到 controller.py,跳过')
            row['note'] = '缺 controller.py'
            rows.append(row)
            continue
        idea = homework / '思路.md'
        row['思路.md'] = '有' if idea.is_file() and idea.stat().st_size > 0 else '缺'
        workdir = workspace(homework)
        done = 0
        try:
            for track in tracks:
                t, reason, detail = run_one(workdir, track, car)
                shown = f'{t:.3f}s' if t is not None else reason
                print(f'  {track.name:14s} {shown:>12s}  {detail}')
                row[track.name] = shown
                done += t is not None
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            if unpacked:
                shutil.rmtree(unpacked, ignore_errors=True)
        row['完赛'] = f'{done}/{len(tracks)}'
        print(f'  完赛 {row["完赛"]}   思路.md {row["思路.md"]}')
        rows.append(row)

    if args.csv:
        fields = ['student', '完赛', '思路.md'] + [t.name for t in tracks] + ['note']
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f'\n写入 {args.csv}')

if __name__ == '__main__':
    main()
