"""正式验收(助教用)。在打了 tag 的干净仓库里运行:

    python accept.py 学生目录1 学生目录2 ... [--perturb 3] [--seed 20261001] [--csv 成绩.csv]

学生目录 = 学生交上来的仓库根目录(fork 的 clone 或解压的 zip),里面要有 homework/。

做的事:
  1. 改动报告:homework/ 以外哪些文件和本仓库不一样(只报告,不影响跑分);
  2. 只取学生的 homework/,和本仓库的 fastsim/ 拼成临时工作区;
  3. 每条赛道(A-D 原版 + 每条 --perturb 个扰动版)单独起一个子进程运行学生的 Controller,
     仿真和判定在本进程里做,本进程从不导入学生代码 —— 学生改不了车模型和计时;
  4. 限制:Controller 初始化 60 秒;控制器每步平均 ≤ 20ms,单步 ≤ 1 秒;
  5. 成绩 = 全部赛道用时之和,全部完赛才有成绩。

学生代码会在本机执行,来源不可信的话请在虚拟机或容器里跑。
学生用到的第三方库(homework/requirements.txt)需要提前装好。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
INIT_LIMIT = 60.0       # Controller 初始化限时 s
STEP_MEAN_LIMIT = 0.020 # 每步平均耗时上限 s
STEP_HARD_LIMIT = 1.0   # 单步最长 s
SKIP = {'.git', '__pycache__', '.pytest_cache', 'out', '.venv', 'venv', '.idea', '.vscode'}


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

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_files(student: Path) -> list[str]:
    """homework/ 以外,和本仓库内容不同或多出来的文件。"""
    ours = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, encoding='utf-8',
                          check=True).stdout.split('\0')
    ours = {f for f in ours if f}
    ours = {f for f in ours if not f.startswith('homework/')}
    out = []
    for p in sorted(student.rglob('*')):
        rel = p.relative_to(student).as_posix()
        if not p.is_file() or rel.startswith('homework/') or SKIP & set(p.relative_to(student).parts):
            continue
        if rel not in ours:
            out.append(f'+ {rel}')
        elif _sha(p) != _sha(ROOT / rel):
            out.append(f'M {rel}')
    out += [f'- {f}' for f in sorted(ours) if not (student / f).exists()]
    return out


def workspace(student: Path) -> str:
    tmp = Path(tempfile.mkdtemp(prefix='fastsim_accept_'))
    shutil.copytree(ROOT / 'fastsim', tmp / 'fastsim', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(student / 'homework', tmp / 'homework', ignore=shutil.ignore_patterns('__pycache__'))
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
    ap.add_argument('students', nargs='+', type=Path)
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
    for student in args.students:
        student = student.resolve()
        print(f'\n===== {student.name} =====')
        if not (student / 'homework' / 'controller.py').is_file():
            print('  没有 homework/controller.py,跳过')
            rows.append({'student': student.name, 'total': '', 'note': '缺 homework/controller.py'})
            continue
        changes = changed_files(student)
        if changes:
            print('  homework/ 以外有改动(验收时一律用本仓库版本):')
            for c in changes:
                print('   ', c)
        workdir = workspace(student)
        row = {'student': student.name}
        times = []
        try:
            for track in tracks:
                t, reason, detail = run_one(workdir, track, car)
                shown = f'{t:.3f}s' if t is not None else reason
                print(f'  {track.name:8s} {shown:>12s}  {detail}')
                row[track.name] = shown
                times.append(t)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        ok = all(t is not None for t in times)
        row['total'] = f'{sum(times):.3f}' if ok else ''
        row['note'] = '; '.join(changes)
        print(f'  总用时 {row["total"]}' if ok else f'  完赛 {sum(t is not None for t in times)}/{len(times)},无成绩')
        rows.append(row)

    ranked = sorted((r for r in rows if r['total']), key=lambda r: float(r['total']))
    print('\n===== 排名 =====')
    for i, r in enumerate(ranked, 1):
        print(f'  {i:2d}. {r["student"]:20s} {r["total"]}s')
    if args.csv:
        fields = ['student'] + [t.name for t in tracks] + ['total', 'note']
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f'\n写入 {args.csv}')


if __name__ == '__main__':
    main()
