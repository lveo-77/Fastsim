import math
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from fastsim import TRACKS, CarParams, PathFollower, State, car_off_track, load_track, run_lap, start_pose
from fastsim.car import Car
from homework.controller import Controller


@pytest.mark.parametrize('name', TRACKS)
def test_track_data_is_consistent(name):
    t = load_track(name)
    assert 30 < t.length < 45
    assert t.on_track(t.centerline).all()                   # 中心线全在赛道上
    normal, dl, dr = t.bounds()
    assert (dl > 0.4).all() and (dr > 0.4).all()            # 规则赛道宽 1.25~1.5m
    assert (dl + dr < 1.8).all()
    # bounds() 给的是到边界的最近距离,沿法向走这么远一定还在赛道上
    assert t.on_track(t.centerline + normal * (dl - 0.01)[:, None]).all()
    assert t.on_track(t.centerline - normal * (dr - 0.01)[:, None]).all()
    # 直道上最近距离就是法向距离,再多走 5cm 就出去了
    assert not t.on_track(t.centerline[:1] + normal[:1] * (dl[0] + 0.05)).any()
    assert not t.on_track(t.centerline[:1] - normal[:1] * (dr[0] + 0.05)).any()


@pytest.mark.parametrize('name', TRACKS)
def test_default_homework_finishes(name):
    t = load_track(name)
    r = run_lap(t, Controller(t, CarParams()))
    assert r.finished
    assert abs(r.time - t.length / 1.0) < 1.0               # 1 m/s 跑一圈 ≈ 长度秒


def test_full_speed_on_centerline_goes_off_track():
    t = load_track('B')
    r = run_lap(t, PathFollower(t.centerline, np.full(len(t.centerline), CarParams().v_max)))
    assert not r.finished and r.reason == 'off_track'


def test_standing_still_times_out():
    r = run_lap(load_track('A'), lambda s: (0.0, 0.0), time_limit=2.0)
    assert r.reason == 'timeout'


@pytest.mark.parametrize('out', [(float('nan'), 0.0), (1.0, float('inf')), None, (1.0,), 'ab'])
def test_bad_controller_output_is_reported(out):
    assert run_lap(load_track('A'), lambda s: out).reason == 'bad_output'


def test_controller_gets_a_read_only_state():
    seen = []

    def ctrl(s):
        seen.append(s)
        with pytest.raises(FrozenInstanceError):
            s.x = 0.0
        return 1.0, 0.0
    run_lap(load_track('B'), ctrl, time_limit=0.5)
    assert isinstance(seen[0], State) and seen[0].t == 0.0 and seen[-1].v > 0


def test_boundary_vertex_poking_into_car_side_is_off_track():
    """C 左上角是直角,内边界的尖角可以从车身侧边插进去,而车的四个角都还在赛道上。"""
    t = load_track('C')
    car = Car()
    corner = t.left[np.argmin(t.left[:, 0] - t.left[:, 1])]     # 内边界左上角那个直角点
    found = False
    for yaw in np.linspace(-math.pi, math.pi, 72, endpoint=False):
        for back in np.linspace(0.05, 0.40, 8):                 # 让尖角落在车身侧边中段、往里 1cm
            c, s = math.cos(yaw), math.sin(yaw)
            for side in (1, -1):
                ly = side * (car.p.width / 2 - 0.01)
                car.reset(corner[0] - back * c + ly * s, corner[1] - back * s - ly * c, yaw)
                if t.on_track(car.corners()).all():
                    assert car_off_track(t, car)
                    found = True
    assert found


def test_finish_time_is_interpolated():
    t = load_track('B')
    r = run_lap(t, Controller(t, CarParams()))
    assert r.time % 0.02 > 1e-6                              # 不是整步数


@pytest.mark.parametrize('name', TRACKS)
def test_perturbed_track_is_valid_and_different(name):
    base = load_track(name)
    t = base.perturbed(7)
    assert t.on_track(t.centerline).all()
    assert abs(t.length - base.length) < 1e-6               # 刚体变换不改长度
    assert np.abs(t.centerline - base.centerline).max() > 0.5
    assert np.array_equal(t.left, base.perturbed(7).left)   # 同一个种子结果一样
    assert run_lap(t, Controller(t, CarParams())).finished
    x, y, _ = start_pose(t)
    assert t.on_track([[x, y]]).all()


def test_car_speed_lags_command_and_respects_limits():
    car = Car()
    car.step(3.0, 0.0, 0.02)
    assert 0 < car.v <= CarParams().accel_max * 0.02 + 1e-9
    for _ in range(500):
        car.step(3.0, 1.0, 0.02)
    assert car.v == pytest.approx(3.0, abs=1e-3)
    assert car.steer == pytest.approx(CarParams().max_steer)
    assert car.slip > 0                                     # 3m/s 满舵抓不住
    assert abs(car.v * car.yaw_rate) == pytest.approx(CarParams().a_lat_max, rel=1e-3)


# ---------------- 验收脚本 ----------------

def _student(tmp_path: Path, name: str, code: str) -> Path:
    d = tmp_path / name
    (d / 'homework').mkdir(parents=True)
    (d / 'homework' / '__init__.py').write_text('')
    (d / 'homework' / 'controller.py').write_text(textwrap.dedent(code), encoding='utf-8')
    return d


def _accept_one(student: Path, track_name='B'):
    import accept
    workdir = accept.workspace(accept.find_homework(student))
    return accept.run_one(workdir, load_track(track_name), CarParams())


def test_accept_runs_default_controller(tmp_path):
    root = Path(__file__).resolve().parents[1]
    d = _student(tmp_path, 'ok', (root / 'homework' / 'controller.py').read_text(encoding='utf-8'))
    t, reason, _ = _accept_one(d)
    assert reason == 'finished' and abs(t - load_track('B').length) < 1.0


def test_accept_is_immune_to_patching_the_car_model(tmp_path):
    d = _student(tmp_path, 'cheat', '''
        import fastsim.car, fastsim.sim
        fastsim.car.Car.step = lambda self, v, s, dt: setattr(self, 'x', self.x + 100)
        fastsim.sim.car_off_track = lambda track, car: False

        class Controller:
            def __init__(self, track, car):
                pass
            def __call__(self, state):
                return 1.0, 0.0
    ''')
    t, reason, _ = _accept_one(d)
    assert t is None and reason == 'off_track'               # 直行 1m/s,正常地在第一个弯出界


def test_accept_reports_crash_and_slow_controllers(tmp_path):
    crash = _student(tmp_path, 'crash', '''
        class Controller:
            def __init__(self, track, car):
                raise RuntimeError('boom')
    ''')
    assert _accept_one(crash)[1] == 'error'
    slow = _student(tmp_path, 'slow', '''
        import time
        class Controller:
            def __init__(self, track, car):
                pass
            def __call__(self, state):
                time.sleep(0.03)
                return 1.0, 0.0
    ''')
    assert _accept_one(slow)[1] == 'too_slow'


@pytest.mark.parametrize('layout', ['homework/', 'wrap/homework/', ''])
def test_accept_finds_homework_in_any_zip_layout(tmp_path, layout):
    import zipfile
    import accept
    z = tmp_path / '2026001_张三.zip'
    with zipfile.ZipFile(z, 'w') as f:
        f.writestr(layout + 'controller.py', 'class Controller: pass\n')
        f.writestr(layout + 'lib/controller.py', '')            # 更深的同名文件不能被选中
        f.writestr('__MACOSX/' + layout + 'controller.py', '')
    out = tmp_path / 'x'
    zipfile.ZipFile(z).extractall(out)
    hw = accept.find_homework(out)
    assert (hw / 'controller.py').read_text() == 'class Controller: pass\n'
