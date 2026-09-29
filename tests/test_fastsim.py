import math

import numpy as np
import pytest

from fastsim import TRACKS, CarParams, PathFollower, load_track, run_lap
from homework.planner import plan


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
def test_baseline_homework_finishes(name):
    t = load_track(name)
    r = run_lap(t, PathFollower(*plan(t, CarParams())))
    assert r.finished
    assert abs(r.time - t.length / 1.0) < 1.0               # 1 m/s 跑一圈 ≈ 长度秒


def test_full_speed_on_centerline_goes_off_track():
    t = load_track('B')
    r = run_lap(t, PathFollower(t.centerline, np.full(len(t.centerline), CarParams().v_max)))
    assert not r.finished and r.reason == 'off_track'


def test_standing_still_times_out():
    r = run_lap(load_track('A'), lambda car: (0.0, 0.0), time_limit=2.0)
    assert r.reason == 'timeout'


def test_car_speed_lags_command_and_respects_limits():
    from fastsim.car import Car
    car = Car()
    car.step(3.0, 0.0, 0.02)
    assert 0 < car.v <= CarParams().accel_max * 0.02 + 1e-9
    for _ in range(500):
        car.step(3.0, 1.0, 0.02)
    assert car.v == pytest.approx(3.0, abs=1e-3)
    assert car.steer == pytest.approx(CarParams().max_steer)
    assert car.slip > 0                                     # 3m/s 满舵抓不住
    assert car.v * abs(car.v * math.tan(car.steer) / car.p.wheelbase) * (1 - car.slip) \
        == pytest.approx(CarParams().a_lat_max, rel=1e-3)
