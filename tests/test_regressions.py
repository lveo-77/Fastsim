import math
from dataclasses import replace
import multiprocessing as mp
import shutil

import numpy as np
import pytest
from fastsim import CarParams, load_track, run_lap, State
from fastsim.car import Car, SWEEP
from fastsim.sim import _cross

@pytest.mark.parametrize('q0,q1,hit', [
    ([1,0],[2,0],True), ([.5,0],[1.5,0],True),
    ([2,0],[3,0],False), ([.5,-1],[.5,1],True),
    ([.5,0],[.5,0],True), ([.5,1e-6],[1,1e-6],False),
])
def test_segment_contact(q0,q1,hit):
    assert _cross(np.array([[0.,0.]]),np.array([[1.,0.]]),
                  np.array([q0],float),np.array([q1],float)) == hit

def test_turning_trace_bounds_corner_displacement_and_includes_start():
    car=Car()
    car.reset(1,2,math.pi-.001)
    car.v=6
    car.steer=car.p.max_steer
    car.step(6,car.steer,.02)
    assert np.allclose(car.trace[0], [1,2,math.pi-.001])
    corners=car.corners(car.trace)
    assert np.linalg.norm(np.diff(corners,axis=0),axis=-1).max() <= SWEEP+1e-10

def test_zero_grip_does_not_allow_turning():
    car=Car(replace(CarParams(),mu=0))
    car.v=2
    car.step(2,.3,.02)
    assert car.yaw_rate == 0

@pytest.mark.parametrize('dt',[0,-.1,float('nan'),float('inf')])
def test_invalid_step_duration(dt):
    with pytest.raises(ValueError):
        Car().step(1,0,dt)

@pytest.mark.parametrize('code,reason',[
    ('def __init__(self, track, car):\n        import os; os._exit(1)', 'error'),
    ('def __init__(self, track, car):\n        import time; time.sleep(5)', 'init_timeout'),
    ('def __init__(self, track, car): pass\n    def __call__(self, state):\n        import os; os._exit(1)', 'error'),
])
def test_failed_worker_is_reaped(tmp_path,monkeypatch,code,reason):
    import accept
    hw=tmp_path/'homework'
    hw.mkdir()
    (hw/'__init__.py').write_text('')
    (hw/'controller.py').write_text('class Controller:\n    '+code+'\n')
    monkeypatch.setattr(accept,'INIT_LIMIT',1.5)
    before={p.pid for p in mp.active_children()}
    work=accept.workspace(hw)
    try:
        assert accept.run_one(work,load_track('B'),CarParams())[1] == reason
    finally:
        shutil.rmtree(work)
    assert not ({p.pid for p in mp.active_children()}-before)


def test_lateral_limit_is_independent_of_longitudinal_limit():
    p = CarParams()
    assert p.a_lat_max / p.g == pytest.approx(.3)
    assert p.accel_max / p.g == pytest.approx(.4)
    assert p.decel_max / p.g == pytest.approx(.4)
    assert p.v_max == 6
    for command in (0, 6):
        car = Car(p)
        car.v = 3
        car.steer = p.max_steer
        car.step(command,p.max_steer,.02)
        assert abs(car.accel) / p.g == pytest.approx(.4)
        assert abs(car.v * car.yaw_rate) / p.g == pytest.approx(.3)
        assert math.hypot(car.accel,car.v*car.yaw_rate) <= p.mu*p.g

def test_side_grip_noise_is_reproducible_and_bounded():
    from fastsim import noisy_params
    p=CarParams()
    a=noisy_params(p,np.random.default_rng(7))
    b=noisy_params(p,np.random.default_rng(7))
    assert a == b and a.lateral_g != p.lateral_g
    assert .27 <= a.lateral_g <= .33
