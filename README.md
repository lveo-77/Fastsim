# Fastsim:已知赛道,跑最快的一圈

比赛时,车第一圈边跑边建图;建完图以后,整条赛道的边界就都知道了。
这个作业模拟的就是这一步:**给你完整的赛道边界,写一个控制器,让车一圈跑得越快越好。**

赛道是比赛的 A、B、C、D 四套。比赛当天随机抽一套,所以你的程序要**同一份代码、四条都能跑**。

![四条赛道](docs/tracks.png)

## 环境

Python 3.10+:

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python grade.py              # 跑 A-D 四条,打印每圈用时
python grade.py --perturb 3  # 再加每条赛道 3 个扰动版(验收会跑,见下文)
python grade.py --plot       # 同时在 out/ 下画出轨迹(颜色是车速)
python -m pytest             # 检查仿真本身没被改坏
```

默认的 `homework/controller.py` 沿中心线、全程 1 m/s,四条都能跑完,总用时约 144 秒。

## 你要做什么

**只改 `homework/` 目录**(可以改文件、加文件、放模型权重)。评分只调用 `homework/controller.py` 里的 `Controller`:

```python
class Controller:
    def __init__(self, track, car_params):
        # 开跑前调用一次:拿到整条赛道和车参数,做准备(规划路线、加载模型……)
        ...

    def __call__(self, state):
        # 之后每 0.02 秒调用一次
        return v_cmd, steer_cmd     # 目标车速 m/s,目标前轮转角 rad(左正)
```

**方法不限**:规划 + 跟踪、PID、MPC、强化学习、DAgger……都可以。默认示例用的是「规划路线和速度表 + `fastsim.PathFollower` 纯追踪跟踪」,最省事的改法是只改其中的 `plan()`。

### 你能拿到的数据

开跑前(`__init__`):

| | 说明 |
|---|---|
| `track.left` / `track.right` | 左(内)、右(外)边界,闭合折线 |
| `track.centerline` | 中心线,5 cm 一个点,首点是起跑线 |
| `track.bounds()` | 中心线每点的左法向,以及到左、右边界的距离 |
| `track.on_track(points)` | 判断一批点在不在赛道上 |
| `track.project(xy)` | 点投影到中心线:走过的弧长、横向偏移 |
| `track.length` | 一圈长度 |
| `car_params` | 车的全部参数,见 [docs/车辆参数.md](docs/车辆参数.md) |

每步(`state`,精确值,只读):

| 字段 | 含义 |
|---|---|
| `t` | 本圈已用时间 s |
| `x`, `y` | 后轴中心位置 m |
| `yaw` | 车头朝向 rad |
| `v` | 车速 m/s |
| `yaw_rate` | 横摆角速度 rad/s |
| `accel` | 纵向加速度 m/s² |
| `steer` | 当前前轮实际转角 rad |

车的物理上限(极速 6 m/s、横向 0.8 G、加减速 0.4 G)和运动模型见 [docs/车辆参数.md](docs/车辆参数.md),代码在 `fastsim/car.py`。

## 规则

- 车从起跑线静止出发,沿中心线方向跑完一整圈计时(冲线时刻插值,精确到毫秒);
- **出界**:车身(0.56 × 0.35 m 矩形)碰到边界,即车的任何一个角出了赛道,或边界的任何一个折点进了车身;
- 超过 120 秒没跑完算超时;
- `Controller` 初始化限时 60 秒;每步平均计算时间不超过 20 ms(对应实车 50 Hz),单步不超过 1 秒;
- 返回值必须是两个有限的数,否则判接口错误;
- **成绩 = 所有验收赛道的用时之和,全部完赛才有成绩。**

### 验收怎么做

- 验收赛道 = A–D 原版 + 每条若干**扰动版**:边界折点随机抖几厘米、整张图随机旋转平移。扰动的随机种子验收当天才定。所以请写通用的方法,不要针对四条赛道分别背答案;
- 验收只取你的 `homework/` 目录,放进一份干净的本仓库里跑;`homework/` 以外改了什么都不生效(会列在报告里);
- 你的代码在单独的进程里运行,只能通过 `Controller` 的返回值影响车;
- 用到的第三方库(torch、scipy……)写进 `homework/requirements.txt`。

## 自己写训练环境(RL / DAgger 等)

仓库不提供现成的训练环境,怎么设计观测和奖励由你决定。但**判定请直接调用下面这些函数**,这样训练时和验收时完全一致:

```python
from fastsim import CarParams, DT, Progress, car_off_track, load_track, start_pose
from fastsim.car import Car

track = load_track('A')             # 或 load_track('A').perturbed(seed) 做数据增强
car = Car(CarParams())
car.reset(*start_pose(track))
progress = Progress(track)
while True:
    car.step(v_cmd, steer_cmd, DT)
    if car_off_track(track, car):   # 出界
        break
    if progress.update(car.x, car.y) >= progress.length:   # 跑完一圈
        break
```

## 思路提示(由易到难)

1. **直道快、弯道慢**:算出路线每一点的曲率 κ,弯道限速 `v ≤ sqrt(a_lat / κ)`;
2. **提前刹车**:车不能瞬间减速。从弯道往回倒推,保证每一段都来得及刹下来;出弯加速同理;
3. **别贴着极限**:一次出界比慢一秒亏得多,给横向加速度和离边界的距离都留余量;
4. **走赛车线**:外-内-外切弯,弯道半径更大,就能过得更快。注意 C 赛道左上角的直角,沿中心线的半径比车的最小转弯半径还小;
5. **换控制器**:纯追踪有滞后、会切弯。PID、MPC、学出来的策略都可能跟得更准、更快。

## 注意

- 赛道坐标是从规则图估读出来的,不是实地测量;
- 车的尺寸和转向参照实车,性能上限是按物理估的假设值,不代表实车能力。

## 助教:正式验收

在打了 tag 的干净 clone 里:

```bash
python accept.py 学生1/ 学生2/ ... --perturb 3 --seed <当天定的数> --csv 成绩.csv
```

学生代码会在本机执行,来源不可信时请在虚拟机或容器里跑;学生的 `homework/requirements.txt` 需要提前装好。
