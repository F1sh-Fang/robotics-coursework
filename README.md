# robotics-coursework

控制 6 自由度球形腕工业机器人末端夹持毛笔完成哈尔滨工业大学校徽毛体书法“**哈工大**”三个字的书写。
系统直接从哈工大校徽图像中分割提取文字与真实粗细骨架，采用**三层解耦架构**（无时间 3D 几何路径、局部自适应动力学时间规划、锁定分支连续解析逆运动学），输出严格 1 ms 时间网格的高精度连续关节角度文本与仿真分析数据。

所有长度单位使用 mm，时间使用 s，内部关节角使用 rad，外部报告与控制器输入支持 deg。详细算法推导与实现细节请参阅 [doc/IMPLEMENTATION.md](file:///home/f1sh/Documents/master/机器人学/robotics-coursework/doc/IMPLEMENTATION.md)。

```text
robotics-coursework/
├── GenerateWritingTrajectory.py   # 生成入口（执行即可一键生成全部关节与末端轨迹）
├── PlotWritingTcpTrajectory.py    # 绘图入口（基于独立 FK 回算绘制 2D/3D 可视化图）
├── requirements-writing.txt       # 依赖清单
├── writing/                       # 核心实现包
│   ├── __init__.py
│   ├── geometry.py                # Layer 1: 3D 几何路径构建、字框排版、B 样条拟合与弧长参数化
│   ├── timing.py                  # Layer 2/3: 七次时间律、LocalTimeLaw 局部时间自适应分配与动力学校验
│   ├── kinematics.py              # POE 正运动学、球形腕解析逆解、最优分支锁定
│   ├── generation.py              # 轨迹全流程驱动、TXT/CSV/JSON 导出与严密闭环精度自检
│   ├── plotting.py                # 2D 毛笔接触圆盘模拟与 3D 笔尖/握笔 TCP 轨迹绘图
│   └── logo_extractor.py          # 校徽单字分割、距离变换 (DT)、笔画控制点中心线
├── logo/                          # 校徽原图、单字分割结果与提取笔画数据
│   ├── logo.png                   # 原始校徽图片
│   ├── hit_strokes.json           # 提取的书法笔顺、中心线与粗细信息数据库
│   └── extracted/                 # 哈、工、大单字图像、骨架与粗细热力图
├── doc/
│   └── IMPLEMENTATION.md          # 详细算法原理、数学推导与实现技术文档
├── tests/                         # 自动化单元测试套件
│   └── test_writing_trajectory.py
├── output/                        # 生成的控制器 TXT、仿真 CSV、报告 JSON 和轨迹图
└── log.md                         # 实机测试与调参演进记录
```

## 快速运行“哈工大”毛笔书写

### 1. 环境依赖安装

```bash
python -m pip install -r requirements-writing.txt
```

### 2. 一键生成书写轨迹

```bash
python GenerateWritingTrajectory.py
```

所有参数均已内置实机调优后的**最佳默认值**，开箱即用，控制流程全自动完成：
- 采样周期严格为 $1\text{ ms}$（1000 Hz）；
- 自动进行 100% 正运动学 (FK) 闭环数值验证（误差 $< 10^{-4}\text{ mm}$）；
- 自动生成 2D 笔顺粗细圆盘模拟图与 3D 空间轨迹图。

### 3. 自定义工作空间与动力学参数

也可根据实际机器人台面位置与工艺要求显式指定参数：

```bash
python GenerateWritingTrajectory.py \
  --brush-depth 5.0 \
  --z 593 \
  --x 480 620 \
  --y -130 130 \
  --lift 10 \
  --tip-offset 0 \
  --write-speed 40 \
  --travel-speed 70 \
  --vertical-speed 35 \
  --cart-accel 150 \
  --cart-jerk 3000
```

主要参数说明：
- `--brush-depth 5.0`：毛笔落笔下压最大进给量（单位 mm）。在细笔画/游丝处保持接触基准高度 $Z=593$，粗笔画与重顿处下压进给至 $Z=588$，软毫受压铺开，真实体现书法粗细浓淡；设为 `0` 时退化为完全平面书写；
- `--z 593`：笔尖接触写字纸面的基准高度（mm）；
- `--x 480 620`、`--y -130 130`：文字书写工作区域范围（mm），排版沿 +Y 方向行进，字形正上方朝向 -X；
- `--lift 10`：笔画间抬笔转移的安全间隙高度（mm）；
- `--tip-offset 0`：握笔 TCP 沿工具 +Z 轴到笔尖的长度偏移（mm）；
- `--write-speed 40`：纸面书写最大线速度上限（mm/s），用于 30 mm 以下的短笔画；
- `--long-write-speed 50`：长笔画书写速度上限（mm/s），90 mm 以上的笔画用此值，30~90 mm 之间线性过渡，设为 `0` 关闭；
- `--travel-speed 70`：空中平移最大线速度上限（mm/s）；
- `--vertical-speed 35`：提笔、落笔最大线速度上限（mm/s）；
- `--cart-accel 150`：笛卡尔空间加速度上限（mm/s²）；
- `--cart-jerk 3000`：笛卡尔空间加加速度上限（mm/s³）；
- `--joint-vmax-deg`、`--joint-amax-deg`：六关节速度、加速度上限，默认沿用实验四配置（30 30 30 40 40 60 deg/s；60 60 60 80 80 120 deg/s²）。

## 输出文件清单（生成于 `output/`）

| 文件 | 内容与格式 | 用途 |
|---|---|---|
| `output/writing_joint_deg.txt` | 六列 J1..J6，单位度（deg），空格分隔，无表头，保留 9 位小数 | **机器人控制器直接加载运行** |
| `output/writing_joint_timed_rad.txt` | 时间戳与六关节角度，单位弧度（rad），逗号分隔，带标准表头 | **MATLAB / Python 仿真与动力学分析** |
| `output/writing_tcp_trajectory.csv` | 时间、笔尖 3D 坐标 $(X,Y,Z)$、握笔 TCP 坐标及动作阶段 ID | **末端运动学轨迹与时序分析** |
| `output/writing_report.json` | 逆解构型、起始关节角、全轨迹峰值指标及 FK 闭环精度验证报告 | **自动化检验与运行参数归档** |
| `output/writing_layout_xy.png` | 2D 笔画布局图，含基于 Z 进给深度的毛笔物理接触圆盘叠加渲染与笔顺序号 | **字体粗细效果与排版检查** |
| `output/writing_tcp_trajectory_3d.png` | 从导出关节角回算的笔尖与握笔 TCP 三维立体空间轨迹图 | **三维姿态与干涉检查** |
| `output/writing_geometry.json` | 第一层无时间几何路径、分段样条控制点与元数据 | **几何分段与中间数据追溯** |

## 重新绘图

无需重新计算时间规划，直接基于 `output/` 中导出的关节文件回算绘图：

```bash
python PlotWritingTcpTrajectory.py
# 若需弹出交互式窗口展示，添加 --show
python PlotWritingTcpTrajectory.py --show
```

## 自动化测试

运行全部 12 项覆盖几何约束、毛笔进给、局部时间律、运动学逆解与动力学限制的自动化测试：

```bash
python -B -m unittest discover -s tests -v
```

