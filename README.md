# robotics-coursework

控制 6 自由度机械臂末端夹持毛笔完成哈尔滨工业大学校徽毛体书法“**哈工大**”三个字的书写。
程序分成无时间的 3D 几何路径、末端时间规划、连续关节逆解三层。
所有长度使用 mm，时间使用 s，内部关节角使用 rad。详细实现方法请参阅 [doc/IMPLEMENTATION.md](file:///home/f1sh/Documents/master/机器人学/robotics-coursework/doc/IMPLEMENTATION.md)。

```text
robotics-coursework/
├── GenerateWritingTrajectory.py   # 生成入口（执行即可一键生成全部关节轨迹）
├── PlotWritingTcpTrajectory.py    # 绘图入口
├── requirements-writing.txt
├── writing/                      # 核心实现（几何、动力学、运动学）
├── logo/                         # 学校 logo、单字分割图像与提取笔画数据
│   ├── logo.png                  # 原始校徽图片
│   ├── hit_strokes.json          # 提取的书法笔顺与粗细数据
│   └── extracted/                # 哈、工、大单字图像与粗细热力图
├── doc/
│   └── IMPLEMENTATION.md         # 算法与实现方法技术文档
├── tests/                        # 自动化测试
└── output/                       # 输出的 TXT、CSV、JSON 和绘制轨迹图
```

## 快速运行“哈工大”毛笔书写

```bash
python -m pip install -r requirements-writing.txt
python GenerateWritingTrajectory.py
```

所有参数均已内置最佳默认值，开箱即用。

也可根据工件位置显式指定主要参数：

```bash
python GenerateWritingTrajectory.py --brush-depth 1.0 --z 420 --x 450 500 --y 50 150 --lift 30 --tip-offset 20
```

- `--brush-depth 1.0`：毛笔落笔下压最大进给量（单位 mm）。在细笔画处保持接触面 $Z=420$，粗笔画与重顿处下压进给至 $Z=419$，软毫受压铺开，真实体现笔画粗细；设为 `0` 时退化为完全平面书写；
- `--z 420`：笔尖刚接触纸面的基准高度（mm）；
- `--x 450 500`、`--y 50 150`：书写工作区域范围（mm）；
- `--lift 30`：抬笔高度（mm）；
- `--tip-offset 20`：握笔位置沿工具 +Z 轴到笔尖的长度偏移（mm）。

## 输出文件清单（生成于 `output/`）

| 文件 | 内容与格式 |
|---|---|
| `output/writing_joint_deg.txt` | 六列 J1..J6，度，空格分隔，无表头，保留 9 位小数，机器人控制器直读 |
| `output/writing_joint_timed_rad.txt` | 时间戳与六关节角度，弧度，逗号分隔，带表头，可直接导入 MATLAB |
| `output/writing_tcp_trajectory.csv` | 时间、笔尖 3D 坐标 (X,Y,Z)、握笔 TCP 坐标及动作阶段 ID |
| `output/writing_report.json` | 逆解构型、起始关节角、全轨迹峰值指标及 FK 闭环精度验证报告 |
| `output/writing_layout_xy.png` | 笔画阅读方向与书写笔顺图 |
| `output/writing_tcp_trajectory_3d.png` | 从导出关节角回算的笔尖与握笔 TCP 三维轨迹图 |

## 重新绘图

```bash
python PlotWritingTcpTrajectory.py
```

## 自动化测试

```bash
python -B -m unittest discover -s tests -v
```
