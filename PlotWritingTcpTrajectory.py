"""机械臂末端与关节轨迹可视化绘图入口。

从 output/ 目录读取导出的关节角文本，利用独立正运动学回算末端笔尖与握笔点三维轨迹，
生成 3D 轨迹图 (writing_tcp_trajectory_3d.png) 与 2D 排版粗细圆盘模拟图 (writing_layout_xy.png)。

运行示例：
    python PlotWritingTcpTrajectory.py
    python PlotWritingTcpTrajectory.py --show
"""
from writing.plotting import main

if __name__ == '__main__':
    main()

