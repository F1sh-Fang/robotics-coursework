"""哈工大校徽毛笔书法轨迹生成 CLI 命令行入口。

直接执行本脚本即可生成 1 ms 离散关节角度文本文件 (TXT) 及末端数据 (CSV/JSON)，
核心业务实现封装于 writing 包中。

运行示例：
    python GenerateWritingTrajectory.py
    python GenerateWritingTrajectory.py --brush-depth 5.0 --z 593 --x 480 620 --y -130 130
"""
from writing.generation import main

if __name__ == '__main__':
    main()

