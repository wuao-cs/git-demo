#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LeetCode 200. 岛屿数量（Number of Islands）

难度：中等
标签：网格 BFS / DFS / 并查集 / Flood Fill / 连通分量

题目描述：
    给你一个由 '1'（陆地）和 '0'（水）组成的 m x n 二维网格 grid，
    请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由
    水平方向和/或竖直方向上相邻的陆地连接形成（斜向不算相邻）。
    此外，你可以假设网格的四条边均被水包围。

    示例 1：
        输入: grid = [
            ["1","1","1","1","0"],
            ["1","1","0","1","0"],
            ["1","1","0","0","0"],
            ["0","0","0","0","0"]
        ]
        输出: 1
    示例 2：
        输入: grid = [
            ["1","1","0","0","0"],
            ["1","1","0","0","0"],
            ["0","0","1","0","0"],
            ["0","0","0","1","1"]
        ]
        输出: 3

解题思路：
    本质是求"四连通分量个数"，经典 Flood Fill 问题。扫描每个格子，
    遇到未访问的陆地就把它所在的整块连通陆地"淹没"（标记为已访问），
    淹没次数即为岛屿数。三种实现：

    1) BFS（主解）：层序扩展，用 deque 队列。适合极大岛屿（一条长蛇形
       岛）也不会栈溢出；空间峰值是同层节点数。
    2) 显式栈 DFS：用 list 当栈，后进先出。思路与 BFS 几乎相同，只是
       出栈顺序不同；同样是迭代实现，规避 Python 递归深度限制
       （3.12 的 C 层递归硬上限约 1000，大网格递归 DFS 会 RecursionError）。
    3) 并查集（对照解）：每个陆地格初始化为独立集合，扫描时向右、向下
       合并相邻陆地，最终集合数即岛屿数。与遍历式解法原理完全不同，
       适合作为随机对拍的交叉验证基准。

复杂度分析（设 m、n 为网格行列数，k 为陆地格数）：
    - BFS / 显式栈 DFS：每个格子至多入队/入栈一次，O(m*n) 时间；
      visited 直接覆写在网格副本上，额外空间 O(m*n)（副本本身），
      队列/栈峰值 O(min(m, n))。
    - 并查集：近似 O(m*n * alpha)（alpha 为反阿克曼函数，近似常数），
      额外空间 O(m*n)。

思路对比：
    - 递归 DFS 写法最短，但大网格会触发递归深度限制，生产代码不推荐；
    - BFS 与迭代 DFS 时间空间同级，BFS 更贴近"逐层扩散"的直觉；
    - 并查集代码量最大，但适合"动态加陆地"的变体（如 LC 305
      岛屿数量 II），静态一次性统计时并不占优。
"""

from collections import deque
import random
import time
from typing import List

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # 四个方向：下、上、右、左


def _copy_grid(grid: List[List[str]]) -> List[List[str]]:
    """深拷贝网格，避免污染调用方传入的原数据。"""
    return [row[:] for row in grid]


def num_islands_bfs(grid: List[List[str]]) -> int:
    """主解：BFS 洪水填充，返回岛屿数量。时间 O(m*n)，空间 O(min(m,n)) 队列峰值。"""
    g = _copy_grid(grid)
    rows = len(g)
    if rows == 0:
        return 0
    cols = len(g[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != "1":  # 水或已淹没的陆地直接跳过
                continue
            count += 1  # 发现新岛屿，计数 +1
            g[r][c] = "0"  # 入队前立刻标记，防止重复入队
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] == "1":
                        g[nr][nc] = "0"  # 标记时机必须在入队时而非出队时
                        queue.append((nr, nc))
    return count


def num_islands_dfs(grid: List[List[str]]) -> int:
    """对照解：显式栈 DFS（迭代，规避递归深度限制）。时间 O(m*n)，空间同 BFS 量级。"""
    g = _copy_grid(grid)
    rows = len(g)
    if rows == 0:
        return 0
    cols = len(g[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != "1":
                continue
            count += 1
            stack = [(r, c)]
            g[r][c] = "0"
            while stack:
                cr, cc = stack.pop()  # 后进先出，深度优先扩散
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] == "1":
                        g[nr][nc] = "0"
                        stack.append((nr, nc))
    return count


def num_islands_union_find(grid: List[List[str]]) -> int:
    """对照解：并查集，与遍历式解法原理独立，用作随机对拍基准。近似 O(m*n*alpha)。"""
    g = _copy_grid(grid)
    rows = len(g)
    if rows == 0:
        return 0
    cols = len(g[0])
    parent = list(range(rows * cols))  # parent[一维下标] = 集合代表
    rank = [0] * (rows * cols)

    def find(x: int) -> int:
        # 路径压缩：沿途节点全部直接挂到根上
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径减半，与完全压缩效果相近
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # 已在同一集合，合并失败（不需要减计数）
        # 按秩合并：矮树挂到高树下
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    land = 0  # 陆地格总数
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != "1":
                continue
            land += 1
            idx = r * cols + c
            # 只需向右、向下合并即可覆盖所有相邻对，避免重复
            if c + 1 < cols and g[r][c + 1] == "1":
                if union(idx, idx + 1):
                    land -= 1  # 每成功合并一次，连通分量数减一
            if r + 1 < rows and g[r + 1][c] == "1":
                if union(idx, idx + cols):
                    land -= 1
    return land


def _random_grid(rows: int, cols: int, p: float, rng: random.Random) -> List[List[str]]:
    """生成随机网格，每个格子以概率 p 为陆地。"""
    return [["1" if rng.random() < p else "0" for _ in range(cols)] for _ in range(rows)]


def main() -> None:
    # ---------- 定点用例 ----------
    cases = [
        ("示例1：一整块岛",
         [["1", "1", "1", "1", "0"],
          ["1", "1", "0", "1", "0"],
          ["1", "1", "0", "0", "0"],
          ["0", "0", "0", "0", "0"]], 1),
        ("示例2：三座岛",
         [["1", "1", "0", "0", "0"],
          ["1", "1", "0", "0", "0"],
          ["0", "0", "1", "0", "0"],
          ["0", "0", "0", "1", "1"]], 3),
        ("全水", [["0", "0"], ["0", "0"]], 0),
        ("全陆地", [["1", "1", "1"], ["1", "1", "1"]], 1),
        ("单格陆地", [["1"]], 1),
        ("单格水", [["0"]], 0),
        ("斜向不相邻（四角 + 中心共五座岛）",
         [["1", "0", "1"],
          ["0", "1", "0"],
          ["1", "0", "1"]], 5),
        ("长蛇形岛（考验迭代实现不爆栈）",
         [["1"] * 2000 + ["0"]] + [["0"] * 2001 for _ in range(1)], 1),
    ]
    solvers = (
        ("BFS      ", num_islands_bfs),
        ("迭代 DFS ", num_islands_dfs),
        ("并查集   ", num_islands_union_find),
    )
    print("=== 定点用例 ===")
    all_pass = True
    for name, grid, expected in cases:
        results = [fn(grid) for _, fn in solvers]
        ok = all(res == expected for res in results)
        all_pass &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  期望={expected}  "
              f"BFS={results[0]} DFS={results[1]} 并查集={results[2]}")

    # ---------- 随机对拍：三解交叉验证 ----------
    print("=== 随机对拍（2000 组，网格 1~20 x 1~20）===")
    rng = random.Random(20261128)
    mismatches = 0
    for _ in range(2000):
        r = rng.randint(1, 20)
        c = rng.randint(1, 20)
        p = rng.choice((0.1, 0.3, 0.5, 0.8))
        grid = _random_grid(r, c, p, rng)
        res = {name: fn(grid) for name, fn in solvers}
        if len(set(res.values())) != 1:
            mismatches += 1
            if mismatches <= 3:
                print(f"  不一致! {res}  网格={grid}")
    all_pass &= mismatches == 0
    print(f"对拍完成：2000 组，不一致 {mismatches} 组")

    # ---------- 性能基准 ----------
    print("=== 性能基准（1000 x 1000，陆地概率 0.3）===")
    big = _random_grid(1000, 1000, 0.3, rng)
    for name, fn in solvers:
        t0 = time.perf_counter()
        ans = fn(big)
        cost = (time.perf_counter() - t0) * 1000
        print(f"  {name}: 岛屿数={ans}  耗时={cost:.1f} ms")

    print("=== 结果 ===")
    print("全部通过" if all_pass and mismatches == 0 else "存在失败项！")


if __name__ == "__main__":
    main()
