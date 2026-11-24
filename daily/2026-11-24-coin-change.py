"""
LeetCode 322. 零钱兑换（Coin Change）
=====================================

难度：中等（Medium）
标签：动态规划 / 广度优先搜索 / 记忆化搜索 / 完全背包

题目描述
--------
给你一个整数数组 coins，表示不同面额的硬币；以及一个整数 amount，表示总金额。
计算并返回可以凑成总金额所需的 **最少的硬币个数**。如果没有任何一种硬币组合能凑成总金额，返回 -1。
你可以认为每种硬币的数量是无限的。

示例：
    输入：coins = [1, 2, 5], amount = 11
    输出：3          # 11 = 5 + 5 + 1

    输入：coins = [2], amount = 3
    输出：-1

约束：
    1 <= coins.length <= 12
    1 <= coins[i] <= 2^31 - 1
    0 <= amount <= 10^4

解题思路（主解：自底向上动态规划）
----------------------------------
1. 状态定义：dp[i] = 凑出金额 i 所需的最少硬币数。
2. 状态转移：最后一枚硬币面额为 c，则 dp[i] = min(dp[i - c] + 1)，对每个 c ∈ coins 且 c <= i。
   直观理解：凑 i 之前一定先凑出了 i - c，再多加一枚面额 c 的硬币。
3. 初始化：dp[0] = 0（金额 0 不需要硬币）；其余 dp[i] = amount + 1 作为"无穷大"哨兵
   （最大可能硬币数是 amount，即全用面额 1，所以 amount + 1 一定大于任何合法答案）。
4. 遍历顺序：外层枚举金额 i 从 1 到 amount，内层枚举硬币。因为完全背包允许重复选取，
   内层用正序即可复用本轮已更新的 dp 值。
5. 答案：dp[amount] 若仍为哨兵值则返回 -1。

复杂度分析
----------
时间：O(amount * n)，n = len(coins)。两层循环，每个状态转移 O(1)。
空间：O(amount)，一维滚动数组；不递归故无调用栈开销。

思路对比（同一份代码里给出三种可对拍的实现）
----------------------------------------------
| 实现                | 时间              | 空间        | 特点                              |
|---------------------|-------------------|-------------|-----------------------------------|
| 自底向上 DP（主解） | O(amount * n)     | O(amount)   | 无递归、无爆栈风险，常数小，工程首选 |
| 自顶向下记忆化 DFS  | O(amount * n)     | O(amount)   | 只计算可达状态，思路最贴近"暴力 + 剪枝"；但真递归会受解释器深度限制，故用显式栈实现 |
| BFS 按层搜索        | O(amount * n)     | O(amount)   | 把金额当图上的点、硬币当边权为 1 的边，第一次到达 amount 的层数即答案，天然求"最少步数" |

三者渐进复杂度相同，DP 与记忆化是"按状态聚合"，BFS 是"按层扩展"。
本题若把 amount 换成"凑出不超过 amount 的最大值"（即 LeetCode 322 的变体/518 零钱兑换 II 计数版），
DP 的写法只需替换聚合函数（min → max / sum），这也是 DP 相对 BFS 更易迁移的原因。
"""

from collections import deque
from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    """自底向上动态规划求最少硬币数。凑不出返回 -1。"""
    if amount < 0:
        return -1
    if amount == 0:
        return 0

    # 过滤掉大于目标金额的硬币：它们永远不可能被用上（面额均为正数）
    usable: List[int] = [c for c in coins if 0 < c <= amount]
    if not usable:
        return -1

    unreachable: int = amount + 1          # 哨兵值，代表"无穷大"
    dp: List[int] = [unreachable] * (amount + 1)
    dp[0] = 0                              # 金额 0 需要 0 枚硬币

    for target in range(1, amount + 1):
        for coin in usable:
            if coin > target:
                continue                   # 面额超过当前目标，跳过
            # 关键转移：先凑出 target - coin，再加这一枚
            candidate: int = dp[target - coin] + 1
            if candidate < dp[target]:
                dp[target] = candidate

    return -1 if dp[amount] == unreachable else dp[amount]


def coin_change_memo(coins: List[int], amount: int) -> int:
    """自顶向下：记忆化 DFS（等价于递归版 DP），用于对拍验证。

    这里用**显式栈**模拟后序 DFS，而不是真的递归：最坏递归深度约为 amount / min(coins)，
    amount=8839、面额含 3 时就有近 3000 层，Python 3.12 起还有独立的 C 层递归硬上限，
    `sys.setrecursionlimit` 也绕不过去（文件头"思路对比"里提到的递归风险，实测确实会炸）。
    显式栈版本保持了"先算子问题、再汇总"的自顶向下顺序，但不受解释器深度限制。
    """
    if amount < 0:
        return -1
    if amount == 0:
        return 0
    usable: List[int] = sorted({c for c in coins if 0 < c <= amount})
    if not usable:
        return -1

    unreachable: int = amount + 1
    memo: dict = {0: 0}                     # 已求解的子问题：剩余金额 -> 最少硬币数
    stack: List[List[int]] = [[amount, 0]]  # [剩余金额, 下一个待扩展的硬币下标]

    while stack:
        rest, idx = stack[-1]
        if rest in memo:                    # 该子问题已算出，直接出栈
            stack.pop()
            continue
        if idx < len(usable):
            stack[-1][1] = idx + 1          # 记录"这个硬币已扩展过"
            coin: int = usable[idx]
            child: int = rest - coin
            if 0 < child and child not in memo:
                stack.append([child, 0])    # 先深入子问题（后序）
        else:
            # 所有子问题就绪，汇总出当前 rest 的最优值
            best: int = unreachable
            for coin in usable:
                if coin > rest:
                    break                   # usable 升序，后面的面额更大
                candidate: int = memo[rest - coin] + 1
                if candidate < best:
                    best = candidate
            memo[rest] = best
            stack.pop()

    ans: int = memo[amount]
    return -1 if ans >= unreachable else ans


def coin_change_bfs(coins: List[int], amount: int) -> int:
    """BFS 按层扩展：图的节点是"剩余金额"，边是"减去一枚硬币"，边权为 1。

    第一次到达 0 时所处的层数就是最少硬币数（BFS 的最短路性质）。
    """
    if amount < 0:
        return -1
    if amount == 0:
        return 0
    usable: List[int] = [c for c in coins if 0 < c <= amount]
    if not usable:
        return -1

    visited: List[bool] = [False] * (amount + 1)
    queue: deque[int] = deque([amount])
    visited[amount] = True
    steps: int = 0                     # 已使用的硬币数（层号）

    while queue:
        steps += 1
        # 逐层扩展：本层所有节点都再用掉一枚硬币
        for _ in range(len(queue)):
            remaining: int = queue.popleft()
            for coin in usable:
                nxt: int = remaining - coin
                if nxt == 0:
                    return steps       # 首次到达 0 即为最优
                if nxt > 0 and not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)

    return -1                          # 队列耗尽仍未到达 0


def _check(coins: List[int], amount: int, expected: int) -> bool:
    """对单个用例同时跑三种实现，断言与期望值一致，并打印结果。"""
    got_dp: int = coin_change(coins, amount)
    got_memo: int = coin_change_memo(coins, amount)
    got_bfs: int = coin_change_bfs(coins, amount)
    ok: bool = got_dp == got_memo == got_bfs == expected
    flag: str = "PASS" if ok else "FAIL"
    print(
        f"[{flag}] coins={coins!s:<26} amount={amount:<6} "
        f"期望={expected:<4} DP={got_dp:<4} 记忆化={got_memo:<4} BFS={got_bfs}"
    )
    return ok


if __name__ == "__main__":
    # 说明：每个用例都覆盖了一种边界/典型形态
    cases: List[tuple] = [
        # 1) 经典样例：需要贪心地多用大面额
        ([1, 2, 5], 11, 3),
        # 2) 边界：单一面额且凑不出（amount 不是面额倍数）
        ([2], 3, -1),
        # 3) 边界：amount = 0，答案应为 0 而不是 -1
        ([1], 0, 0),
        # 4) 边界：只有一枚硬币且刚好凑出
        ([7], 7, 1),
        # 5) 经典易错样例：贪心会失败，必须靠 DP 回溯组合（官方测试用例）
        ([186, 419, 83, 408], 6249, 20),
        # 6) 边界：所有面额都大于 amount，直接无解
        ([5, 10], 3, -1),
        # 7) 大金额 + 稀疏面额：验证性能与正确性（官方测试用例）
        ([3, 7, 405, 436], 8839, 25),
    ]

    passed: int = 0
    for coins, amount, expected in cases:
        if _check(coins, amount, expected):
            passed += 1

    # 随机对拍：三种实现互相印证（纯标准库，固定种子保证可复现）
    import random

    random.seed(20261124)
    mismatch: int = 0
    for _ in range(300):
        n: int = random.randint(1, 5)
        coins: List[int] = random.sample(range(1, 25), n)
        amount: int = random.randint(0, 120)
        a, b, c = (
            coin_change(coins, amount),
            coin_change_memo(coins, amount),
            coin_change_bfs(coins, amount),
        )
        if not (a == b == c):
            mismatch += 1
            print(f"[MISMATCH] coins={coins} amount={amount} DP={a} memo={b} bfs={c}")

    print(f"\n定点用例：{passed}/{len(cases)} 通过；随机对拍 300 组，不一致 {mismatch} 组。")
    assert passed == len(cases), "定点用例未全部通过"
    assert mismatch == 0, "三实现结果不一致"
    print("全部用例通过 ✅")
