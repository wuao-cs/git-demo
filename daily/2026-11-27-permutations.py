#!/usr/bin/env python3
"""LC 46. 全排列（Permutations）

难度：Medium
标签：回溯、递归、深度优先搜索

题目描述：
    给定一个不含重复数字的整数数组 nums，返回其所有可能的全排列，
    可以按任意顺序返回答案。
    示例：
        输入: nums = [1,2,3]
        输出: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

解题思路：
    全排列是回溯法的入门经典：把「选数」看成一条逐层深入的路径，
    每层从未被占用的数字里挑一个放到当前位置，选完递归处理下一层；
    回退时撤销选择（恢复现场），让同层兄弟分支能尝试别的数字。

    本文件给出三种实现互相对拍：
    1) permute_used：回溯 + used[] 占用标记数组（主解，最通用、最易讲清思路）；
    2) permute_swap：回溯 + 原地交换。把「候选」始终维护在 nums[pos:] 后缀里，
       每层把一个候选换到 pos 位置，不额外开 used/path，空间常数更小；
    3) permute_insert：递归分治对照解。「n 个数的排列 = 每个数字打头
       × 其余 n-1 个数的排列」，逐个把首元素插入到子排列的每个缝隙里。

    思路对比：
    - used 版：路径显式存 path，回溯模板最标准，适合推广到有重复数字
      （LC 47，需先排序 + 相邻剪枝）和组合类题目；
    - swap 版：无 path/used，空间占用最小，但原数组被改动，且不便于
      直接扩展到「字典序去重排列」；若要求还原可先拷贝一份；
    - insert 版：最贴近数学定义，代码短，但每次插入都是 O(n) 拷贝，
      常数开销大，实践中不如前两种。

复杂度分析（以 n = len(nums) 为例，排列总数 n!）：
    - 时间：三种解都是 O(n * n!)——共 n! 个排列，每个排列要 O(n) 复制到结果
      （swap 版每层交换 O(1)，但最终收集仍是 O(n)）；
    - 空间：
        used 版  O(n) 递归栈 + O(n) path/used；
        swap 版  O(n) 递归栈（结果集不计入）；
        insert 版递归栈 O(n)，但中间列表拷贝使实际常数显著更大。
    输出结果集本身占 O(n * n!)，所有解法相同。

测试策略：
    - 边界：空数组（返回 [[]]，唯一「空排列」）、单元素；
    - n=2、n=3 全量枚举并与暴力生成（基于排序穷举）核对；
    - n=4 校验排列个数 = 4! = 24 且内部两两一致；
    - 随机对拍：三实现输出（排序后）完全一致。
"""

from __future__ import annotations

import random
from typing import List


def permute_used(nums: List[int]) -> List[List[int]]:
    """主解：回溯 + used 标记数组。

    每层从「未被使用」的数字里选一个追加到 path，递归后撤销。
    """
    n = len(nums)
    res: List[List[int]] = []
    path: List[int] = []
    used = [False] * n  # used[i] 表示 nums[i] 是否已在当前路径上

    def backtrack() -> None:
        # 路径长度等于 n，说明凑满一个排列，复制入结果集
        if len(path) == n:
            res.append(path[:])  # 必须浅拷贝，否则后续回退会污染结果
            return
        for i in range(n):
            if used[i]:
                continue  # 跳过已在路径上的数字（剪枝）
            used[i] = True  # 做选择
            path.append(nums[i])
            backtrack()  # 递归处理下一层
            path.pop()  # 撤销选择，恢复现场
            used[i] = False

    backtrack()
    return res


def permute_swap(nums: List[int]) -> List[List[int]]:
    """对照解：回溯 + 原地交换。

    把候选始终维护在 nums[pos:] 后缀：每层把某个候选交换到 pos，
    固定 pos 后递归处理 pos+1。不使用额外 used/path 数组。
    注意：会改动传入的 nums 本身。
    """
    n = len(nums)
    res: List[List[int]] = []
    arr = nums[:]  # 拷贝一份，避免污染调用方数据（也便于对拍）

    def backtrack(pos: int) -> None:
        if pos == n:  # 后缀为空，整个数组即为一个排列
            res.append(arr[:])
            return
        for i in range(pos, n):
            arr[pos], arr[i] = arr[i], arr[pos]  # 候选换到当前位置
            backtrack(pos + 1)
            arr[pos], arr[i] = arr[i], arr[pos]  # 换回来，恢复现场

    backtrack(0)
    return res


def permute_insert(nums: List[int]) -> List[List[int]]:
    """对照解：递归分治（首元素枚举 + 子排列插入）。

    perm(n) = { x + p | x ∈ nums, p ∈ perm(nums - {x}) }，
    把打头元素 x 依次插入子排列的每个缝隙得到完整排列。
    """
    if len(nums) <= 1:  # 边界：空数组返回 [[]]（唯一空排列），单元素直接返回
        return [nums[:]]
    res: List[List[int]] = []
    for i, first in enumerate(nums):
        rest = nums[:i] + nums[i + 1:]  # 去掉打头元素后的子问题
        for sub in permute_insert(rest):
            res.append([first] + sub)  # 首元素放到子排列最前面
    return res


def brute_force(nums: List[int]) -> List[List[int]]:
    """对拍基准：基于「字典序下一排列」的暴力枚举（不递归）。

    对排序后的数组反复调用 next_permutation，穷尽全部排列。
    """
    if not nums:
        return [[]]
    arr = sorted(nums)
    res: List[List[int]] = []
    total = 1
    for k in range(2, len(arr) + 1):
        total *= k  # 预先算 n!，控制循环次数

    def next_permutation(a: List[int]) -> bool:
        # 从右向左找第一个升序对 a[i] < a[i+1]
        i = len(a) - 2
        while i >= 0 and a[i] >= a[i + 1]:
            i -= 1
        if i < 0:
            return False  # 已是最大排列
        # 找到 i 右侧最后一个大于 a[i] 的位置并交换
        j = len(a) - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
        a[i + 1:] = reversed(a[i + 1:])  # 后缀反转成最小字典序
        return True

    for _ in range(total):
        res.append(arr[:])
        if not next_permutation(arr):
            break
    return res


def main() -> None:
    """定点用例 + 随机对拍，全部实现互检。"""
    solvers = [
        ("回溯+used", permute_used),
        ("回溯+swap", permute_swap),
        ("分治+insert", permute_insert),
    ]
    cases: List[List[int]] = [
        [],                # 边界：空数组 -> [[]]
        [7],               # 边界：单元素
        [1, 2],            # n=2
        [1, 2, 3],         # 题目原例，n=3 全量核对
        [4, 3, 2, 1],      # n=4，仅校验个数与一致性（24 个不全打印）
    ]

    print("=" * 62)
    for nums in cases:
        print(f"nums = {nums}")
        expect_count = 1
        for k in range(2, len(nums) + 1):
            expect_count *= k
        for name, fn in solvers:
            got = fn(nums)
            assert sorted(got) == sorted(brute_force(nums)), (
                f"{name} 与暴力法不一致: nums={nums}"
            )
            assert len(got) == expect_count, (
                f"{name} 排列数错误: nums={nums}, got={len(got)}, "
                f"expect {expect_count}"
            )
            if len(nums) <= 3:  # 小规模直接打印结果便于人工核对
                print(f"  {name:<8} -> {got}")
            else:
                print(f"  {name:<8} -> 共 {len(got)} 个排列（不展开）")
        print("-" * 62)

    # 随机对拍：三实现输出排序后必须完全一致，共 1000 组
    rng = random.Random(20261127)
    for _ in range(1000):
        n = rng.randint(0, 6)
        nums = rng.sample(range(-50, 50), n)  # 无重复数字，符合题设
        outs = [sorted(fn(nums)) for _, fn in solvers]
        assert outs[0] == outs[1] == outs[2], (
            f"随机对拍失败: nums={nums}"
        )
    print("随机对拍 1000 组：三实现全部一致 ✓")

    # 性能粗测：n=9 共 9! = 362880 个排列
    import time
    big = list(range(9))
    for name, fn in solvers:
        t0 = time.perf_counter()
        cnt = len(fn(big))
        dt = (time.perf_counter() - t0) * 1000
        print(f"n=9 全排列：{name:<8} {cnt} 个，耗时 {dt:.1f} ms")

    print("全部测试通过 ✓")


if __name__ == "__main__":
    main()
