r"""
LeetCode 215. 数组中的第 K 个最大元素（Kth Largest Element in an Array）
=========================================================================

难度：中等（Medium）
标签：数组 / 分治 / 快速选择（Quickselect） / 堆（优先队列）

题目描述
--------
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
要求不排序整个数组（理想情况下时间复杂度优于 O(n log n)）。

示例 1：
    输入: nums = [3,2,1,5,6,4], k = 2
    输出: 5

示例 2：
    输入: nums = [3,2,3,1,2,4,5,5,6], k = 4
    输出: 4

约束：
    1 <= k <= nums.length <= 10^5
    -10^4 <= nums[i] <= 10^4


解题思路
--------
「第 k 个最大」= 升序排序后下标 n-k 的元素。围绕这一点有三条路：

方案一：小顶堆（本文件主解，find_kth_largest）
    维护一个大小恒为 k 的小顶堆，堆顶就是这 k 个候选里最小的那个，也就是
    当前扫描到的前 k 大元素中的「门槛」。新元素只要大于门槛就替换掉堆顶。
    扫描结束后堆顶即全局第 k 大。
    适用：数据以流式/一次性到达、k 远小于 n 时非常划算；也是面试里最稳的答案。
    复杂度：时间 O(n log k)，空间 O(k)。

方案二：快速选择 Quickselect（find_kth_largest_quickselect）
    借用快排的 partition：随机选枢轴，一趟划分后枢轴落到最终位置 p；
    若 p == 目标下标则直接返回，否则只递归（本实现用循环）进入包含目标的那一侧。
    每趟期望把问题规模减半，故期望 O(n)；最坏（每次枢轴都极值）退化为 O(n²)，
    随机选枢轴可让最坏情况以极大概率不出现。
    复杂度：时间 平均 O(n)、最坏 O(n²)，空间 O(1)（原地，不算拷贝）。

方案三：整体排序（find_kth_largest_sort）
    最朴素，O(n log n) / O(n)，作为正确性基准和「及格线」放进来对比。

思路对比
--------
    ┌────────────┬──────────────┬──────────────┬─────────┬──────────────────────┐
    │ 方案       │ 时间(平均)   │ 时间(最坏)   │ 额外空间│ 适合场景             │
    ├────────────┼──────────────┼──────────────┼─────────┼──────────────────────┤
    │ 小顶堆     │ O(n log k)   │ O(n log k)   │ O(k)    │ 流式数据、k << n     │
    │ 快速选择   │ O(n)         │ O(n²)        │ O(1)    │ 数据可原地打乱、只求值│
    │ 整体排序   │ O(n log n)   │ O(n log n)   │ O(n)    │ 代码最短、需多次查询 │
    └────────────┴──────────────┴──────────────┴─────────┴──────────────────────┘

    - k 很小时堆最优（log k 极小），k 接近 n/2 时快速选择更占优。
    - 注意题目要「第 k 大」而不是「第 k 个不同的值」，重复元素按个数计。
    - 快速选择会打乱原数组，所以入口先拷贝一份，避免副作用。
"""

from __future__ import annotations

import heapq
import random
import time
from typing import List


# ---------------------------------------------------------------------------
# 方案一：大小为 k 的小顶堆 —— 主解
# ---------------------------------------------------------------------------
def find_kth_largest(nums: List[int], k: int) -> int:
    """返回 nums 中第 k 大的元素（重复元素按个数计）。

    用小顶堆把「前 k 大」的候选一直揣在手里，堆顶就是当前第 k 大。
    时间 O(n log k)，空间 O(k)。
    """
    n = len(nums)
    if not 1 <= k <= n:
        raise ValueError(f"k 必须在 [1, {n}] 范围内，收到 k={k}")

    # 先用前 k 个元素建堆：heapify 是 O(k)，比逐个 heappush 的 O(k log k) 更快
    heap: List[int] = nums[:k]
    heapq.heapify(heap)

    for value in nums[k:]:
        # 只有严格大于堆顶（当前门槛）才需要替换，等于时换不换结果都一样
        if value > heap[0]:
            heapq.heapreplace(heap, value)  # 弹出堆顶 + 压入新值，一次下沉完成

    return heap[0]  # 堆里剩下的正是最大的 k 个，堆顶即第 k 大


# ---------------------------------------------------------------------------
# 方案二：快速选择 Quickselect（原地划分，循环代替递归，无递归深度风险）
# ---------------------------------------------------------------------------
def find_kth_largest_quickselect(nums: List[int], k: int, seed: int = 0) -> int:
    """用随机枢轴的快速选择求第 k 大，期望 O(n)，额外空间 O(1)。

    seed 只用于让测试可复现；生产环境可省略（默认跟随系统熵）。
    """
    n = len(nums)
    if not 1 <= k <= n:
        raise ValueError(f"k 必须在 [1, {n}] 范围内，收到 k={k}")

    arr = list(nums)                      # 拷贝，避免打乱调用方数组
    target = n - k                        # 升序视角下的目标下标
    lo, hi = 0, n - 1
    rng = random.Random(seed)

    while lo < hi:
        # 1) 随机选枢轴并换到区间末尾，规避「已排序输入」的最坏情况
        pivot_idx = rng.randint(lo, hi)
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]

        # 2) 一趟分区：小于枢轴的都挪到左侧，store 指向第一个「大于等于」的位置
        store = lo
        for i in range(lo, hi):
            if arr[i] < pivot:
                arr[store], arr[i] = arr[i], arr[store]
                store += 1

        # 3) 枢轴归位到 store，它左边全小、右边全 >=
        arr[store], arr[hi] = arr[hi], arr[store]

        # 4) 只往目标所在的那一侧继续收缩，另一侧直接丢弃
        if store == target:
            return arr[store]
        if store < target:
            lo = store + 1
        else:
            hi = store - 1

    return arr[lo]  # 区间收缩到单点时即为答案


# ---------------------------------------------------------------------------
# 方案三：整体排序 —— 正确性基准 / 及格线
# ---------------------------------------------------------------------------
def find_kth_largest_sort(nums: List[int], k: int) -> int:
    """直接排序取倒数第 k 个，O(n log n) / O(n)，仅作对照。"""
    if not 1 <= k <= len(nums):
        raise ValueError(f"k 超出范围：k={k}, n={len(nums)}")
    return sorted(nums, reverse=True)[k - 1]


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
def _check(name: str, got: int, expected: int) -> bool:
    ok = got == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got}, expected={expected}")
    return ok


def main() -> None:
    print("=" * 72)
    print("LC 215 数组中的第 K 个最大元素 —— 定点用例")
    print("=" * 72)

    # (用例名, nums, k, 期望值) —— 覆盖常规、k=1、k=n、全相同、负数、单元素
    cases = [
        ("示例1 常规", [3, 2, 1, 5, 6, 4], 2, 5),
        ("示例2 含重复", [3, 2, 3, 1, 2, 4, 5, 5, 6], 4, 4),
        ("k=1 即最大值", [7, 1, 9, 3], 1, 9),
        ("k=n 即最小值", [7, 1, 9, 3], 4, 1),
        ("全部相同", [2, 2, 2, 2], 3, 2),
        ("全为负数", [-1, -5, -3, -2], 2, -2),
        ("单元素", [42], 1, 42),
        ("已升序(易触发最坏划分)", list(range(1, 51)), 7, 44),
        ("已降序", list(range(50, 0, -1)), 7, 44),
        ("重复块边界 k=500", [1] * 500 + [9] * 500, 500, 9),
        ("重复块边界 k=501", [1] * 500 + [9] * 500, 501, 1),
    ]

    passed = 0
    total = 0
    for name, nums, k, expected in cases:
        for fn in (find_kth_largest, find_kth_largest_quickselect, find_kth_largest_sort):
            total += 1
            if _check(f"{name} / {fn.__name__}", fn(nums, k), expected):
                passed += 1
    print(f"\n定点用例：{passed}/{total} 通过")

    # ---- 随机对拍：三种实现互相比对，防止某一路写错 ----
    print("\n" + "-" * 72)
    print("随机对拍（1000 组，三实现结果须一致）")
    print("-" * 72)
    rng = random.Random(20261126)
    mismatch = 0
    for _ in range(1000):
        n = rng.randint(1, 60)
        nums = [rng.randint(-20, 20) for _ in range(n)]
        k = rng.randint(1, n)
        base = find_kth_largest_sort(nums, k)
        if find_kth_largest(nums, k) != base or find_kth_largest_quickselect(nums, k) != base:
            mismatch += 1
            print(f"  [FAIL] nums={nums}, k={k}, 期望 {base}")
    print(f"  对拍结果：1000 组，不一致 {mismatch} 组")

    # ---- 异常入参 ----
    print("\n" + "-" * 72)
    print("边界入参校验")
    print("-" * 72)
    for bad_k in (0, -1, 6):
        try:
            find_kth_largest([1, 2, 3, 4, 5], bad_k)
            print(f"  [FAIL] k={bad_k} 未抛出异常")
        except ValueError as exc:
            print(f"  [PASS] k={bad_k} 正确抛错：{exc}")

    # ---- 性能粗测：n = 200_000 ----
    print("\n" + "-" * 72)
    print("性能粗测 n=200000（k=1000）")
    print("-" * 72)
    big = [rng.randint(-10**4, 10**4) for _ in range(200_000)]
    for label, fn in (
        ("小顶堆 O(n log k)", lambda: find_kth_largest(big, 1000)),
        ("快速选择 期望 O(n)", lambda: find_kth_largest_quickselect(big, 1000)),
        ("整体排序 O(n log n)", lambda: find_kth_largest_sort(big, 1000)),
    ):
        start = time.perf_counter()
        result = fn()
        cost_ms = (time.perf_counter() - start) * 1000
        print(f"  {label:<22} 结果={result:<7} 耗时 {cost_ms:7.2f} ms")


if __name__ == "__main__":
    main()
