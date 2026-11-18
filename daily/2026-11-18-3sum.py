"""
LeetCode 15. 三数之和 (3Sum)

难度：Medium
标签：数组 / 双指针 / 排序

题目描述
--------
给你一个整数数组 nums，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足
i != j、i != k 且 j != k，同时还满足 nums[i] + nums[j] + nums[k] == 0。
请你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

示例 1：
    输入：nums = [-1, 0, 1, 2, -1, -4]
    输出：[[-1, -1, 2], [-1, 0, 1]]

示例 2：
    输入：nums = [0, 1, 1]
    输出：[]

示例 3：
    输入：nums = [0, 0, 0]
    输出：[[0, 0, 0]]

提示：
    3 <= nums.length <= 3000
    -10^5 <= nums[i] <= 10^5

解题思路
--------
核心是把「三数之和」降维成「两数之和」：先排序，再固定第一个数 nums[i]，
剩下两个数用左右双指针在有序区间里逼近 -nums[i]。

1. 排序：让数组有序，双指针才有「和偏小就左指针右移、偏大就右指针左移」的单调依据，
   同时相同的值会聚在一起，去重变得便宜。
2. 枚举 i 作为最小的那个数（i 只需走到 n-3）：
   - 剪枝 A：nums[i] > 0 时直接 break。数组已升序，后面所有数都 >= nums[i] > 0，
     三数之和不可能为 0。
   - 剪枝 B：i > 0 且 nums[i] == nums[i-1] 时 continue，跳过重复的固定值，
     否则会枚举出重复三元组。
3. 双指针：left = i + 1，right = n - 1，target = -nums[i]。
   - 和 < target：说明偏小，left++（更大的数在右边）。
   - 和 > target：说明偏大，right--。
   - 和 == target：记录答案，然后 left++、right--，并各自跳过相邻重复值，
     保证同一个三元组只被记录一次。

去重策略是本题的易错点：三处去重（固定值 i、左指针、右指针）必须都做，
否则要么漏解、要么重复。用排序 + 双指针后，去重是「跳过相邻相同值」的自然动作。

思路对比
--------
| 方案                | 时间      | 空间        | 说明                                        |
|---------------------|-----------|-------------|---------------------------------------------|
| 三重循环暴力        | O(n^3)    | O(1)        | 3000 规模下 2.7e10 次运算，必然超时          |
| 哈希表（固定 i + 两数之和）| O(n^2) | O(n)     | 思路直白，但额外哈希开销 + 去重逻辑更绕      |
| 排序 + 双指针（本解）| O(n^2)   | O(1) 额外   | 常数小、无需额外结构，去重最自然，面试首选    |

复杂度分析
----------
时间复杂度：O(n^2)。
    排序 O(n log n)；外层枚举 i 共 O(n) 次，每次双指针只单向移动，合计 O(n)，
    因此主体是 O(n^2)，排序被吸收。
空间复杂度：O(1) 额外空间（不计排序所需的 O(log n) 栈空间与存放答案的数组）。
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple


def three_sum(nums: Sequence[int]) -> List[List[int]]:
    """返回所有和为 0 且不重复的三元组（升序排列）。

    Args:
        nums: 整数序列，长度 >= 3。

    Returns:
        三元组列表，每个三元组内部升序；不同三元组之间按字典序有序。

    Examples:
        >>> three_sum([-1, 0, 1, 2, -1, -4])
        [[-1, -1, 2], [-1, 0, 1]]
        >>> three_sum([0, 1, 1])
        []
    """
    n = len(nums)
    if n < 3:
        return []  # 不足三个数，不可能构成三元组

    ordered: List[int] = sorted(nums)  # 排序：双指针单调性的前提，也让重复值相邻
    result: List[List[int]] = []

    # 枚举第一个数（最小的数），最多只需枚举到倒数第三个位置
    for i in range(n - 2):
        first = ordered[i]

        # 剪枝 A：最小的数都大于 0，后面只会更大，三数之和必为正
        if first > 0:
            break

        # 去重 1：跳过重复的固定值，避免重复枚举同一组解
        if i > 0 and first == ordered[i - 1]:
            continue

        left, right = i + 1, n - 1
        target = -first  # 剩下两个数需要凑出的和

        while left < right:
            current = ordered[left] + ordered[right]

            if current < target:
                left += 1  # 和偏小，左指针右移取更大的数
            elif current > target:
                right -= 1  # 和偏大，右指针左移取更小的数
            else:
                result.append([first, ordered[left], ordered[right]])

                # 找到一组解后两指针同时内收，并跳过相邻重复值（去重 2、3）
                left += 1
                right -= 1
                while left < right and ordered[left] == ordered[left - 1]:
                    left += 1
                while left < right and ordered[right] == ordered[right + 1]:
                    right -= 1

    return result


def three_sum_brute_force(nums: Sequence[int]) -> List[List[int]]:
    """暴力对照实现：枚举所有组合再用集合去重，仅用于交叉验证，O(n^3)。"""
    uniques = {tuple(sorted(triple)) for triple in combinations(nums, 3) if sum(triple) == 0}
    return [list(triple) for triple in sorted(uniques)]


def _normalize(triplets: List[List[int]]) -> List[Tuple[int, ...]]:
    """把结果规范成可比较的形式：三元组内部排序，整体再排序，忽略返回顺序差异。"""
    return sorted(tuple(sorted(triple)) for triple in triplets)


def _run_tests() -> int:
    """执行全部用例，返回失败个数（0 表示全通过）。"""
    cases: List[Tuple[str, List[int], List[List[int]]]] = [
        ("经典样例（含重复 -1）", [-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
        ("空数组（边界：无元素）", [], []),
        ("元素不足三个（边界：长度为 2）", [0, 0], []),
        ("恰好三个数且无解", [1, 2, 3], []),
        ("全零（边界：答案为自身重复）", [0, 0, 0, 0, 0], [[0, 0, 0]]),
        ("存在重复答案需去重", [-2, 0, 0, 2, 2], [[-2, 0, 2]]),
        (
            "长样例：多组解 + 多重重复",
            [-4, -2, -2, -2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 6, 6],
            [[-4, -2, 6], [-4, 0, 4], [-4, 1, 3], [-4, 2, 2], [-2, -2, 4], [-2, 0, 2]],
        ),
        ("全正数（边界：触发剪枝 A）", [1, 2, 3, 4, 5], []),
    ]

    failures = 0
    for name, nums, expected in cases:
        actual = three_sum(nums)
        ok = _normalize(actual) == _normalize(expected)
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        input    = {nums}")
        print(f"        expected = {_normalize(expected)}")
        print(f"        actual   = {_normalize(actual)}")

    # 随机交叉验证：与 O(n^3) 暴力解比对，确认无漏解、无重复
    import random

    random.seed(20261118)
    mismatched = 0
    for _ in range(20):
        sample = [random.randint(-20, 20) for _ in range(60)]
        if _normalize(three_sum(sample)) != _normalize(three_sum_brute_force(sample)):
            mismatched += 1
    ok = mismatched == 0
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] 随机交叉验证（20 组，n=60，对照暴力解，不一致 {mismatched} 组）")

    # 性能观察：n = 3000 的上限规模，验证 O(n^2) 实际可跑
    import time

    big = [random.randint(-100_000, 100_000) for _ in range(3000)]
    start = time.perf_counter()
    big_result = three_sum(big)
    elapsed = time.perf_counter() - start
    print(f"[INFO] 规模 n=3000 跑完，解得 {len(big_result)} 组，耗时 {elapsed * 1000:.1f} ms")

    print(f"\n测试结论：{'全部通过' if failures == 0 else f'{failures} 个用例失败'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_tests() else 0)
