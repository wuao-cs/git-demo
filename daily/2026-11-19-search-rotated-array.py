"""
LeetCode 33. 搜索旋转排序数组 (Search in Rotated Sorted Array)

难度：中等
标签：数组、二分查找

题目描述
--------
整数数组 nums 按升序排列，数组中的值互不相同。
在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了旋转，
使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]。
例如 [0,1,2,4,5,6,7] 在下标 3 处旋转后变为 [4,5,6,7,0,1,2]。

给你旋转后的数组 nums 和一个整数 target，如果 nums 中存在这个目标值 target，
则返回它的下标，否则返回 -1。要求时间复杂度为 O(log n)。

示例
----
输入：nums = [4,5,6,7,0,1,2], target = 0   -> 输出：4
输入：nums = [4,5,6,7,0,1,2], target = 3   -> 输出：-1
输入：nums = [1],               target = 0   -> 输出：-1

解题思路（方法一：一次二分，主解法）
------------------------------------
旋转数组有一个关键性质：**任意切一刀（取 mid）后，左右两段中至少有一段是完全有序的**。

于是每轮循环：
1. 若 nums[mid] == target，直接返回 mid。
2. 判断哪一段有序：
   - 若 nums[left] <= nums[mid]，说明 [left, mid] 升序（左段有序）；
   - 否则右段 [mid, right] 必然升序。
3. 只在**有序的那一段**上用普通二分判断 target 是否落在其值域内：
   - 落进去就把区间收缩到该段，否则收缩到另一半。

因为每轮都把搜索区间折半，且落点判断是 O(1) 的，整体仍是标准二分。

易错点：
- 判断左段有序必须写 `nums[left] <= nums[mid]`（含等号），这样 left == mid 的退化情况
  （区间只剩 1~2 个元素）会走"左段有序"分支，逻辑依然自洽。
- 值域判断要用严格不等号配合开区间：左段有序时 target 落在 [nums[left], nums[mid])，
  右段有序时落在 (nums[mid], nums[right]]，避免把 mid 本身重复纳入。

思路对比
--------
方法一（一次二分，判断有序半边）
    时间 O(log n)，空间 O(1)。一轮循环内同时完成"定位有序段"和"排除一半"，常数最小，
    是本题的标准解法。

方法二（两次二分：先找旋转点，再在有序段二分）
    时间 O(log n)，空间 O(1)。先把最小值下标（即旋转点）二分出来（同 LC 153），
    再判断 target 落在哪个有序区间里做第二次二分。思路更"模块化"、可复用于
    "在旋转数组中找最小值"一类题，但要走两趟二分，常数约大一倍，代码也更长。

方法三（线性扫描）
    时间 O(n)，空间 O(1)。仅用于本地随机交叉验证的对照实现，不满足题目 O(log n) 要求。

复杂度分析（方法一）
--------------------
时间：O(log n)。每次迭代区间长度减半，最多约 log2(n) 轮，每轮 O(1)。
空间：O(1)。只用 left / right / mid 三个下标变量，迭代写法无递归栈开销。
"""

from __future__ import annotations

import random
import time


def search(nums: list[int], target: int) -> int:
    """在旋转后的升序数组中查找 target，返回下标，不存在返回 -1。

    一次二分，每轮判断 mid 左右哪一段有序，只在有序段上做值域判断。

    Args:
        nums: 由互不相同的升序整数旋转而成的数组。
        target: 待查找的目标值。

    Returns:
        target 在 nums 中的下标；若不存在则返回 -1。
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2  # 这样写可避免 (left + right) 溢出（Python 无此问题，延续 C/Java 习惯）

        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:
            # 左半段 [left, mid] 有序：target 是否落在该段的值域内
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # 在有序的左段里继续找
            else:
                left = mid + 1   # 左段值域装不下 target，只能去右段
        else:
            # 右半段 [mid, right] 有序：target 是否落在该段的值域内
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # 在有序的右段里继续找
            else:
                right = mid - 1  # 右段值域装不下 target，只能去左段

    return -1


def find_min_index(nums: list[int]) -> int:
    """返回旋转升序数组中最小元素的下标（即旋转点），同 LeetCode 153。

    二分查找：与右端点比较，若 nums[mid] > nums[right] 说明最小值在右半段，
    否则（含相等情形，本题元素互不相同）最小值在左半段或就是 mid。

    Args:
        nums: 旋转后的升序数组，长度至少为 1。

    Returns:
        最小元素所在下标。
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1  # 最小值一定在 mid 右侧
        else:
            right = mid      # mid 可能就是最小值，不能 mid - 1

    return left


def search_two_pass(nums: list[int], target: int) -> int:
    """方法二：两次二分。先定位旋转点，再在对应的有序段里二分。

    用于对照验证主解法，并展示"分段处理旋转数组"这一通用套路。

    Args:
        nums: 由互不相同的升序整数旋转而成的数组。
        target: 待查找的目标值。

    Returns:
        target 在 nums 中的下标；若不存在则返回 -1。
    """
    if not nums:
        return -1

    pivot = find_min_index(nums)  # 旋转点：右段的起点

    def binary_search(lo: int, hi: int) -> int:
        """在闭区间 [lo, hi]（本身有序）上做普通二分。"""
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    # 右段 [pivot, n-1] 必然升序，且其首元素是整个数组的最小值
    if nums[pivot] <= target <= nums[-1]:
        return binary_search(pivot, len(nums) - 1)
    # 否则只可能在左段 [0, pivot-1]
    return binary_search(0, pivot - 1)


def search_linear(nums: list[int], target: int) -> int:
    """方法三：O(n) 线性扫描，仅作随机交叉验证的对照基准。"""
    for index, value in enumerate(nums):
        if value == target:
            return index
    return -1


# -----------------------------------------------------------------------------
# 测试
# -----------------------------------------------------------------------------

# (nums, target, 期望下标, 说明)
TEST_CASES: list[tuple[list[int], int, int, str]] = [
    ([4, 5, 6, 7, 0, 1, 2], 0, 4, "典型样例：目标落在右段起点"),
    ([4, 5, 6, 7, 0, 1, 2], 3, -1, "典型样例：目标不存在"),
    ([4, 5, 6, 7, 0, 1, 2], 6, 2, "目标在左段中部"),
    ([1], 0, -1, "边界：单元素且不命中"),
    ([1], 1, 0, "边界：单元素且命中"),
    ([3, 1], 1, 1, "边界：两元素，旋转在尾部"),
    ([1, 3], 3, 1, "边界：两元素，实际未旋转"),
    ([5, 1, 3], 5, 0, "边界：目标恰为数组首元素（左段退化为单点）"),
    ([1, 2, 3, 4, 5], 5, 4, "边界：完全有序、目标为最大值"),
    ([2, 3, 4, 5, 1], 1, 4, "边界：旋转点在最末尾"),
]


def _check(actual: int, expected: int) -> str:
    """返回单条用例的通过标记。"""
    return "PASS" if actual == expected else f"FAIL(期望 {expected})"


def run_fixed_cases() -> bool:
    """跑定点用例，主解法与方法二同时校验。"""
    print("=" * 72)
    print("定点用例")
    print("=" * 72)

    all_passed = True
    for index, (nums, target, expected, note) in enumerate(TEST_CASES, start=1):
        got_main = search(nums, target)
        got_alt = search_two_pass(nums, target)
        passed = got_main == expected and got_alt == expected
        all_passed = all_passed and passed

        show_nums = str(nums) if len(str(nums)) <= 32 else f"{str(nums)[:29]}...]"
        print(
            f"[{index:>2}] {'OK ' if passed else 'NG '} nums={show_nums:<32} "
            f"target={target:<4} -> 主解 {got_main:>3} | 两次二分 {got_alt:>3} | {note}"
        )

    print(f"\n定点用例结果：{'全部通过' if all_passed else '存在失败'}"
          f"（{len(TEST_CASES)} 条）")
    return all_passed


def run_random_cross_check(rounds: int = 2000, seed: int = 20261119) -> bool:
    """随机生成旋转数组，与线性扫描逐一对拍，覆盖边界更充分。"""
    rng = random.Random(seed)
    mismatch = 0

    for _ in range(rounds):
        size = rng.randint(1, 40)
        base = sorted(rng.sample(range(-50, 50), size))  # 互不相同的升序数组
        k = rng.randint(0, size - 1)
        rotated = base[k:] + base[:k]                     # 在下标 k 处旋转
        target = rng.choice(base) if rng.random() < 0.6 else rng.randint(-60, 60)

        expected = search_linear(rotated, target)
        got_main = search(rotated, target)
        got_alt = search_two_pass(rotated, target)

        if got_main != expected or got_alt != expected:
            mismatch += 1
            if mismatch <= 3:  # 只打印前几条，避免刷屏
                print(f"  不一致：nums={rotated} target={target} "
                      f"期望 {expected}，主解 {got_main}，两次二分 {got_alt}")

    print("=" * 72)
    print(f"随机交叉验证：{rounds} 组，与 O(n) 暴力法对照，不一致 {mismatch} 组")
    print("=" * 72)
    return mismatch == 0


def run_performance_probe() -> None:
    """粗略性能探测：大数组上连续查询，确认仍是对数级。"""
    size = 200_000
    rotated = list(range(size // 2, size)) + list(range(size // 2))
    queries = [random.randrange(size) for _ in range(20_000)]

    start = time.perf_counter()
    for value in queries:
        search(rotated, value)
    elapsed = time.perf_counter() - start

    print(f"性能探测：n={size:,} 的旋转数组上执行 {len(queries):,} 次查询，"
          f"耗时 {elapsed * 1000:.1f} ms（{elapsed / len(queries) * 1e6:.2f} µs/次）")


if __name__ == "__main__":
    fixed_ok = run_fixed_cases()
    random_ok = run_random_cross_check()
    print()
    run_performance_probe()

    print()
    if fixed_ok and random_ok:
        print("=> 全部用例通过 ✅")
    else:
        print("=> 有用例未通过 ❌")
        raise SystemExit(1)
