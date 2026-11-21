"""
LeetCode 128. Longest Consecutive Sequence（最长连续序列）

难度：Medium
标签：数组 / 哈希表 / 并查集（本题用哈希集合）/ 排序（对照解）

题目描述：
    给定一个未排序的整数数组 nums，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    要求设计并实现时间复杂度为 O(n) 的算法解决此问题。

    示例 1：
        输入：nums = [100, 4, 200, 1, 3, 2]
        输出：4
        解释：最长连续序列是 [1, 2, 3, 4]，长度为 4。
    示例 2：
        输入：nums = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
        输出：9
        解释：最长连续序列是 [0, 1, 2, 3, 4, 5, 6, 7, 8]，长度为 9。

解题思路（哈希集合，只从「段起点」向右扩展）：
    1. 把数组全部丢进哈希集合 num_set，实现 O(1) 的「某个数是否存在」判断，同时天然完成去重。
    2. 遍历集合中的每个数 x，先判断 x - 1 是否也在集合里：
       - 如果 x - 1 存在，说明 x 只是某个连续段内部的元素，从它出发一定会重复扫描别的起点已经扫过的部分，直接跳过；
       - 只有 x - 1 不存在时，x 才是某个连续段的最小值（起点），此时才从 x 开始不断检查 x + 1、x + 2 ...
         直到断开为止，得到该段的长度并更新答案。
    3. 因为每个数最多被「作为起点扩展一次」且最多被「作为段内元素命中一次」，
       总扩展次数是 O(n)，整体时间复杂度 O(n)。

    关键点：为什么跳过 x - 1 存在的元素不会漏解？
        任何连续段 [a, a+1, ..., b] 都只会被它的最小值 a 触发一次完整扩展，
        段内其他元素 b' 满足 b' - 1 存在，因此一定被 continue 跳过。
        所以每段恰好统计一次，不会漏、也不会重复。

思路对比：
    ① 排序后一次扫描：先排序，再统计相邻差为 1 的连续长度。
       实现最直观，但排序需要 O(n log n)，不满足题目 O(n) 的要求，且原数组会被修改（需要拷贝）。
       本题仍然保留该写法 longest_consecutive_sorting 作为对照与对拍基准。
    ② 哈希集合 + 只从起点扩展（主解）：时间 O(n)、空间 O(n)，是本题的标准解。
    ③ 暴力：对每个数都往两边找，最坏 O(n^3)，仅用于校验小样例，不实际使用。
    ④ 并查集：把 x 与 x + 1 合并，最后取最大连通块大小，同样是 O(n α(n))，但代码量明显更大，
       本题用哈希集合已经足够，故不实现。

复杂度分析（主解 longest_consecutive）：
    时间复杂度：O(n)。建集合 O(n)；while 循环总步数不超过 n（每个数至多被扩展一次），
               因此外层次数与内层扩展步数合计 O(n)。
    空间复杂度：O(n)。哈希集合最多存放 n 个不同元素。
    （注意：这里说的 n 是「不同元素的个数」，与输入长度同阶。）

复杂度分析（对照解 longest_consecutive_sorting）：
    时间复杂度：O(n log n)，瓶颈在排序。
    空间复杂度：O(n)，需要拷贝一份数组避免修改入参（Python 的 sorted 本身返回新列表）。
"""

from typing import List, Set


def longest_consecutive(nums: List[int]) -> int:
    """哈希集合解法：只从每个连续段的起点向右扩展，返回最长连续序列长度。

    Args:
        nums: 未排序的整数数组，可能为空、可能含重复元素、可能含负数。

    Returns:
        最长连续整数序列的长度；空数组返回 0。
    """
    if not nums:
        return 0

    num_set: Set[int] = set(nums)  # 去重 + O(1) 存在性查询
    best: int = 0

    for num in num_set:
        # 只有「没有前驱」的数才是连续段的起点，非起点直接跳过以避免重复扫描
        if num - 1 in num_set:
            continue

        current: int = num
        length: int = 1
        # 从起点一路向右数，直到数字断开
        while current + 1 in num_set:
            current += 1
            length += 1

        if length > best:
            best = length

    return best


def longest_consecutive_sorting(nums: List[int]) -> int:
    """对照解：排序后单次扫描，时间复杂度 O(n log n)。

    用于验证主解正确性（随机对拍），也便于直观理解「连续」的含义。
    """
    if not nums:
        return 0

    ordered: List[int] = sorted(nums)  # sorted 返回新列表，不修改入参
    best: int = 1
    length: int = 1

    for i in range(1, len(ordered)):
        if ordered[i] == ordered[i - 1]:
            continue  # 重复元素不影响长度，跳过
        if ordered[i] == ordered[i - 1] + 1:
            length += 1
        else:
            length = 1  # 断开，重新开始计数
        if length > best:
            best = length

    return best


def longest_consecutive_brute(nums: List[int]) -> int:
    """暴力解：对每个数都往两边扩展，最坏 O(n^3)，仅用于小样例校验。"""
    num_set: Set[int] = set(nums)
    best: int = 0

    for num in num_set:
        length: int = 1
        # 向左找
        left: int = num - 1
        while left in num_set:
            length += 1
            left -= 1
        # 向右找
        right: int = num + 1
        while right in num_set:
            length += 1
            right += 1
        if length > best:
            best = length

    return best


if __name__ == "__main__":
    # (输入, 期望输出, 说明)
    cases: List[tuple] = [
        ([], 0, "空数组：边界，应返回 0"),
        ([1], 1, "单元素：连续段长度 1"),
        ([100, 4, 200, 1, 3, 2], 4, "题目示例 1：连续段 [1,2,3,4]"),
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9, "题目示例 2：含重复 0，连续段 0..8"),
        ([1, 2, 0, 1], 3, "重复元素 + 乱序：[0,1,1,2] -> 3"),
        ([-1, -2, -3, 5, 4], 3, "含负数且跨零：[4,5] 与 [-3,-2,-1] 各长 3"),
        ([7, 7, 7], 1, "全相同元素：去重后长度 1"),
        ([9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6], 7, "长段 [-1,0,1] 与 3..9 交错"),
    ]

    all_passed: bool = True
    for idx, (nums, expected, note) in enumerate(cases, start=1):
        got: int = longest_consecutive(nums)
        ok: bool = got == expected
        all_passed = all_passed and ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] case {idx}: "
            f"nums={nums} -> {got} (期望 {expected})  # {note}"
        )

    # 随机对拍：主解 vs 排序解 vs 暴力解（小规模）
    import random

    random.seed(20261121)
    mismatch: int = 0
    for _ in range(3000):
        size: int = random.randint(0, 12)
        arr: List[int] = [random.randint(-10, 10) for _ in range(size)]
        a: int = longest_consecutive(arr)
        b: int = longest_consecutive_sorting(arr)
        c: int = longest_consecutive_brute(arr)
        if not (a == b == c):
            mismatch += 1
            print(f"[FAIL] 对拍不一致: nums={arr}, hash={a}, sort={b}, brute={c}")
    print(f"[{'PASS' if mismatch == 0 else 'FAIL'}] 随机对拍 3000 组（哈希 / 排序 / 暴力三解一致）")

    # 性能粗测：20 万随机整数，验证 O(n) 表现在线性可接受
    import time

    big: List[int] = [random.randint(-500_000, 500_000) for _ in range(200_000)]
    start: float = time.perf_counter()
    result: int = longest_consecutive(big)
    elapsed_ms: float = (time.perf_counter() - start) * 1000
    print(f"[INFO] n=200000 随机数据：最长长度 {result}，耗时 {elapsed_ms:.1f} ms")

    print("\n全部用例通过" if all_passed and mismatch == 0 else "\n存在失败用例，请检查")
