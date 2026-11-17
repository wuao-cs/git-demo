"""LeetCode 3. 无重复字符的最长子串 (Longest Substring Without Repeating Characters)

难度：中等 (Medium)
标签：哈希表 / 字符串 / 滑动窗口

题目描述：
    给定一个字符串 s，请你找出其中不含重复字符的最长「子串」的长度。

    示例 1：
        输入：s = "abcabcbb"
        输出：3
        解释：最长无重复字符子串是 "abc"，长度为 3。
    示例 2：
        输入：s = "bbbbb"
        输出：1
        解释：最长无重复字符子串是 "b"，长度为 1。
    示例 3：
        输入：s = "pwwkew"
        输出：3
        解释：最长无重复字符子串是 "wke"，长度为 3。
              注意 "pwke" 是子序列而非子串，子串必须连续。

解题思路（滑动窗口 + 哈希表记录下标）：
    用两个指针 left、right 维护一个窗口 [left, right]，窗口内始终保证「无重复字符」。
    1. right 逐位向右扩张，把字符 s[right] 纳入窗口；
    2. 若 s[right] 已经在窗口内出现过（记上次出现位置为 last_seen），说明窗口失效，
       需要收缩左边界。关键点：left 直接跳到 last_seen + 1，而不是一格格地右移；
       同时必须取 max(left, last_seen + 1)，避免 left 回头（典型反例 "abba"）。
    3. 每次扩张后，窗口长度 right - left + 1 即为一个候选答案，取最大值。

思路对比：
    方案一：暴力枚举所有子串，用 set 判重。
            时间 O(n^3)（n^2 个子串 × O(n) 判重，若边扩边判重可降到 O(n^2)），空间 O(k)。
            直观但会重复扫描大量已知无重复的区间，面试中仅作为「先给出一个可行解」。
    方案二：滑动窗口 + set，左指针逐格右移。
            时间 O(2n)（right 与 left 各最多走 n 步），空间 O(k)。
            已经是最优复杂度，但每条重复字符都会触发多次 left 右移，常数偏大。
    方案三：滑动窗口 + 哈希表记录字符最后下标（本文采用）。
            时间 O(n)（left、right 均单调右移，各至多 n 步，不回溯），空间 O(k)。
            省去了方案二的逐格收缩，代价是需要额外维护「下标」而非「是否存在」。

    k 为字符集大小，本题为 ASCII 时可视为常数 O(1)；
    严格写成 O(min(n, k)) 更准确，因为窗口长度不可能超过字符集大小。

复杂度分析：
    时间复杂度：O(n)，n = len(s)。right 遍历一次，left 只增不减，总移动次数不超过 n。
    空间复杂度：O(k)，k 为字符集大小。哈希表最多存 min(n, k) 个键；
                本题若限定 ASCII/字母表，则退化为 O(1)。

关联题目：LC 76 最小覆盖子串、LC 438 找到字符串中所有字母异位词、LC 340 至多包含 K 个不同字符的最长子串。
"""

from __future__ import annotations


def longest_substring_brute_force(s: str) -> int:
    """方案一：暴力枚举所有子串（仅用于对照，不推荐）。

    外层固定起点，内层向右扩张，一旦出现重复字符就终止当前起点的枚举。
    """
    n: int = len(s)
    best: int = 0
    for start in range(n):
        seen: set[str] = set()
        for end in range(start, n):
            if s[end] in seen:
                break  # 出现重复字符，当前起点下更长的子串必然也重复，直接换起点
            seen.add(s[end])
            best = max(best, end - start + 1)
    return best


def longest_substring_sliding_set(s: str) -> int:
    """方案二：滑动窗口 + set，左指针逐格右移。"""
    n: int = len(s)
    window: set[str] = set()  # 窗口内出现的字符集合
    left: int = 0
    best: int = 0
    for right in range(n):
        # 只要当前字符仍在窗口内，就一格一格地把左边界右移
        while s[right] in window:
            window.remove(s[left])
            left += 1
        window.add(s[right])  # 此时窗口内一定无重复，纳入当前字符
        best = max(best, right - left + 1)
    return best


def length_of_longest_substring(s: str) -> int:
    """方案三（主解法）：滑动窗口 + 哈希表记录字符最后出现的下标。

    Args:
        s: 任意字符串，可为空串。

    Returns:
        不含重复字符的最长连续子串的长度；空串返回 0。

    Time:  O(n)
    Space: O(min(n, k))
    """
    last_index: dict[str, int] = {}  # 字符 -> 它在 s 中最后一次出现的下标
    left: int = 0  # 窗口左边界
    best: int = 0  # 历史最长无重复子串长度

    for right, char in enumerate(s):
        if char in last_index:
            # 字符重复：左边界跳到「上次出现位置 + 1」；
            # 取 max 是为了防止 left 往回跳（如 "abba" 中处理第二个 'a' 时 last_index['a']=0）
            left = max(left, last_index[char] + 1)
        last_index[char] = right  # 更新该字符的最新下标
        best = max(best, right - left + 1)  # 收缩后窗口 [left, right] 必无重复，更新答案
    return best


def main() -> None:
    """标准用例 + 边界用例自测，并校验三种解法结果一致。"""
    # (输入, 期望答案) —— 覆盖空串、单字符、全重复、交错重复、子序列干扰、left 回溯陷阱
    cases: list[tuple[str, int]] = [
        ("", 0),  # 空串：边界
        (" ", 1),  # 单个空格：空格也是一个字符
        ("bbbbb", 1),  # 全重复：答案恒为 1
        ("abcabcbb", 3),  # 经典用例：答案为 "abc"
        ("pwwkew", 3),  # 答案为 "wke"，不是子序列 "pwke"
        ("dvdf", 3),  # 答案为 "vdf"，考察窗口起点不能回到 0
        ("abba", 2),  # left 回溯陷阱：处理后不能把左边界拉回
        ("tmmzuxt", 5),  # 答案为 "mzuxt"，末尾重复字符不影响前面
    ]

    all_passed: bool = True
    for raw, expected in cases:
        got: int = length_of_longest_substring(raw)
        bf: int = longest_substring_brute_force(raw)
        ss: int = longest_substring_sliding_set(raw)
        consistent: bool = got == bf == ss
        ok: bool = got == expected and consistent
        all_passed = all_passed and ok
        print(
            f"{'PASS' if ok else 'FAIL'} | s={raw!r:<12} "
            f"expected={expected} | 跳跃版={got} set版={ss} 暴力={bf}"
        )

    print("-" * 60)
    print("全部用例通过" if all_passed else "存在未通过用例，请检查实现")


if __name__ == "__main__":
    main()
