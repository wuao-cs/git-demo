r"""LeetCode 199. 二叉树的右视图 (Binary Tree Right Side View)

难度: Medium
标签: 树 / 深度优先搜索 / 广度优先搜索 / 二叉树

题目描述
--------
给定一棵二叉树的根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，
返回从右侧所能看到的节点值。也就是说，返回每一层最右边的那个节点。

示例:
    输入: [1, 2, 3, null, 5, null, 4]
        1
       / \
      2   3
       \   \
        5   4
    输出: [1, 3, 4]

约束:
    - 树中节点数范围为 [0, 100]
    - -100 <= Node.val <= 100

解题思路
--------
本题的题眼是: "右视图" 并不是"一直往右走"，而是"每一层的最右节点"。
例如上面示例里第二层的 5 在左子树的右侧，它依然是该层最右节点 —— 所以必须按层理解。

方案一: BFS 层序遍历
    用队列逐层扫描，每层的最后一个出队节点就是能看到的那一个。
    这是最贴合题意的写法，代码几乎不需要思考。

方案二: DFS 先访问右子树（迭代栈实现）
    按「根 → 右 → 左」的顺序做前序遍历，并记录当前深度。
    某个深度第一次被访问到的节点，一定是这一层最靠右的节点
    （因为右子树永远先于左子树被访问）。
    判断方式很简洁: 当 depth == len(result) 时，说明这是新的一层。

方案三: DFS 递归（同方案二思路，写法更短）
    递归深度等于树高，对深度极端不平衡的树有 RecursionError 风险，
    仅作思路对照，实际工程/竞赛中优先用方案一或方案二。

思路对比
--------
              | 遍历顺序        | 记录时机                | 空间(递归/栈)
方案一 BFS    | 逐层从左到右    | 每层最后一个出队        | O(树的最大宽度)
方案二 DFS栈  | 根-右-左        | 每个深度首次入栈        | O(树高) 最好, 最坏 O(n)
方案三 DFS递归| 根-右-左        | 每个深度首次进入        | O(树高) 调用栈，受递归上限约束

三者在最坏情况下都是 O(n) 时间、O(n) 空间，但 DFS 的空间只与"树高"相关，
在层数少、单层很宽的树上（例如完全二叉树）明显比 BFS 省内存。

复杂度分析
----------
时间复杂度: O(n)，每个节点恰好被访问一次（三种实现相同）。
空间复杂度:
    - BFS: O(w)，w 为树的最大宽度，完全二叉树最后一层约 n/2，即最坏 O(n)。
    - DFS: O(h)，h 为树高，平衡树为 O(log n)，退化成链表时为 O(n)。
"""

from __future__ import annotations

from collections import deque
from typing import Optional


class TreeNode:
    """二叉树节点。"""

    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ) -> None:
        self.val: int = val
        self.left: Optional[TreeNode] = left
        self.right: Optional[TreeNode] = right

    def __repr__(self) -> str:  # 便于调试时直接打印节点
        return f"TreeNode({self.val})"


# --------------------------------------------------------------------------- #
# 辅助函数: 用 "层序数组 + null" 构造二叉树 / 反向导出，用于测试用例
# --------------------------------------------------------------------------- #
def build_tree(level_order: list[Optional[int]]) -> Optional[TreeNode]:
    """按层序数组构造二叉树，None 表示空节点（LeetCode 的输入格式）。

    只把非 None 的元素当作真实节点依次挂到父节点上，
    因此 ["1", None, "2", None, "3"] 会得到一条纯右斜链。
    """
    if not level_order or level_order[0] is None:
        return None

    root = TreeNode(level_order[0])
    queue: deque[TreeNode] = deque([root])
    index = 1  # 下一个待挂载的值在数组中的位置

    while queue and index < len(level_order):
        parent = queue.popleft()

        # 先挂左孩子
        if index < len(level_order) and level_order[index] is not None:
            parent.left = TreeNode(level_order[index])
            queue.append(parent.left)
        index += 1

        # 再挂右孩子
        if index < len(level_order) and level_order[index] is not None:
            parent.right = TreeNode(level_order[index])
            queue.append(parent.right)
        index += 1

    return root


def tree_to_level_order(root: Optional[TreeNode]) -> list[Optional[int]]:
    """把二叉树导出为层序数组（去掉尾部多余的 None），仅用于打印验证。"""
    if root is None:
        return []

    result: list[Optional[int]] = []
    queue: deque[Optional[TreeNode]] = deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            result.append(None)
            continue
        result.append(node.val)
        queue.append(node.left)
        queue.append(node.right)

    while result and result[-1] is None:  # 去掉尾部占位的 None
        result.pop()
    return result


# --------------------------------------------------------------------------- #
# 方案一: BFS 层序遍历 —— 每层的最后一个节点就是右视图
# --------------------------------------------------------------------------- #
def right_side_view_bfs(root: Optional[TreeNode]) -> list[int]:
    """BFS 逐层扫描，收集每层最右侧节点。时间 O(n)，空间 O(最大宽度)。"""
    if root is None:
        return []

    result: list[int] = []
    queue: deque[TreeNode] = deque([root])

    while queue:
        level_size = len(queue)  # 当前层的节点个数，先固定下来
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:  # 该层最后一个出队的节点
                result.append(node.val)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)

    return result


# --------------------------------------------------------------------------- #
# 方案二: DFS「根-右-左」+ 显式栈 —— 每个深度首次访问到的即最右节点
# --------------------------------------------------------------------------- #
def right_side_view_dfs(root: Optional[TreeNode]) -> list[int]:
    """右优先前序遍历（显式栈）。时间 O(n)，空间 O(树高)。"""
    if root is None:
        return []

    result: list[int] = []
    stack: list[tuple[TreeNode, int]] = [(root, 0)]

    while stack:
        node, depth = stack.pop()

        # 该深度第一次被访问 → 一定是这一层最右侧的节点
        if depth == len(result):
            result.append(node.val)

        # 栈是后进先出: 先压左再压右，保证右子树先出栈
        if node.left is not None:
            stack.append((node.left, depth + 1))
        if node.right is not None:
            stack.append((node.right, depth + 1))

    return result


# --------------------------------------------------------------------------- #
# 方案三: DFS 递归 —— 同方案二思路，写法更紧凑（深度大时有递归风险）
# --------------------------------------------------------------------------- #
def right_side_view_dfs_recursive(root: Optional[TreeNode]) -> list[int]:
    """右优先递归前序遍历。时间 O(n)，空间 O(树高) 调用栈。"""
    result: list[int] = []

    def dfs(node: Optional[TreeNode], depth: int) -> None:
        if node is None:
            return
        if depth == len(result):  # 该深度首次到达
            result.append(node.val)
        dfs(node.right, depth + 1)  # 先右后左，顺序不能反
        dfs(node.left, depth + 1)

    dfs(root, 0)
    return result


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def run_tests() -> None:
    """5 组以上定点用例，含空树、单节点、单链等边界情况。"""
    cases: list[tuple[list[Optional[int]], list[int], str]] = [
        # (输入层序数组, 期望输出, 说明)
        ([1, 2, 3, None, 5, None, 4], [1, 3, 4], "题目示例：5 在左子树右侧"),
        ([], [], "边界：空树"),
        ([1], [1], "边界：单节点"),
        ([1, 2], [1, 2], "边界：只有左孩子"),
        ([1, None, 2, None, 3], [1, 2, 3], "退化：纯右斜链"),
        ([1, 2, None, 3, None, 4], [1, 2, 3, 4], "退化：纯左斜链"),
        ([1, 2, 3, 4, None, None, 5], [1, 3, 5], "不对称：右侧提前结束"),
        ([1, -2, -3, None, -5], [1, -3, -5], "含负数值"),
    ]

    implementations = [
        ("BFS 层序", right_side_view_bfs),
        ("DFS 栈", right_side_view_dfs),
        ("DFS 递归", right_side_view_dfs_recursive),
    ]

    passed = 0
    total = 0

    for level_order, expected, note in cases:
        root = build_tree(level_order)
        print(f"\n输入: {level_order}   ({note})")
        print(f"  期望: {expected}")
        for name, func in implementations:
            got = func(root)
            total += 1
            ok = got == expected
            passed += ok
            flag = "OK  " if ok else "FAIL"
            print(f"  [{flag}] {name:<8} -> {got}")

    print(f"\n===== 定点用例: {passed}/{total} 通过 =====")
    assert passed == total, "存在未通过的定点用例"


def cross_check(rounds: int = 2000, max_nodes: int = 40) -> None:
    """随机树对拍：三种实现结果必须完全一致。"""
    import random

    random.seed(20261125)  # 固定种子，保证可复现

    for _ in range(rounds):
        size = random.randint(0, max_nodes)
        level_order: list[Optional[int]] = []
        for _ in range(size):
            # 约 1/4 的位置留空，制造各种不规则形状
            level_order.append(None if random.random() < 0.25 else random.randint(-100, 100))
        if level_order and level_order[0] is None:
            level_order[0] = random.randint(-100, 100)  # 根节点不能为空

        root = build_tree(level_order)
        base = right_side_view_bfs(root)
        assert right_side_view_dfs(root) == base, f"DFS 栈与 BFS 不一致: {level_order}"
        assert right_side_view_dfs_recursive(root) == base, f"DFS 递归与 BFS 不一致: {level_order}"

        # 额外校验: 右视图长度必须等于树高
        assert len(base) == tree_height(root), f"层数与树高不符: {level_order}"

    print(f"随机对拍 {rounds} 轮通过（三种实现结果一致，且层数 == 树高）")


def tree_height(root: Optional[TreeNode]) -> int:
    """求树高（层数），用迭代 BFS 实现以避免深树递归问题。"""
    if root is None:
        return 0
    depth = 0
    queue: deque[TreeNode] = deque([root])
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
    return depth


if __name__ == "__main__":
    run_tests()
    cross_check()

    # 演示: 打印一棵示例树的层序还原，确认建树函数本身没写错
    demo = [1, 2, 3, None, 5, None, 4]
    demo_root = build_tree(demo)
    print(f"\n建树自检: {demo} -> {tree_to_level_order(demo_root)}")
    print(f"树高: {tree_height(demo_root)}，右视图: {right_side_view_bfs(demo_root)}")
