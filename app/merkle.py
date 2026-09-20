"""Merkle 包含证明（可选择披露证明）：生成与离线校验共用的纯函数规则。

全部哈希均为 SHA-256 的 64 位小写十六进制输出；规范 JSON 复用
``app.hashing.canonical_json``（键名排序、无空白、不转义非 ASCII）。

叶子（绑定 chain_id / seq / event_hash）
----------------------------------------
::

    leaf_hash = SHA256("merkle-leaf:v1:" + canonical_json({
        "chain_id": ..., "event_hash": ..., "seq": ...,
    }))

叶子摘要域中同时混入 chain_id 与 seq：同一 event_hash 出现在别的链或别的
序号下时叶子不同，证明不可跨链 / 跨序号挪用。

内部节点（固定域分隔）
----------------------
::

    node_hash = SHA256("merkle-node:v1:" + left_hash + right_hash)

前缀使内部哈希与叶子哈希分属不同域（第二原像防护）；左右两个哈希各自固定
为 64 个十六进制字符，直接拼接无歧义。

奇数层：最后一个节点与自身复制配对（node_hash(last, last)）后进入上一层，
因此任意数量（>=1）的事件集合都有唯一 Merkle 根。

快照摘要
--------
::

    snapshot_digest = SHA256("merkle-snapshot:v1:" + canonical_json({
        "chain_id": ..., "event_count": n,
        "merkle_root": ..., "tail_hash": ...,
    }))

把「生成证明时该链的事件集合（root + count）与链尾快照 tail_hash」绑定为
一个规范摘要：即使篡改者保留了一条对叶子仍成立的旧路径，只要事件总数或
链尾与出证时不一致，快照摘要即不匹配。

证明路径自叶向根排列，每项给出兄弟哈希与相对位置：``side="left"`` 表示
兄弟在左（当前节点在右，父哈希 = H(兄弟, 当前)）；``side="right"`` 反之。
奇数层末节点复制配对时，兄弟就是当前节点本身（``side="right"``）。
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .hashing import canonical_json, digest_hex

# 各摘要域的固定版本化前缀，互不相同。
LEAF_DOMAIN = "merkle-leaf:v1:"
NODE_DOMAIN = "merkle-node:v1:"
SNAPSHOT_DOMAIN = "merkle-snapshot:v1:"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# 校验阶段（按顺序返回首个失败阶段）。
STAGE_STRUCTURE = "structure"
STAGE_MERKLE_ROOT = "merkle_root"
STAGE_SNAPSHOT = "snapshot_digest"


def is_hex64(value: Any) -> bool:
    """是否为 64 位小写十六进制哈希字符串。"""
    return isinstance(value, str) and bool(_HEX64_RE.match(value))


def leaf_hash(chain_id: str, seq: int, event_hash: str) -> str:
    """计算绑定 chain_id / seq / event_hash 的 Merkle 叶子哈希。"""
    payload = canonical_json(
        {"chain_id": chain_id, "event_hash": event_hash, "seq": seq}
    )
    return digest_hex(LEAF_DOMAIN + payload)


def internal_hash(left: str, right: str) -> str:
    """计算内部节点哈希：固定域分隔前缀 + 左哈希 + 右哈希。"""
    return digest_hex(NODE_DOMAIN + left + right)


def snapshot_digest(
    chain_id: str, event_count: int, tail_hash: str, merkle_root: str
) -> str:
    """计算绑定链尾快照（含事件数与 Merkle 根）的规范摘要。"""
    payload = canonical_json(
        {
            "chain_id": chain_id,
            "event_count": event_count,
            "merkle_root": merkle_root,
            "tail_hash": tail_hash,
        }
    )
    return digest_hex(SNAPSHOT_DOMAIN + payload)


def expected_proof_length(event_count: int) -> int:
    """n 个叶子（n>=1）的包含证明长度：ceil(log2(n))；n=1 时为 0。"""
    return (event_count - 1).bit_length()


def _reduce_level(level: list[str]) -> list[str]:
    """相邻两两配对折叠一层；奇数个时末节点复制与自身配对。"""
    if len(level) == 1:
        return level
    nxt: list[str] = []
    for i in range(0, len(level) - 1, 2):
        nxt.append(internal_hash(level[i], level[i + 1]))
    if len(level) % 2 == 1:
        last = level[-1]
        nxt.append(internal_hash(last, last))
    return nxt


def tree_levels(leaves: list[str]) -> list[list[str]]:
    """返回自叶向根的各层；leaves 至少 1 个，levels[0] 即叶子层。"""
    if not leaves:
        raise ValueError("事件集合为空，无法构造 Merkle 树")
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        levels.append(_reduce_level(levels[-1]))
    return levels


def merkle_root(leaves: list[str]) -> str:
    """叶子列表 -> Merkle 根。"""
    return tree_levels(leaves)[-1][0]


def build_inclusion_proof(
    leaves: list[str], index: int
) -> tuple[list[dict[str, str]], str]:
    """为 ``leaves[index]`` 生成自叶向根的兄弟路径，并同时返回 Merkle 根。

    每项 ``{"side": "left"|"right", "sibling_hash": ...}``，side 表示兄弟
    相对当前节点的位置。奇数层末节点复制配对时兄弟为其自身（side=right）。
    """
    if not leaves:
        raise ValueError("事件集合为空，无法生成包含证明")
    if not 0 <= index < len(leaves):
        raise IndexError(f"叶子序号越界: {index}（叶子数 {len(leaves)}）")

    levels = tree_levels(leaves)
    proof: list[dict[str, str]] = []
    pos = index
    for level in levels[:-1]:
        if pos % 2 == 0:
            # 当前节点在左：兄弟在右；若它是奇数层末节点则复制自身配对。
            sibling = level[pos + 1] if pos + 1 < len(level) else level[pos]
            proof.append({"side": "right", "sibling_hash": sibling})
        else:
            proof.append({"side": "left", "sibling_hash": level[pos - 1]})
        pos //= 2
    return proof, levels[-1][0]


def apply_inclusion_proof(
    leaf: str, seq: int, siblings: list[dict[str, str]]
) -> Optional[str]:
    """仅凭叶子与兄弟路径重算 Merkle 根；路径结构非法时返回 None。

    校验方不需要知道事件集合本身：路径层数是否与 event_count 匹配由调用方
    另行（结构阶段）检查；这里只负责按 side 逐层折叠（奇数层复制配对的情形
    已体现在出证路径中——兄弟即叶子自身）。
    """
    current = leaf
    pos = seq
    for step in siblings:
        side = step.get("side")
        sibling = step.get("sibling_hash")
        if not is_hex64(sibling):
            return None
        if side == "left":
            current = internal_hash(sibling, current)
        elif side == "right":
            current = internal_hash(current, sibling)
        else:
            return None
        pos //= 2
    return current


def verify_event_proof(
    *,
    chain_id: str,
    seq: Any,
    event_hash: Any,
    event_count: Any,
    tail_hash: Any,
    merkle_root: Any,
    snapshot_digest_value: Any,
    siblings: Any,
) -> dict[str, Any]:
    """完全依据请求体重算并校验包含证明，**不读取数据库**。

    依次执行三个阶段，返回首个失败阶段（``failed_stage`` 为 null 表示全部
    通过）：

    1. ``structure``：字段形态与值域——chain_id 非空、event_count>=1、
       0<=seq<event_count、各哈希为 64 位小写十六进制、proof 长度恰为
       ceil(log2(event_count)) 且每项 side 合法；
    2. ``merkle_root``：重算叶子（绑定 chain_id/seq/event_hash）并按兄弟
       路径逐层折叠，结果须等于请求体给出的 merkle_root；
    3. ``snapshot_digest``：按规范 JSON 重算快照摘要，须等于请求体给出的
       snapshot_digest（绑定 event_count 与 tail_hash）。
    """
    # --- 阶段 1：结构 / 值域校验 ------------------------------------------------
    if not isinstance(chain_id, str) or not chain_id.strip():
        return _failure(STAGE_STRUCTURE, "chain_id 缺失或为空")
    if isinstance(event_count, bool) or not isinstance(event_count, int):
        return _failure(STAGE_STRUCTURE, "event_count 必须为整数")
    if event_count < 1:
        return _failure(STAGE_STRUCTURE, "event_count 必须 >= 1")
    if isinstance(seq, bool) or not isinstance(seq, int):
        return _failure(STAGE_STRUCTURE, "event.seq 必须为整数")
    if not 0 <= seq < event_count:
        return _failure(
            STAGE_STRUCTURE,
            f"event.seq 越界：须满足 0 <= seq < event_count（seq={seq}, "
            f"event_count={event_count}）",
        )
    for name, value in (
        ("event.event_hash", event_hash),
        ("tail_hash", tail_hash),
        ("merkle_root", merkle_root),
        ("snapshot_digest", snapshot_digest_value),
    ):
        if not is_hex64(value):
            return _failure(STAGE_STRUCTURE, f"{name} 必须是 64 位小写十六进制哈希")
    if not isinstance(siblings, list):
        return _failure(STAGE_STRUCTURE, "proof 必须为兄弟路径数组")
    expected_len = expected_proof_length(event_count)
    if len(siblings) != expected_len:
        return _failure(
            STAGE_STRUCTURE,
            f"proof 长度 {len(siblings)} 与 event_count={event_count} 不匹配"
            f"（应为 {expected_len}）",
        )
    for i, step in enumerate(siblings):
        if not isinstance(step, dict):
            return _failure(STAGE_STRUCTURE, f"proof[{i}] 必须为对象")
        if step.get("side") not in ("left", "right"):
            return _failure(
                STAGE_STRUCTURE, f"proof[{i}].side 必须为 left 或 right"
            )
        if not is_hex64(step.get("sibling_hash")):
            return _failure(
                STAGE_STRUCTURE, f"proof[{i}].sibling_hash 必须是 64 位十六进制哈希"
            )

    # --- 阶段 2：重算叶子 + 路径 -> Merkle 根 ----------------------------------
    computed_leaf = leaf_hash(chain_id, seq, event_hash)
    computed_root = apply_inclusion_proof(computed_leaf, seq, siblings)
    if computed_root is None:  # 理论上结构阶段已排除，防御性兜底
        return _failure(STAGE_STRUCTURE, "proof 路径无法折叠为 Merkle 根", computed_leaf)
    if computed_root != merkle_root:
        return _failure(
            STAGE_MERKLE_ROOT,
            "重算的 Merkle 根与请求体 merkle_root 不一致：叶子或兄弟路径与"
            "所声称的事件集合不符",
            computed_leaf,
            computed_root,
        )

    # --- 阶段 3：重算 snapshot_digest ------------------------------------------
    computed_snapshot = snapshot_digest(
        chain_id, event_count, tail_hash, computed_root
    )
    if computed_snapshot != snapshot_digest_value:
        return _failure(
            STAGE_SNAPSHOT,
            "重算的 snapshot_digest 与请求体不一致：chain_id / event_count / "
            "tail_hash / merkle_root 与出证快照不符",
            computed_leaf,
            computed_root,
            computed_snapshot,
        )

    return {
        "valid": True,
        "failed_stage": None,
        "reason": None,
        "computed_leaf_hash": computed_leaf,
        "computed_merkle_root": computed_root,
        "computed_snapshot_digest": computed_snapshot,
    }


def _failure(
    stage: str,
    reason: str,
    computed_leaf: Optional[str] = None,
    computed_root: Optional[str] = None,
    computed_snapshot: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "valid": False,
        "failed_stage": stage,
        "reason": reason,
        "computed_leaf_hash": computed_leaf,
        "computed_merkle_root": computed_root,
        "computed_snapshot_digest": computed_snapshot,
    }
