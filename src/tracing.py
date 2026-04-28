"""节点执行追踪：基于 LangGraph 的 stream() API，不侵入任何节点代码

用法：
    from src.tracing import run_with_trace
    result = run_with_trace(app, {"question": "..."})

也可保存 JSON 复盘：
    result = run_with_trace(app, state, save_path="evals/output/trace_xxx.json")
"""
import json
import time
from pathlib import Path
from typing import Any


# 单值显示长度上限（超过截断）
MAX_LEN = 200


def run_with_trace(
    app,
    initial_state: dict,
    *,
    console: bool = True,
    save_path: str | None = None,
) -> dict:
    """跑 graph 并记录每节点 state 变化、耗时。返回最终 state。

    - console: 是否打印到控制台（默认开）
    - save_path: 若给路径，写一份 JSON 复盘文件
    """
    steps = []
    t_start = time.time()
    last_state = dict(initial_state)

    if console:
        _hr()
        print("[trace] 起点 state:")
        _print_dict(initial_state)
        _hr()

    # stream_mode="updates" 每步只吐出当次节点写的字段（不是全 state）
    for chunk in app.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():
            t = time.time() - t_start
            steps.append({
                "node": node_name,
                "elapsed_total_s": round(t, 3),
                "update": _to_jsonable(update),
            })
            if console:
                print(f"\n[trace] [+{t:.2f}s] 节点 `{node_name}` 写入:")
                _print_dict(update, indent="  ")

            if isinstance(update, dict):
                last_state.update(update)

    elapsed = time.time() - t_start
    if console:
        _hr()
        print(f"[trace] 完成，总耗时 {elapsed:.2f}s | {len(steps)} 个节点")
        print("[trace] 最终 state:")
        _print_dict(last_state)
        _hr()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "initial": _to_jsonable(initial_state),
                "steps": steps,
                "final": _to_jsonable(last_state),
                "total_elapsed_s": round(elapsed, 3),
            }, f, ensure_ascii=False, indent=2)
        print(f"[trace] 复盘已写入: {save_path}")

    return last_state


# ─────── 辅助函数 ───────

def _hr():
    print("=" * 70)


def _short(v: Any) -> str:
    """单值压缩成一行显示"""
    if isinstance(v, list):
        if not v:
            return "[]"
        # 列表只看长度 + 第一个元素的概览
        first = _short(v[0]) if v else ""
        return f"[{len(v)} items] 首项: {first[:80]}"
    if isinstance(v, dict):
        keys = ", ".join(list(v.keys())[:5])
        return f"{{{keys}}}"
    s = str(v).replace("\n", " ")
    if len(s) <= MAX_LEN:
        return s
    return s[:MAX_LEN] + f"...(共 {len(s)} 字)"


def _print_dict(d: dict, indent: str = "  "):
    if not isinstance(d, dict):
        print(f"{indent}{_short(d)}")
        return
    for k, v in d.items():
        print(f"{indent}- {k}: {_short(v)}")


def _to_jsonable(obj: Any):
    """把 dataclass / 复杂对象转成可 json 序列化的形式"""
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ─────── 自测：用 hello_graph 验证工具能跑 ───────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.hello_graph import app

    run_with_trace(app, {"question": "电钻指示灯啥意思"})
