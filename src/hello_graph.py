"""LangGraph hello world: 最小可跑骨架"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: str


def greet(state: State) -> dict:
    return {"answer": f"你说的是：{state['question']}（来自 LangGraph hello world）"}


graph = StateGraph(State)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

app = graph.compile()


if __name__ == "__main__":
    result = app.invoke({"question": "电钻指示灯啥意思"})
    print(result)
