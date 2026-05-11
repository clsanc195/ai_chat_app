from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

SYSTEM_PROMPT = """
You are a helpful assistant, excellent at providing helpful advice for gen ai questions
When users ask about non gen ai topics, you will respond with a simple statement about your expertise and decline.
"""

SUMMARY_THRESHOLD = 5

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

llm = ChatAnthropic(model="claude-opus-4-7")

def call_model(state: ChatState) -> dict:
    summary = state.get("summary", "")
    system_text = SYSTEM_PROMPT + (
        f"\n\nConversation summary so far:\n{summary}" if summary else ""
    )
    messages = [SystemMessage(content=system_text), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def summarize(state: ChatState) -> dict:
    older = state["messages"][:-2]
    if not older:
        return {}
    prior = state.get("summary", "")
    prompt = (
        "Summarize the following conversation concisely. Always keep the summary under 300 words.\n\n"
        "Prioritize conquered challenges, established restrictions or preferences\n\n"
        "Exclude sidetracks or unrelated information\n\n"
        "If a previous summary is provided, extend it rather than restarting it.\n\n"
        f"Previous summary:\n{prior or '(none)'}\n\n"
        "Conversation to summarize:\n"
        + "\n".join(f"{m.type}: {m.content}" for m in older)
    )
    new_summary = llm.invoke([HumanMessage(content=prompt)]).content
    return {
        "summary": new_summary,
        "messages": [RemoveMessage(id=m.id) for m in older],
    }


def should_summarize(state: ChatState) -> str:
    return "summarize" if len(state["messages"]) >= SUMMARY_THRESHOLD else END


graph_builder = StateGraph(ChatState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("summarize", summarize)
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges(
    "call_model", should_summarize, {"summarize": "summarize", END: END}
)
graph_builder.add_edge("summarize", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-stream"}}

    for chunk, metadata in graph.stream(
            {"messages": [{"role": "user", "content": "Tell me a short joke about Python."}]},
            config=config,
            stream_mode="messages",
    ):
        if chunk.content:
            print(chunk.content, end="", flush=True)