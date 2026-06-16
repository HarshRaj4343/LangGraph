from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
import uuid
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState,
)
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from prompt import SYSTEM_PROMPT_TEMPLATE
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

store = InMemoryStore()

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")
class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

model = llm.with_structured_output(MemoryDecision)

MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information compared to CURRENT USER DETAILS.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

def remember_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")
    items = store.search(ns)
    existing = "\n".join(it.value["data"] for it in items) if items else "(empty)"
    last_msg = state["messages"][-1].content
    decision: MemoryDecision = model.invoke(
        [
            SystemMessage(content=MEMORY_PROMPT.format(user_details_content=existing)),
            {"role": "user", "content": last_msg},
        ]
    )

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(ns, str(uuid.uuid4()), {"data": mem.text})

    return {}  

def chat_node(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    items = store.search(ns)
    user_details = "\n".join(it.value["data"] for it in items) if items else ""

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT_TEMPLATE.format(
            user_details_content=user_details or "(empty)"
        )
    )

    response = llm.invoke([system_msg] + state["messages"])
    return {"messages": [response]}



builder = StateGraph(MessagesState)
builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

config = {"configurable": {"user_id": "u1"}}

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="Hi, my name is Harsh. I am a student at IIT Mandi and I like machine learning."
            )
        ]
    },
    config=config,
)

print("Assistant:")
print(result["messages"][-1].content)

ns = ("user", "u1", "details")
items = store.search(ns)

print("\nStored Memories:")
for item in items:
    print("-", item.value["data"])

# Second message (should use memories)
result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What do you know about me?"
            )
        ]
    },
    config=config,
)

print("\nAssistant (memory recall):")
print(result["messages"][-1].content)

# Check memories again
items = store.search(ns)

print("\nStored Memories After Second Turn:")
for item in items:
    print("-", item.value["data"])