from dotenv import load_dotenv
load_dotenv()
import uuid
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

POSTGRES_URI = "postgresql://postgres:postgres@localhost:5442/postgres"

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

store = PostgresStore.from_conn_string(POSTGRES_URI).__enter__()
store.setup()

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant.

Known user details:
{user_details_content}

Use the stored memories when relevant.
"""

MEMORY_PROMPT = """
You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS:
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific information worth storing.
- Store stable preferences, identity details, goals, and ongoing projects.
- Mark is_new=true only if the memory is genuinely new.
- Keep memories short and atomic.
- No speculation.
- If nothing should be stored, return should_write=false.
"""

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_model = llm.with_structured_output(MemoryDecision)

def remember_node(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]

    ns = ("user", user_id, "details")

    items = store.search(ns)

    existing = (
        "\n".join(item.value["data"] for item in items)
        if items
        else "(empty)"
    )

    last_message = state["messages"][-1].content

    decision: MemoryDecision = memory_model.invoke(
        [
            SystemMessage(
                content=MEMORY_PROMPT.format(
                    user_details_content=existing
                )
            ),
            {
                "role": "user",
                "content": last_message,
            },
        ]
    )

    if decision.should_write:
        for memory in decision.memories:
            if memory.is_new:
                store.put(
                    ns,
                    str(uuid.uuid4()),
                    {"data": memory.text},
                )

    return {}

def chat_node(
    state: MessagesState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]

    ns = ("user", user_id, "details")

    items = store.search(ns)

    user_details = (
        "\n".join(item.value["data"] for item in items)
        if items
        else "(empty)"
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=SYSTEM_PROMPT_TEMPLATE.format(
                    user_details_content=user_details
                )
            )
        ]
        + state["messages"]
    )

    return {"messages": [response]}

builder = StateGraph(MessagesState)

builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

graph = builder.compile(store=store)

config = {
    "configurable": {
        "user_id": "u1"
    }
}

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

print("\nStored Memories:")
for item in store.search(ns):
    print("-", item.value["data"])

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

print("\nStored Memories:")
for item in store.search(ns):
    print("-", item.value["data"])