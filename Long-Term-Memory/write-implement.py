from dotenv import load_dotenv
load_dotenv()
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

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

store = InMemoryStore()

class MemoryDecision(BaseModel):
    should_write: bool = Field(
        description="Whether to store any memories"
    )

    memories: List[str] = Field(
        default_factory=list,
        description="Atomic user memories to store"
    )


model = llm.with_structured_output(MemoryDecision)


# ============================================================
# VERSION 1
# ============================================================

def remember_only_node(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")
    last_msg = state["messages"][-1].content

    decision: MemoryDecision = model.invoke(
        [
            SystemMessage(
                content=(
                    "Extract LONG-TERM memories from the user's message.\n"
                    "Only store stable, user-specific info "
                    "(identity, preferences, ongoing projects/goals).\n"
                    "Return should_write=false if nothing is worth storing."
                )
            ),
            {
                "role": "user",
                "content": last_msg,
            },
        ]
    )

    if decision.should_write:
        for memory in decision.memories:
            store.put(
                namespace,
                str(uuid.uuid4()),
                {"data": memory},
            )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": "Memory processed."
            }
        ]
    }

MEMORY_PROMPT = """
You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term
  (identity, stable preferences, ongoing projects/goals).
- Do NOT store greetings, chit-chat, temporary requests,
  or generic facts.
- If no new memory is worth storing, return should_write=false.
- Memories must be short atomic strings.

Return data matching the schema exactly.
"""


def chat_creates_memory_node(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    namespace = ("user", user_id, "details")

    existing_items = store.search(namespace)

    existing_texts = [
        item.value.get("data", "")
        for item in existing_items
        if item.value.get("data")
    ]

    user_details_content = (
        "\n".join(f"- {text}" for text in existing_texts)
        if existing_texts
        else "(empty)"
    )

    last_text = state["messages"][-1].content

    decision: MemoryDecision = model.invoke(
        [
            SystemMessage(
                content=MEMORY_PROMPT.format(
                    user_details_content=user_details_content
                )
            ),
            {
                "role": "user",
                "content": f"USER MESSAGE:\n{last_text}",
            },
        ]
    )
    if decision.should_write:
        for memory in decision.memories:
            store.put(
                namespace,
                str(uuid.uuid4()),
                {"data": memory},
            )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": "Noted."
            }
        ]
    }

builder = StateGraph(MessagesState)
builder.add_node(
    "chat",
    chat_creates_memory_node,
)

builder.add_edge(
    START,
    "chat",
)

builder.add_edge(
    "chat",
    END,
)

graph = builder.compile(store=store)
if __name__ == "__main__":

    config = {
        "configurable": {
            "user_id": "test_user"
        }
    }

    test_messages = [
        "My name is Harsh.",
        "My name is Harsh.",                    # exact duplicate
        "I like Python programming.",
        "I like Python programming.",          # exact duplicate
        "I enjoy Python programming.",         # semantic duplicate
        "I am a student at IIT Mandi.",
        "I am a student at IIT Mandi.",        # exact duplicate
    ]

    for i, msg in enumerate(test_messages, start=1):
        print(f"TEST MESSAGE #{i}")
        print(msg)

        graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": msg,
                    }
                ]
            },
            config,
        )

        memories = store.search(
            ("user", "test_user", "details")
        )

        print("\nCurrent Memories:")
        for idx, memory in enumerate(memories, start=1):
            print(f"{idx}. {memory.value['data']}")

        print(f"\nTotal Stored: {len(memories)}")