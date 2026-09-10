from reliability_router.models import ContextItem
from reliability_router.retrieval import InMemoryRetriever


def test_retriever_ranks_lexical_match() -> None:
    context = [
        ContextItem(id="a", text="The API latency target is 200 milliseconds."),
        ContextItem(id="b", text="The office has a blue wall."),
    ]
    result = InMemoryRetriever().retrieve("What is the API latency target?", context, top_k=2)
    assert [item.id for item in result] == ["a", "b"]
