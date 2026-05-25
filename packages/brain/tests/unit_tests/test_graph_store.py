"""Unit tests for the KnowledgeGraph store."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from june_brain.memory import KnowledgeGraph


@pytest.fixture
def memory_dir(tmp_path):
    with patch("june_brain.memory.MEMORY_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def graph(memory_dir):
    # Ensure the base SQLite schema exists (graph tables live in it).
    from june_brain.memory import Memory
    Memory("test_user")
    return KnowledgeGraph("test_user")


def test_add_node_returns_stable_id(graph):
    node = graph.add_node("Ana", kind="person")
    assert node["node_id"] == "person:ana"
    assert node["label"] == "Ana"
    assert node["kind"] == "person"


def test_add_node_is_upsert(graph):
    graph.add_node("Ana", kind="person", props={"relation": "sister"})
    updated = graph.add_node("Ana", kind="person", props={"relation": "sister", "city": "Lisbon"})
    assert updated["props"]["city"] == "Lisbon"
    nodes = graph.find_nodes(kind="person")
    assert len(nodes) == 1


def test_find_nodes_filters_by_kind_and_query(graph):
    graph.add_node("Ana", kind="person")
    graph.add_node("Lisbon", kind="place")
    graph.add_node("Anaheim", kind="place")
    people = graph.find_nodes(kind="person")
    assert [n["label"] for n in people] == ["Ana"]
    places = graph.find_nodes(kind="place", query="lis")
    assert [n["label"] for n in places] == ["Lisbon"]


def test_add_edge_and_neighbors(graph):
    ana = graph.add_node("Ana", kind="person")
    lisbon = graph.add_node("Lisbon", kind="place")
    graph.add_edge(src=ana["node_id"], dst=lisbon["node_id"], kind="lives_in")
    out = graph.neighbors(ana["node_id"], direction="out")
    assert len(out) == 1
    assert out[0]["node"]["label"] == "Lisbon"
    assert out[0]["edge"]["kind"] == "lives_in"
    in_ = graph.neighbors(lisbon["node_id"], direction="in")
    assert len(in_) == 1
    assert in_[0]["node"]["label"] == "Ana"


def test_remove_node_cascades_edges(graph):
    ana = graph.add_node("Ana", kind="person")
    lisbon = graph.add_node("Lisbon", kind="place")
    graph.add_edge(ana["node_id"], lisbon["node_id"], kind="lives_in")
    graph.remove_node(ana["node_id"])
    assert graph.get_node(ana["node_id"]) is None
    assert graph.neighbors(lisbon["node_id"]) == []


def test_mentions_near_matches_word_boundary(graph):
    graph.add_node("Ana", kind="person")
    graph.add_node("Marco", kind="person")
    hits = graph.mentions_near("I had coffee with Ana today")
    assert [h["label"] for h in hits] == ["Ana"]
    assert graph.mentions_near("banana bread") == []  # no word boundary on "ana"
