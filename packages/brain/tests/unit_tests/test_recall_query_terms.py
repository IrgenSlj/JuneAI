"""The lexical channel must search for content words, not function words.

The FTS query ORs its terms, and RRF credits a document once per channel it
appears in. So leaving "what is my" in the term list floods the capped candidate
pool with facts that merely contain "my", and those flooded facts then outrank
the fact the user asked for. Measured on the golden corpus this cost the
supersession category a third of its recall.
"""

from __future__ import annotations

from june_brain.memory.recall import RetrievalConfig, _query_terms


def test_function_words_are_dropped() -> None:
    assert _query_terms("What is my diet?") == ["diet"]


def test_content_words_survive_in_order() -> None:
    assert _query_terms("Which gym do I train at now?") == ["gym", "train", "now"]


def test_rare_tokens_are_preserved_verbatim() -> None:
    # The lexical channel is the only one that reliably finds these.
    assert _query_terms("Kovoplast") == ["kovoplast"]
    assert _query_terms("IEC 61400 certification") == ["iec", "61400", "certification"]


def test_all_stopword_query_still_searches() -> None:
    # Returning nothing would silently disable the channel; searching for the
    # literal words is a worse-but-honest answer.
    assert _query_terms("who is who") == ["who", "is"]


def test_single_characters_are_ignored() -> None:
    assert _query_terms("a b diet") == ["diet"]


def test_duplicates_collapse() -> None:
    assert _query_terms("diet diet DIET") == ["diet"]


def test_term_count_is_capped() -> None:
    query = " ".join(f"term{i}" for i in range(40))
    assert len(_query_terms(query)) == 12


def test_candidate_pool_default_is_the_measured_one() -> None:
    # Pinned deliberately: this value was chosen from the golden-corpus sweep,
    # so a silent change to it is a silent change to recall quality.
    assert RetrievalConfig().candidate_pool == 15
    assert RetrievalConfig.load().candidate_pool == 15
