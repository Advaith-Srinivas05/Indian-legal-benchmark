"""Word-level helpers the question checks depend on."""

from __future__ import annotations

import pytest

from benchmark import terms


@pytest.mark.parametrize("text", [
    "fourteen years", "12%", "12 %", "100 percent", "90 per cent", "one lakh rupees",
    "two thousand and five hundred rupees", "30 days", "seventeen months", "five thousands rupees",
])
def test_every_way_the_corpus_states_a_quantity_is_recognised(text):
    assert terms.QUANTITY.search(text), text


@pytest.mark.parametrize("text", ["section 12", "anganwadi", "Chapter XVI", "12", "daysman"])
def test_a_bare_number_or_a_word_is_not_a_quantity(text):
    assert not terms.QUANTITY.search(text), text


def test_leakage_is_containment_so_a_lifted_question_scores_high():
    provision = "The authority shall decide every application within thirty days of its receipt " * 5
    assert terms.leakage("decide every application within thirty days", provision) == 1.0
    assert terms.leakage("how long does an official have to respond", provision) == 0.0


def test_a_fact_must_appear_as_a_whole_phrase():
    assert terms.appears_in("thirty days", "within  Thirty\nDays of receipt")
    assert not terms.appears_in("thirty day", "within thirty days")
    assert not terms.appears_in("", "anything")
