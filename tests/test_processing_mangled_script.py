"""Devanagari extracted through a legacy font, and why script alone misses it.

The gap these cover: a Hindi page set in an ASCII-mapped PDF font extracts as
*Latin* characters, so `script_profile` reports it as fully Latin and
`classify_page` admits it as English. `assess_pages` defends against that at the
document level with pooled vocabulary -- but only for a document that is wholly
transliterated. A **bilingual** document's genuine English half supplies the
function words, the document is cleared, and its damaged pages sail through.

The cost is not only that garbage is indexed. Quality is measured over the
admitted pages, so the document scores as damaged and is quarantined whole --
its sound English half with it.
"""

from __future__ import annotations

import pytest

from processing import config, language

# Verbatim from page 14 of the Indian Railways (Open Lines) General and
# Subsidiary Rules -- the Hindi rendering of the arrangement of rules that page
# 15 gives in English. Every character here is Latin.
MANGLED_HINDI = """
3.21
^ ^ ^ , f * TM3 ^ ^ T K f * TM* R H T
3.22
^%W$TK^$m*f$m^Wm
3.23
f ^ T O f e
3.24
f t l ^ t ^ R t
3.25
^ f f ^ W ^ f t W ^ R ^ f ^ t
3.26
W ^ f * P H # ^ ^ ^
3.27
l ^ ^ ^ ^ f t w ^ ^ ^ ^ ^ f ^ f ^ ; ^ ^ ^
3.28
^m^^^^^nm^^^^^m^nm^^^m^^
3.29
f l ^ f t W ^ ^ f f ^ ^ f t w f ^ ^ ^ S T O
3.30 wm^T^^zm^m^f^wm
3.31
' ^ ^ I W ^ H f T R t ^ M
3.32
3 # ^ ^ ( T T S T O ^ ) f ^ R ^ ^ f ^ ^ ^ ^ ^ R m ^ ^
"""

# Page 15 of the same document: the same rules, in English.
CLEAN_ENGLISH = """
3.21
SignalsonBracketpostsorSignalBndgeorGantry
3.22
Placingofmorethanonesignalonthesamepost
3.23
Electnc Repeater
3.24
BackHghts
C
Equipment of Signals
3.25
Obligation to provide fixed signals at stations
3.26
Connnissioningoffixedsignals
3.27
Minimum equipment of fixed signals at stations provided with manually
operated multiple aspect signalling
3.28
Minimum equipment of fixed signals at stations
"""

# The case the test must never break: English that carries almost no function
# words. A schedule of species names is the reason `classify_page` judges on
# script rather than vocabulary in the first place.
SPECIES_SCHEDULE = """
SCHEDULE I
PART I MAMMALS
Andaman wild pig Sus scrofa andamanensis
Bharal Ovis nahura
Binturong Arctictis binturong
Black buck Antilope cervicapra
Caracal Felis caracal
Capped langur Presbytis pileatus
Chinese pangolin Manis pentadactyla
Chinkara Gazella gazella bennetti
Clouded leopard Neofelis nebulosa
Desert cat Felis libyca
Desert fox Vulpes bucopus
Dugong Dugong dugon
Fishing cat Felis viverrina
Four horned antelope Tetraceros quadricornis
Gangetic dolphin Platanista gangetica
Giant squirrel Ratufa indica
Golden cat Felis temmincki
Golden langur Presbytis geei
Himalayan ibex Capra ibex
Hispid hare Caprolagus hispidus
Hoolock gibbon Hylobates hoolock
Indian lion Panthera leo persica
Indian wild ass Equus hemionus khur
Kashmir stag Cervus elaphus hanglu
Leopard cat Felis bengalensis
Lion tailed macaque Macaca silenus
Loris Loris tardigradus
Malabar civet Viverra megaspila
Marbled cat Felis marmorata
Markhor Capra falconeri
Musk deer Moschus moschiferus
Nilgiri langur Presbytis johni
Nilgiri tahr Hemitragus hylocrius
"""


class TestMangledScript:
    def test_it_names_legacy_font_devanagari_as_damaged(self):
        assert language.mangled_script(MANGLED_HINDI)["mangled"] is True

    def test_the_damage_is_invisible_to_the_script_test(self):
        # The whole reason this check has to exist: not one non-Latin character.
        profile = language.script_profile(MANGLED_HINDI)
        assert profile["letters"] > 0
        latin = profile["by_script"].get("LATIN", 0)
        unknown = profile["by_script"].get("UNKNOWN", 0)
        assert profile["letters"] - latin - unknown == 0

    def test_it_leaves_ordinary_english_alone(self):
        assert language.mangled_script(CLEAN_ENGLISH)["mangled"] is False

    def test_it_leaves_english_without_function_words_alone(self):
        # A schedule of Latin species names has a function-word rate near zero
        # and is unambiguously part of an English statute. Absence of vocabulary
        # is necessary evidence here and never sufficient -- the word shape has
        # to have collapsed as well, and this page's has not.
        result = language.mangled_script(SPECIES_SCHEDULE)
        assert result["mangled"] is False
        assert result["function_word_rate"] < config.LANGUAGE_MANGLED_FUNCTION_WORD_RATE

    def test_too_little_text_is_never_a_finding(self):
        result = language.mangled_script("^ ^ f * TM3 ^ ^ T K")
        assert result["mangled"] is False
        assert "too little to judge" in result["reason"]

    def test_it_reports_the_signals_behind_the_verdict(self):
        result = language.mangled_script(MANGLED_HINDI)
        assert result["mean_word_length"] < config.LANGUAGE_MANGLED_MEAN_WORD_LENGTH
        assert result["word_count"] >= config.LANGUAGE_MANGLED_MIN_WORDS
        assert "legacy font" in result["reason"]


#: A real page carries several hundred letters; `classify_page` refuses to judge
#: anything under `LANGUAGE_PAGE_MIN_LETTERS` (200) and says so. The blocks above
#: are single columns lifted from one page, so they are repeated to the length a
#: page actually has before being handed to a page-level test.
def _as_page_text(block: str) -> str:
    return (block + "\n") * 3


class _Page:
    """The minimum `classify_page` and `indexable_pages` read."""

    def __init__(self, page_number: int, text: str):
        self.page_number = page_number
        self.text = _as_page_text(text)
        self.selected_text = self.text


class TestClassifyPage:
    def test_a_damaged_page_is_not_english(self):
        verdict = language.classify_page(_Page(1, MANGLED_HINDI))
        assert verdict["verdict"] == "non_en"
        assert verdict["mangled_script"] is True

    def test_an_english_page_beside_it_is_unaffected(self):
        verdict = language.classify_page(_Page(2, CLEAN_ENGLISH))
        assert verdict["verdict"] == "en"
        assert "mangled_script" not in verdict

    def test_a_genuinely_devanagari_page_is_not_labelled_mangled(self):
        # It is non-English for the ordinary reason, which the script test sees.
        # The mangled label means specifically "another script wearing Latin
        # glyphs", and mislabelling would misreport why a page was excluded.
        page = _Page(3, "यह पृष्ठ देवनागरी में है " * 40)
        verdict = language.classify_page(page)
        assert verdict["verdict"] == "non_en"
        assert "mangled_script" not in verdict


#: The page-level fixtures above are contents listings, which is what page 14 of
#: the Railways document actually is -- but a *document* is established as English
#: by pooled vocabulary (`LANGUAGE_MIN_CLEAN_WORDS`, 150), and a column of run-on
#: heading text carries almost none. Document-level tests therefore need prose.
ENGLISH_PROSE = (
    "1. Short title and extent. This Act may be called the Sample Act, 1999 and "
    "it extends to the whole of India, save as otherwise provided in this "
    "section, and shall come into force on such date as the Central Government "
    "may, by notification in the Official Gazette, appoint in this behalf. "
    "Every rule made under this section shall be laid, as soon as may be after "
    "it is made, before each House of Parliament while it is in session for a "
    "total period of thirty days which may be comprised in one session or in "
    "two or more successive sessions, and if before the expiry of the session "
    "immediately following the session both Houses agree in making any "
    "modification in the rule, that rule shall thereafter have effect only in "
    "such modified form as may be agreed upon by both the Houses. "
)


class TestIndexablePages:
    def test_damaged_pages_are_excluded_and_english_ones_kept(self):
        pages = [_Page(n, MANGLED_HINDI if n % 2 == 0 else ENGLISH_PROSE)
                 for n in range(1, 11)]
        assessment = language.assess_pages(pages, metadata_language="en")
        kept = language.indexable_pages(pages, assessment)
        assert [p.page_number for p in kept] == [1, 3, 5, 7, 9]

    def test_the_document_is_routed_page_by_page_not_rejected_whole(self):
        # The failure this whole change exists to stop: the document is
        # bilingual, so it is routed, not quarantined.
        pages = [_Page(n, MANGLED_HINDI if n % 2 == 0 else ENGLISH_PROSE)
                 for n in range(1, 11)]
        assessment = language.assess_pages(pages, metadata_language="en")
        assert assessment.content_language == "bilingual_en"
        assert assessment.eligible_for_indexing is True

    def test_an_all_english_document_keeps_every_page(self):
        pages = [_Page(n, ENGLISH_PROSE) for n in range(1, 6)]
        assessment = language.assess_pages(pages, metadata_language="en")
        kept = language.indexable_pages(pages, assessment)
        assert len(kept) == 5
