"""Language gating: this corpus is **strictly English-only**.

India Code publishes most central acts twice — an English PDF and a Hindi
(Devanagari) PDF — as two bitstreams of the *same* item. Only the English one
may ever enter the corpus.

The rule this module enforces is deliberately asymmetric:

* Hindi is detected **eagerly**: any credible Hindi signal disqualifies a file.
* English must be **positively proven** from India Code's own metadata. The
  absence of a Hindi signal is *not* evidence of English (a filename that does
  not say "hindi" proves nothing), so a file we cannot classify is reported as
  ambiguous and never downloaded.

Evidence is ranked, and the winning rule is recorded on the bitstream as
``language_source`` so every stored document can explain *how* its language was
determined:

===============================  ====================================================
``language_source``              Evidence
===============================  ====================================================
``indiacode_bitstream_label``    The page labels the file "English" / "Hindi".
``indiacode_hindi_title``        The link text is India Code's own "Hindi Title" field.
``indiacode_devanagari_script``  The link text/label is written in Devanagari.
``indiacode_filename_pattern``   Hindi companion filename convention (``H<digit>…``).
``indiacode_metadata_title``     The link text is India Code's English title field
                                 (``Short Title`` / ``DC.title`` / ``citation_title``).
``indiacode_citation_pdf_url``   The file is the ``citation_pdf_url`` PDF of an item
                                 whose citation title is in Latin script.
===============================  ====================================================

Only the last two can ever yield ``"en"``; the rest can only yield ``"hi"``.
Every Hindi rule is evaluated before any English rule, so conflicting evidence
resolves to rejection rather than to an unsafe acceptance.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .errors import LanguageError
from .models import BitstreamRef, ParsedItem
from .utils import same_url

#: The only language this corpus accepts.
CORPUS_LANGUAGE = "en"

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
#: India Code's Hindi companion files are conventionally named ``H1967-15.pdf``.
#: Used only to *reject* a file as Hindi — never to accept one as English.
_HINDI_FILENAME_RE = re.compile(r"^H\d", re.IGNORECASE)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def _normalise(text: Optional[str]) -> str:
    """Fold a title/label for comparison: NFKC, lowercase, alphanumerics only.

    Also drops a leading article, so India Code's ``Short Title`` ("The
    Passports Act, 1967") and its ``DC.title`` ("Passports Act, 1967") compare
    equal. Devanagari survives as Devanagari, though combining marks are
    dropped along with punctuation — harmless here because both sides of every
    comparison are folded the same way.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).strip().lower()
    folded = re.sub(r"[^\w]+", " ", folded, flags=re.UNICODE).strip()
    folded = re.sub(r"^the\s+", "", folded)
    return re.sub(r"\s+", " ", folded)


#: Strings that, *on their own*, are an explicit language label on the page.
#: Matched against the whole (folded) label, never as a substring, so an act
#: titled "The English Education Act" is not mistaken for a label.
_ENGLISH_LABELS = frozenset(
    _normalise(value)
    for value in ("english", "english version", "in english", "eng", "en", "en_US")
)
_HINDI_LABELS = frozenset(
    _normalise(value)
    for value in (
        "hindi", "hindi version", "in hindi", "hin", "hi",
        "हिंदी",              # हिंदी
        "हिन्दी",        # हिन्दी
    )
)
#: Unambiguous single words, used when scanning a short phrase such as a table
#: cell reading "English version" or a column headed "Files(Eng)". Deliberately
#: excludes the two-letter codes "en"/"hi", which are far too easy to hit by
#: accident inside ordinary text.
_ENGLISH_WORDS = frozenset(_normalise(value) for value in ("english", "eng"))
_HINDI_WORDS = frozenset(
    _normalise(value)
    for value in ("hindi", "hin", "हिंदी", "हिन्दी")
)


@dataclass(frozen=True)
class LanguageVerdict:
    """The outcome of classifying one bitstream.

    ``language`` is ``None`` when no rule fired — meaning *undetermined*, which
    callers must treat as "do not download", never as English.
    """

    language: Optional[str] = None
    source: Optional[str] = None
    evidence: Optional[str] = None

    @property
    def is_english(self) -> bool:
        return self.language == CORPUS_LANGUAGE

    @property
    def is_undetermined(self) -> bool:
        return self.language is None


UNDETERMINED = LanguageVerdict()


# --- classification ------------------------------------------------------------


def classify_bitstream(
    *,
    filename: str,
    link_text: Optional[str] = None,
    explicit_label: Optional[str] = None,
    is_citation_pdf: bool = False,
    english_titles: Sequence[str] = (),
    hindi_titles: Sequence[str] = (),
) -> LanguageVerdict:
    """Classify one bitstream's language from India Code's own metadata.

    Parameters mirror what the landing page actually offers:

    ``link_text``
        The visible text of the file's ``<a>`` link. On India Code this is the
        document's title *in the file's own language*, which is why matching it
        against the item's English/Hindi title fields is strong evidence.
    ``explicit_label``
        A neighbouring cell/attribute that states the language outright.
    ``is_citation_pdf``
        Whether this file is the one advertised by ``<meta name="citation_pdf_url">``.
    ``english_titles`` / ``hindi_titles``
        India Code's own title fields, used as the reference strings.

    Returns a :class:`LanguageVerdict`; ``language is None`` means the language
    could not be established and the file must not be downloaded.
    """
    text = _normalise(link_text)

    # Stage 1 - reject. Any credible Hindi signal disqualifies the file, and it
    # is checked first on purpose: when evidence conflicts (say a file filed
    # under "Files(Eng)" but described in Devanagari) the safe reading wins and
    # the document is reported rather than quietly taken as English.
    if language_label_of(explicit_label, allow_phrase=True) == "hi":
        return LanguageVerdict(
            "hi", "indiacode_bitstream_label", f"page labels the file {explicit_label!r}"
        )
    if language_label_of(link_text, allow_phrase=False) == "hi":
        return LanguageVerdict(
            "hi", "indiacode_bitstream_label", f"page labels the file {link_text!r}"
        )
    for hindi_title in hindi_titles:
        if text and text == _normalise(hindi_title):
            return LanguageVerdict(
                "hi", "indiacode_hindi_title", f"link text matches Hindi Title {hindi_title!r}"
            )
    for value in (link_text, explicit_label):
        if is_devanagari_text(value):
            return LanguageVerdict(
                "hi", "indiacode_devanagari_script", "link text/label is in Devanagari script"
            )
    if filename and _HINDI_FILENAME_RE.match(filename):
        return LanguageVerdict(
            "hi", "indiacode_filename_pattern",
            f"filename {filename!r} follows the Hindi companion convention",
        )

    # Stage 2 - prove. English is only ever asserted on positive evidence.
    for raw, allow_phrase in ((explicit_label, True), (link_text, False)):
        if language_label_of(raw, allow_phrase=allow_phrase) == "en":
            return LanguageVerdict(
                "en", "indiacode_bitstream_label", f"page labels the file {raw!r}"
            )
    for english_title in english_titles:
        if text and text == _normalise(english_title):
            return LanguageVerdict(
                "en", "indiacode_metadata_title",
                f"link text matches India Code English title {english_title!r}",
            )
    if is_citation_pdf and _any_latin(english_titles):
        return LanguageVerdict(
            "en", "indiacode_citation_pdf_url",
            "file is the citation_pdf_url of an item titled in Latin script",
        )

    return UNDETERMINED


def language_label_of(text: Optional[str], *, allow_phrase: bool = True) -> Optional[str]:
    """Return ``"en"``/``"hi"`` if *text* is an explicit language label.

    With ``allow_phrase`` a short phrase containing an unambiguous language word
    ("English version", "PDF in Hindi") also counts; otherwise the whole string
    must be the language name. Returns ``None`` for anything else — notably for
    document titles, which must never be read as labels.
    """
    normalised = _normalise(text)
    if not normalised:
        return None
    if normalised in _HINDI_LABELS:
        return "hi"
    if normalised in _ENGLISH_LABELS:
        return "en"
    if allow_phrase:
        tokens = normalised.split()
        if len(tokens) <= 3:
            if any(token in _HINDI_WORDS for token in tokens):
                return "hi"
            if any(token in _ENGLISH_WORDS for token in tokens):
                return "en"
    return None


def contains_devanagari(text: Optional[str]) -> bool:
    """True if *text* contains any Devanagari character."""
    return bool(text) and bool(_DEVANAGARI_RE.search(text))


def is_devanagari_text(text: Optional[str]) -> bool:
    """True if *text* is **written in** Devanagari, not merely touched by it.

    Containing a Devanagari character is not enough. India Code publishes acts
    whose English titles quote a Hindi phrase — e.g. "The Viksit Bharat …
    (विकसित भारत—जी राम जी) Act, 2025" — and those are English documents. A
    genuine Hindi companion is labelled with the whole title in Devanagari, so
    comparing the two scripts separates the cases cleanly.
    """
    if not text:
        return False
    devanagari = len(_DEVANAGARI_RE.findall(text))
    if not devanagari:
        return False
    return devanagari >= len(_LATIN_LETTER_RE.findall(text))


def is_latin_text(text: Optional[str]) -> bool:
    """True if *text* is **written in** the Latin alphabet.

    Used to prefer a readable English description when India Code lists one file
    under several descriptions — never to decide a language, which always rests
    on the evidence rules above.
    """
    if not text:
        return False
    latin = len(_LATIN_LETTER_RE.findall(text))
    if not latin:
        return False
    letters = sum(1 for character in text if character.isalpha())
    return latin * 2 >= letters


def normalise_title(text: Optional[str]) -> str:
    """Public alias of the title-folding used for comparisons."""
    return _normalise(text)


def _any_latin(values: Iterable[str]) -> bool:
    return any(
        value and _LATIN_LETTER_RE.search(value) and not _DEVANAGARI_RE.search(value)
        for value in values
    )


# --- selection -----------------------------------------------------------------


def select_english_bitstream(item: ParsedItem) -> BitstreamRef:
    """Return the one bitstream of *item* that is proven to be English.

    Raises :class:`LanguageError` — never a silent fallback — when the item has
    no English file, or when the choice cannot be made confidently. The caller
    turns that into a ``FAILED`` result, so nothing is downloaded.
    """
    candidates = list(item.bitstreams)
    if getattr(item, "requested_bitstream_url", None):
        requested = item.requested_bitstream_url
        # Compared as URLs, not as text: India Code publishes the same bitstream
        # with different percent-encoding in different places, and a file that
        # *is* listed must not be reported as missing over an escaped bracket.
        candidates = [b for b in candidates if same_url(b.url, requested)]
        if not candidates:
            raise LanguageError(
                f"The requested file {requested} was not found among the bitstreams "
                f"listed on {item.source_url}, so its language cannot be verified. "
                "Refusing to download an unverified file (English-only corpus)."
            )

    if not candidates:
        raise LanguageError(f"No downloadable files were found for {item.source_url}.")

    english = [b for b in candidates if b.language == CORPUS_LANGUAGE]
    if len(english) == 1:
        return english[0]
    if len(english) > 1:
        primary = [b for b in english if b.is_primary]
        if len(primary) == 1:
            return primary[0]
        raise LanguageError(
            f"{len(english)} files on {item.source_url} are classified as English "
            f"({', '.join(b.filename for b in english)}) and none is unambiguously "
            "the primary one; refusing to guess which is the authoritative text."
        )

    undetermined = [b for b in candidates if b.language is None]
    if undetermined:
        raise LanguageError(
            "Language could not be determined from India Code metadata for "
            f"{', '.join(b.filename for b in undetermined)} on {item.source_url}. "
            "The file is NOT assumed to be English and was not downloaded "
            f"(available: {_describe(candidates)})."
        )

    raise LanguageError(
        f"No English version is available for {item.source_url}; this corpus is "
        f"English-only, so nothing was downloaded (available: {_describe(candidates)})."
    )


def _describe(bitstreams: Iterable[BitstreamRef]) -> str:
    return ", ".join(f"{b.filename}=" + (b.language or "unknown") for b in bitstreams)
