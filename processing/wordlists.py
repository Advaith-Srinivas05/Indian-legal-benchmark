"""Word lists used to judge whether extracted text is English legal prose.

Two lists, kept apart because they answer different questions.

:data:`FUNCTION_WORDS` is the *language* evidence. Function words are the part of
English that survives OCR damage best — they are short, high-frequency and
mostly unambiguous — so the share of readable words that are function words
still separates English from another language after a scan has been mangled.
They are also what a transliteration of Hindi or Gujarati conspicuously lacks.

:data:`LEGAL_WORDS` is *domain* evidence, added only to the extraction-quality
signal. Indian statutory prose is dense in this vocabulary, so a page of real
legal text scores high on it while OCR noise does not — but a document could be
perfectly good English and contain none of it (a schedule of place names, a
tariff table), which is why it never decides the language question on its own.

Both lists are deliberately small, fixed and dependency-free. Nothing here needs
a downloaded corpus, so the whole test suite stays offline.
"""

from __future__ import annotations

#: English function words — determiners, prepositions, conjunctions, pronouns,
#: auxiliaries — plus the legal-drafting connectives that behave like them
#: ("shall", "provided", "notwithstanding", "thereof").
FUNCTION_WORDS = frozenset("""
a an the and or but if then than that this these those of to in on at by for
with from as is are was were be been being am has have had do does did not no
nor so such which who whom whose what when where why how all any both each few
more most other some only own same too very can will just shall should may
might must would could it its he she her him his they them their there here
under over above below after before between into during without within upon
against about out off again further once because while until unless whether
either neither also per said thereof therein thereto thereunder thereafter
thereby hereby herein hereof hereto hereunder hereinafter notwithstanding
provided subject respectively aforesaid whereas whereof wherein whereby
accordingly otherwise pursuant every another one two three four five six seven
eight nine ten first second third
""".split())

#: Vocabulary characteristic of Indian statutory drafting.
LEGAL_WORDS = frozenset("""
act acts section sections subsection subsections clause clauses rule rules
regulation regulations schedule schedules chapter chapters part parts article
articles preamble proviso provisos explanation explanations
government governments state states central union territory territories
district districts court courts judge judges magistrate magistrates tribunal
authority authorities officer officers board committee commission council
person persons party parties company companies society societies
prescribed notification notifications order orders power powers apply applies
application applicable appoint appointed appointment commencement extent
title short repeal repealed savings amendment amended amend substituted
inserted omitted penalty penalties punishment punishable offence offences fine
fines imprisonment period periods date dates year years month months day days
appeal appeals revision inquiry enquiry hearing evidence witness document
documents register registered registration licence license permit certificate
india indian public private service services provision provisions force effect
made make making respect purpose purposes behalf manner conditions terms
payment payable tax taxes duty duties fee fees property land premises
""".split())

#: Everything that counts as a recognisable word when scoring quality.
COMMON_WORDS = FUNCTION_WORDS | LEGAL_WORDS
