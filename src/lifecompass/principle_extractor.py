"""
Stage 2 (direct mode) -- extract. One model call reads the whole transcript, as paragraphs, and writes the
principles itself, copying the speaker's words. There are no unit numbers, windows, blocks, editor or cold reader:
everything those stages decided is in prompts/extract_prompt.txt.

Code does only what is mechanical: it builds the input, parses the answer (a bad item is skipped, not the whole
reply), skips an empty text, and gives the domain "Other" when the model's domain is not in the list. It never
cuts, trims, merges, ranks or judges a principle.

One mechanical guard enforces the verbatim rule (design decision 2026-11-08 and 2026-11-09): every principle that is stored is one
continuous passage of the transcript, with stutter fragments (a lone letter or the bare article left hanging before the full word)
taken out. A text that is not such a passage is repaired to the closest one (the model's words are aligned to the transcript);
only a text with no real anchor in the transcript is not stored, and it is written whole in the run log.
The transcriber's marker for speech it could not make out ([غير واضح]) is never stored either: a text that contains it is split at it and each
side is kept as its own principle (prompt rule 9 asks the model to do that; this is the mechanical fallback, applied after the guard above
because a repair can pull the marker back in). Whether a side holds a whole thought is not judged by code.
"""
from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, model_validator

from . import llm, tunables
from .document import SourceDocument
from .runlog import block as _block, detail as _detail

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExtractedPrinciple(BaseModel):
    """One principle as the model wrote it. Accepts the shapes a model may use for it, so that no finished principle is thrown
    away because of its shape (design decision 2026-11-07): a bare string, a missing or empty domain (-> "Other"), and the text under
    another key (principle / quote / passage / content / statement)."""
    text: str
    domain: str          # the reply shape is written in prompts/extract_prompt.txt (no JSON-Schema text is sent); a missing one is filled in before validation

    @model_validator(mode="before")
    @classmethod
    def _accept_the_shapes_a_model_may_use(cls, data):
        if isinstance(data, str):
            return {"text": data, "domain": "Other"}
        if isinstance(data, dict):
            data = dict(data)
            if not isinstance(data.get("text"), str):
                for key in ("principle", "quote", "passage", "content", "statement"):
                    if isinstance(data.get(key), str):
                        data["text"] = data[key]
                        break
            if not isinstance(data.get("domain"), str) or not data["domain"].strip():
                data["domain"] = "Other"
        return data


class ExtractionResult(BaseModel):
    items: list[ExtractedPrinciple]


def _load_prompt() -> str:
    return (PROJECT_ROOT / "prompts" / "extract_prompt.txt").read_text(encoding="utf-8")


def render(doc: SourceDocument, title: str) -> str:
    """The request: the title and the whole talk, one paragraph per document entry, separated by a blank line
    (prompts/extract_request.txt, part of the stage fingerprint)."""
    template = (PROJECT_ROOT / "prompts" / "extract_request.txt").read_text(encoding="utf-8").rstrip("\n")
    return template.format(title=title, transcript="\n\n".join(u.text for u in doc.units))


def _locate(doc: SourceDocument, text: str):
    """A plain fact for the run log: where `text` sits in the transcript as one continuous passage, as
    (first paragraph, first char, last paragraph, last char), else None. Paragraph breaks are only layout, so a
    run of spaces or line breaks counts as one space on both sides. Nothing is acted on."""
    chars, origin = [], []                   # the transcript with whitespace collapsed, and where each char came from
    for u in doc.units:
        if chars and chars[-1] != " ":
            chars.append(" ")
            origin.append((u.index, 0))
        for k, ch in enumerate(u.text):
            if ch.isspace():
                if chars and chars[-1] != " ":
                    chars.append(" ")
                    origin.append((u.index, k))
                continue
            chars.append(ch)
            origin.append((u.index, k))
    needle = " ".join(text.split())
    i = "".join(chars).find(needle) if needle else -1
    if i < 0:
        return None
    (p1, c1), (p2, c2) = origin[i], origin[i + len(needle) - 1]
    return p1, c1, p2, c2 + 1


# A repaired passage is never more than this many times longer than what the model wrote: it may bring back the few words the model
# left out, but it never bridges two different places of the talk.
MAX_REPAIR_GROWTH = tunables.get("extract_guard", "max_repair_growth")   # config.toml [extract_guard]
# ... and at least this share of the model's words must be found in it. Below that the text was not copied from this transcript at
# all (there is nothing to repair it to) and it is left out.
MIN_REPAIR_COVERAGE = tunables.get("extract_guard", "min_repair_coverage")


def _bare(word: str) -> str:
    """A word without its punctuation (quotation marks, commas, full stops ...), to compare what was said and not how it is
    punctuated."""
    bare = "".join(ch for ch in word if not unicodedata.category(ch).startswith("P"))
    return bare or word


def _is_fragment(keys: list, i: int) -> bool:
    """A stutter fragment: a lone Arabic letter right before a word that starts with the same letter ("م من", "ب بيت", "و وقت"),
    or the definite article left hanging on its own ("ال كتاب"). Orthography decides, never meaning: attached clitics are written
    attached, so a detached letter like that is a word cut off before it was finished."""
    key = keys[i]
    if key == "ال":
        return True
    return len(key) == 1 and "\u0621" <= key <= "\u064a" and i + 1 < len(keys) and keys[i + 1].startswith(key)


def _without_fragments(words: list) -> list:
    keys = [_bare(w) for w in words]
    return [w for i, w in enumerate(words) if not _is_fragment(keys, i)]


def _clean_view(doc: SourceDocument):
    """The transcript as the extraction sees it: (its words without stutter fragments, the same words as bare keys, the words joined
    by single spaces)."""
    words = _without_fragments(" ".join(u.text for u in doc.units).split())
    return words, [_bare(w) for w in words], " ".join(words)


def _is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _on_word_boundaries(joined: str, needle: str) -> bool:
    """True when `needle` occurs in `joined` as whole words: its first character starts a word (an opening quotation mark may come
    before it, and so may the attached connective letter ف / و that extract rule 5 lets a passage leave out), and its last character
    ends one (punctuation may cling to it). A passage that starts or stops inside a word is not a passage of the transcript."""
    if not needle:
        return False
    i = joined.find(needle)
    while i >= 0:
        k = i
        while k > 0 and _is_punctuation(joined[k - 1]):
            k -= 1
        starts = k == 0 or joined[k - 1] == " " or (joined[k - 1] in "فو" and (k == 1 or joined[k - 2] == " "))
        j = i + len(needle)
        while j < len(joined) and _is_punctuation(joined[j]):
            j += 1
        ends = j == len(joined) or joined[j] == " "
        if starts and ends:
            return True
        i = joined.find(needle, i + 1)
    return False


def _widen_to_whole_words(joined: str, needle: str):
    """`needle` found in `joined` but starting or stopping inside a word: the same passage widened to the whole words at both ends
    (the transcript's own words), or None when `needle` is not in `joined` at all."""
    j = joined.find(needle)
    if j < 0:
        return None
    lo = joined.rfind(" ", 0, j) + 1
    hi = joined.find(" ", j + len(needle))
    return joined[lo:(len(joined) if hi < 0 else hi)]


def _conform(view, text: str):
    """Makes `text` one continuous passage of the transcript (as `_clean_view` shows it). Returns (new_text, how) with `how` one of
    "as is" (it already was one), "stutter removed" (it was, once its stutter fragments were taken out) or "repaired" (the model's
    words are aligned to the transcript and the closest continuous passage is returned, in the transcript's own words), or None
    when fewer than MIN_REPAIR_COVERAGE of its words are found there. Mechanical: it never judges meaning, and what it returns is
    always a continuous passage of the cleaned transcript."""
    words, keys, joined = view
    mine = text.split()
    kept = _without_fragments(mine)
    if not kept:
        return None
    needle = " ".join(kept)
    if _on_word_boundaries(joined, needle):
        return (text, "as is") if len(kept) == len(mine) else (needle, "stutter removed")
    widened = _widen_to_whole_words(joined, needle)      # it starts or stops inside a word: the whole words are the passage
    if widened is not None:
        return widened, "repaired"
    blocks = [b for b in SequenceMatcher(None, keys, [_bare(w) for w in kept], autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return None
    limit = MAX_REPAIR_GROWTH * len(kept)
    lo = hi = max(range(len(blocks)), key=lambda n: (blocks[n].size, -n))
    start, end, matched = blocks[lo].a, blocks[lo].a + blocks[lo].size, blocks[lo].size
    while True:                                           # grow over the nearest neighbour while the passage stays within the limit
        options = []
        if lo > 0:
            left = blocks[lo - 1]
            options.append((start - (left.a + left.size), end - left.a, left, -1))
        if hi + 1 < len(blocks):
            right = blocks[hi + 1]
            options.append((right.a - end, right.a + right.size - start, right, 1))
        options = [o for o in options if o[1] <= limit]
        if not options:
            break
        _gap, _span, block, side = min(options, key=lambda o: o[0])
        matched += block.size
        if side < 0:
            lo, start = lo - 1, block.a
        else:
            hi, end = hi + 1, block.a + block.size
    if matched / len(kept) < MIN_REPAIR_COVERAGE:
        return None
    return " ".join(words[start:end]), "repaired"


def conform_to_transcript(transcript_text: str, text: str):
    """The same literal-text guard the principles go through, for any other text that must be one continuous passage of the
    transcript (the standalone image captions). `transcript_text` is the transcript as stored (SourceDocument.full_text()).
    Returns (new_text, how) with how "as is" / "stutter removed" / "repaired", or None when fewer than MIN_REPAIR_COVERAGE of
    the words are in the transcript (see `_conform`). Mechanical: it never judges meaning."""
    words = _without_fragments(transcript_text.split())
    return _conform((words, [_bare(w) for w in words], " ".join(words)), text)


DOMAINS = {
    "Self", "Relationships", "Family", "Money", "Career", "Health",
    "Learning", "Faith", "Society", "Decisions", "Other",
}
UNCLEAR_MARKER = "[غير واضح]"          # what the transcriber writes in place of speech it could not make out (transcribe prompt)
_CLINGING_PUNCTUATION = " \t\n،,.؟?!:؛;"      # what can cling to the marker's far side, and is not part of the passage


def split_at_unclear(text: str) -> list:
    """The passages of `text` on either side of the unclear-speech marker, in order, without the marker (and without the punctuation that
    clung to it). A text with no marker comes back as it is, in a list of one; a text that is only the marker gives []. Mechanical: whether
    a side holds a whole thought is not judged here (extract rule 9 gives that to the model); a side of any length is kept."""
    if UNCLEAR_MARKER not in text:
        return [text]
    return [piece for piece in (part.strip().lstrip(_CLINGING_PUNCTUATION).strip() for part in text.split(UNCLEAR_MARKER)) if piece]


def extract_principles(doc: SourceDocument, title: str, *, log=print) -> tuple:
    """Returns ([(text, domain), ...], model_name). Raises llm.StageSkippedError when no model could answer."""
    _detail(f"EXTRACT input: title={title!r}, {len(doc.units)} paragraph(s), "
            f"{sum(len(u.text.split()) for u in doc.units)} words, doc_hash={doc.doc_hash}")
    res, model = llm.generate_json(render(doc, title), ExtractionResult, system=_load_prompt(),
                                   context="extract[1]", log=log)
    _detail(f"EXTRACT reply parsed: {len(res.items)} item(s) from {model}")
    out = []
    view = _clean_view(doc)
    repaired = rejected = destuttered = split_unclear = 0
    for n, p in enumerate(res.items, 1):
        text = p.text.strip()
        trimmed = text != p.text
        if not text:
            _detail(f"  EXTRACT item {n}: empty text -> skipped (domain={p.domain!r})")
            continue
        fixed = _conform(view, text)
        if fixed is None:
            rejected += 1
            _detail(f"  EXTRACT item {n}: LEFT OUT, fewer than {MIN_REPAIR_COVERAGE:.0%} of its words are in the transcript, so "
                    f"there is no passage to repair it to (domain={p.domain!r}; {len(text.split())} words); it is NOT stored. "
                    f"Its full text follows.")
            _block(f"EXTRACT item {n} LEFT OUT (as the model wrote it)", text)
            log(f"  ! Extract item {n} was left out: its words are not in the transcript (the full text is in the run log).")
            continue
        new_text, how = fixed
        if how != "as is":
            if how == "repaired":
                repaired += 1
            else:
                destuttered += 1
            _detail(f"  EXTRACT item {n}: {how.upper()}: {len(text.split())} word(s) as the model wrote it -> "
                    f"{len(new_text.split())} word(s) of the transcript")
            _block(f"EXTRACT item {n} as the model wrote it (before: {how})", text)
            text = new_text
        pieces = split_at_unclear(text)
        if pieces != [text]:        # the marker for speech nobody could make out is never part of a principle (prompt rule 9)
            split_unclear += 1
            _detail(f"  EXTRACT item {n}: contains the unclear-speech marker -> SPLIT at it into {len(pieces)} passage(s) "
                    f"({', '.join(str(len(q.split())) for q in pieces) or 'none'} word(s)); the marker is not stored, each side is kept as "
                    f"its own principle (whether a side holds a whole thought is not judged here)")
            _block(f"EXTRACT item {n} as it was before the split at the unclear-speech marker", text)
        for text in pieces:
            where = _locate(doc, text)
            _detail(f"  EXTRACT item {n}: domain={p.domain!r}" + ("" if p.domain in DOMAINS else " -> Other (not in the list)")
                    + f"; {len(text.split())} words; "
                    + ((f"found as one continuous passage in paragraph {where[0]}, chars {where[1]}-{where[3]}"
                        if where[0] == where[2] else
                        f"found as one continuous passage from paragraph {where[0]} char {where[1]} to paragraph "
                        f"{where[2]} char {where[3]} (crosses {where[2] - where[0]} paragraph break(s))") if where
                       else "not found letter for letter in the raw transcript (stutter fragments were taken out, or it was repaired)")
                    + ("; surrounding whitespace removed" if trimmed else ""))
            _block(f"EXTRACT item {n} text", text)
            out.append((text, p.domain if p.domain in DOMAINS else "Other"))
    log(f"  Extract: {len(out)} principle(s) written by {model}"
        + (f"; stutter removed from {destuttered}" if destuttered else "")
        + (f"; {repaired} repaired to the transcript's own words" if repaired else "")
        + (f"; {rejected} left out (not in the transcript)" if rejected else "")
        + (f"; {split_unclear} split at the unclear-speech marker" if split_unclear else "") + ".")
    return out, model
