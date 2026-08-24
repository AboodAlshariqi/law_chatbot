import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

PROCESSED_DIR = Path("../data/processed")

# sjc.bh case type, confirmed live from the site's own search-form radio values (see 01_scraping/01a).
SJC_TYPE_CODES = {"M": "مدني", "J": "جنائي", "S": "شرعي", "T": "تجاري", "E": "انتخابات", "P": "توحيد المبادئ"}

# Both with and without the "ال" definite-article prefix — the corpus uses both
# ("مادة الأولى" in most laws, but plain "مادة اولي" in others, e.g. the 1976 Labor Law).
ORDINAL_WORDS_DEF = ["الاولي", "الأولى", "الثانية", "الثالثة", "الرابعة", "الخامسة",
                     "السادسة", "السابعة", "الثامنة", "التاسعة", "العاشرة"]
ORDINAL_WORDS_BARE = ["اولي", "أولى", "ثانية", "ثالثة", "رابعة", "خامسة",
                      "سادسة", "سابعة", "ثامنة", "تاسعة", "عاشرة"]
ORDINAL_WORDS = ORDINAL_WORDS_DEF + ORDINAL_WORDS_BARE
ORDINAL_TO_DIGIT = {}
for _i, (_def, _bare) in enumerate(zip(ORDINAL_WORDS_DEF, ORDINAL_WORDS_BARE)):
    ORDINAL_TO_DIGIT[_def] = str(_i + 1)
    ORDINAL_TO_DIGIT[_bare] = str(_i + 1)
ORDINAL_TO_DIGIT["الأولى"] = "1"
ORDINAL_TO_DIGIT["أولى"] = "1"

# Real-world dash usage in the corpus is inconsistent — some documents use a plain ASCII hyphen
# ("مادة -1-"), others an en-dash with spaces ("مادة – 1 –"), and a few even mix both within the
# same document (one side ASCII, the other en-dash). Treat all of them as interchangeable.
DASH = r"[-\u2013\u2014]"

# Compound numbers (e.g. treaty articles like "1.1", "2.3") captured fully, not truncated at the
# first digit. Accepts: "(N)" parentheses, "-N-"/"– N –" dash-wrapped, a bare digit, or an ordinal word.
NUMBER_PART = (
    r"(?:\(\s*(\d+(?:\.\d+)*)\s*\)"
    r"|" + DASH + r"\s*(\d+(?:\.\d+)*)\s*" + DASH +
    r"|(\d+(?:\.\d+)*)"
    r"|" + "|".join(ORDINAL_WORDS) + r")"
)
ARTICLE_HEADER = re.compile(r"(?:المادة|مادة)\s*" + NUMBER_PART)

# A real article header always starts a fresh sentence or a fresh chapter/section — never embedded
# mid-sentence. This replaces an earlier, less reliable approach (a content-based lookahead for
# citation phrasing like "من هذا القانون") which missed real bugs: e.g. a sentence citing two
# articles in a row ("...المنصوص عليها في المادة (12) أو ... المنصوص عليها في المادة (40) من هذا
# القانون.") only carries the disambiguating "من هذا القانون" phrase after the SECOND reference, so
# a lookahead-only check wrongly accepted the first one as a real header. Checking what precedes the
# match instead — sentence-final punctuation, or a chapter/section heading — is far more robust:
# confirmed live on real bug cases (K0421's Article 2 getting mislabeled as Article 12) and doesn't
# regress on any of the ~130 real + citation matches checked by hand in that same document.
SENTENCE_END = re.compile(r"[.:!؟\n]" + DASH + r"?\s*\Z")  # optional trailing dash covers "قرر الآتي:-"

# Chapter/section headings ("الباب الثاني"، "الفصل الأول"...) don't end in a period, so an article
# immediately following one wouldn't otherwise be recognized as a real header. Requires whitespace or
# start-of-text immediately before the heading word — "الفصل" also means "dismissal" in Arabic, and
# without a word-boundary check this matched as a false positive inside "والفصل" ("and dismissal"),
# a common phrase in labor-law text. Confirmed fixed against that exact case.
HEADING_MARKER = re.compile(r"(?:\A|\s)(?:الباب|الفصل|القسم)\s+\S+\s*[^.:!؟]{0,40}\Z")

# Real articles are never just a 2-4 word fragment. Any remaining segment under this size after
# citation exclusion gets merged into its neighbor instead of standing alone as a near-empty chunk.
MIN_SEGMENT_CHARS = 40


def article_no_from_match(m: re.Match) -> str | None:
    for g in m.groups():
        if g and (g.isdigit() or "." in g):
            return g
    matched_text = m.group(0)
    for word, digit in sorted(ORDINAL_TO_DIGIT.items(), key=lambda kv: -len(kv[0])):
        if word in matched_text:
            return digit
    return None


def is_real_header(text: str, m: re.Match) -> bool:
    """True if this article-number match is a genuine heading, not a reference buried inside
    another article's prose. A real heading starts a fresh sentence (or the document/a chapter) —
    a citation like "...المنصوص عليها في المادة (12)..." never does."""
    before = text[:m.start()]
    if before.strip() == "":
        return True
    if SENTENCE_END.search(before):
        return True
    return bool(HEADING_MARKER.search(before))


def merge_tiny_segments(segments: list[dict], min_chars: int) -> list[dict]:
    """Safety net: fold any segment that's still under min_chars into the segment that follows it,
    so nothing near-empty ever survives as its own chunk."""
    if not segments:
        return segments
    out = [dict(segments[0])]
    for seg in segments[1:]:
        if len(out[-1]["text"]) < min_chars:
            out[-1]["text"] = (out[-1]["text"] + " " + seg["text"]).strip()
        else:
            out.append(dict(seg))
    if len(out) > 1 and len(out[-1]["text"]) < min_chars:
        last = out.pop()
        out[-1]["text"] = (out[-1]["text"] + " " + last["text"]).strip()
    return out


def segment_legislation_by_article(text: str) -> list[dict]:
    """One segment per real article header - no size-based splitting here (that's the fallback
    splitter's job for genuinely oversized articles). Matches that are actually mid-sentence
    citations to another article are excluded as boundaries, and any still-tiny leftover segment
    gets merged into its neighbor."""
    all_matches = list(ARTICLE_HEADER.finditer(text))
    matches = [m for m in all_matches if is_real_header(text, m)]
    if not matches:
        return [{"text": text, "article_no": None}]
    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append({"text": text[start:end].strip(), "article_no": article_no_from_match(m)})
    return merge_tiny_segments(segments, MIN_SEGMENT_CHARS)


# BGE-M3's real context limit is 8192 tokens (confirmed from its config) - 6000 leaves a safety margin.
# Still used for lloc's per-article fallback split below (a handful of individual articles genuinely
# exceed it), but NO LONGER used for sjc/ccb - see the judgments cell for why.
MAX_TOKENS = 6000
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
fallback_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=MAX_TOKENS, chunk_overlap=200, separators=["\n\n", ". ", " ", ""]
)


def n_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_lloc_documents(path: Path) -> list[Document]:
    records = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    for r in records:
        if not r.get("normalized_text"):
            continue
        base_meta = {
            "source": "lloc",
            "doc_id": r["code"],
            "title": r.get("title") or "",
            "categories": ", ".join(r.get("categories", [])),
        }
        for seg in segment_legislation_by_article(r["normalized_text"]):
            docs.append(Document(page_content=seg["text"], metadata={**base_meta, "article_no": seg["article_no"]}))
    return docs


SJC_NO_JUDGMENT_PLACEHOLDER = "مجموعة الاحكام الصادرة من محكمة التمييز لا يوجد"


def load_judgment_documents(path: Path, source: str, id_field: str) -> list[Document]:
    """sjc.bh returns a fixed placeholder ("collection of judgments - none available") for a small
    number of case entries that have no judgment text published at all - not a scraping bug, a genuine
    source-side gap. Skipped here rather than embedded as a useless chunk.

    Also deduplicates by case number: a small number of sjc records (58 out of 9,094) are exact
    duplicates of another record filed under the SAME case number in the raw scraped data - kept
    once, first occurrence. This is unrelated to, and should not be confused with, DIFFERENT case
    numbers legitimately sharing identical judgment text (a real pattern - see the notebook intro)."""
    records = json.loads(path.read_text(encoding="utf-8"))
    docs = []
    skipped_placeholder = 0
    skipped_duplicate_id = 0
    seen_ids = set()
    for r in records:
        if not r.get("normalized_text"):
            continue
        if r["normalized_text"].strip() == SJC_NO_JUDGMENT_PLACEHOLDER:
            skipped_placeholder += 1
            continue
        doc_id = r.get(id_field) or r.get("key") or r.get("case_id")
        if doc_id in seen_ids:
            skipped_duplicate_id += 1
            continue
        seen_ids.add(doc_id)
        metadata = {"source": source, "doc_id": doc_id}
        if source == "sjc" and doc_id:
            parts = doc_id.split(" ")
            type_code = parts[1] if len(parts) > 1 else None
            metadata["case_type"] = SJC_TYPE_CODES.get(type_code)
        docs.append(Document(page_content=r["normalized_text"], metadata=metadata))
    if skipped_placeholder:
        print(f"{source}: skipped {skipped_placeholder} 'no judgment available' placeholder records")
    if skipped_duplicate_id:
        print(f"{source}: skipped {skipped_duplicate_id} exact-duplicate case-number records")
    return docs


lloc_docs = load_lloc_documents(PROCESSED_DIR / "lloc_normalized.json")
sjc_docs = load_judgment_documents(PROCESSED_DIR / "sjc_normalized.json", "sjc", "key")
ccb_docs = load_judgment_documents(PROCESSED_DIR / "ccb_normalized.json", "ccb", "case_id")

print(f"lloc: {len(lloc_docs)} article-segments")
print(f"sjc:  {len(sjc_docs)} per-case documents")
print(f"ccb:  {len(ccb_docs)} per-case documents")

missing_type = sum(1 for d in sjc_docs if not d.metadata.get("case_type"))
print(f"sjc documents missing case_type: {missing_type}")


def split_with_fallback(doc: Document) -> list[Document]:
    """One chunk per document by default. Only if it exceeds MAX_TOKENS does it get subdivided —
    every sub-chunk keeps the parent's full metadata (article_no/doc_id/case_type unchanged) plus a
    sub_chunk_index, so a citation still points back to the right article/case."""
    if n_tokens(doc.page_content) <= MAX_TOKENS:
        return [doc]
    pieces = fallback_splitter.split_text(doc.page_content)
    return [
        Document(page_content=piece, metadata={**doc.metadata, "sub_chunk_index": i})
        for i, piece in enumerate(pieces)
    ]


lloc_splits = []
lloc_oversized = 0
for d in lloc_docs:
    pieces = split_with_fallback(d)
    if len(pieces) > 1:
        lloc_oversized += 1
    lloc_splits.extend(pieces)

with_article_no = sum(1 for d in lloc_splits if d.metadata.get("article_no"))
print(f"{len(lloc_docs)} article-segments -> {len(lloc_splits)} chunks "
      f"({lloc_oversized} articles needed a fallback split, {with_article_no} chunks tagged with an article_no)")
lloc_splits[1].page_content[:300]

sjc_splits = sjc_docs
ccb_splits = ccb_docs

print(f"sjc: {len(sjc_docs)} cases -> {len(sjc_splits)} chunks (always 1:1, no fragmentation)")
print(f"ccb: {len(ccb_docs)} cases -> {len(ccb_splits)} chunks (always 1:1, no fragmentation)")


import pickle

all_splits = lloc_splits + sjc_splits + ccb_splits
OUT_PKL = Path("../data/processed/document_splits_v2.pkl")
OUT_PKL.write_bytes(pickle.dumps(all_splits))
print(f"Saved {len(all_splits)} total chunks -> {OUT_PKL}")

# Same {"page_content", "metadata"} record format the embeddings notebook (04) expects on upload.
# Written to a NEW file (not document_splits.json) so the file backing the CURRENT live vectorstore
# is left untouched until this is deliberately re-embedded and swapped in.
OUT_JSON = Path("../data/processed/document_splits_v2.json")
records = [{"page_content": d.page_content, "metadata": d.metadata} for d in all_splits]
OUT_JSON.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
print(f"Saved {len(records)} total chunks -> {OUT_JSON}")


