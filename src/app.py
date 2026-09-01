import asyncio
import functools
import json
import os
import re
import time
import tomllib
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional

import chainlit as cl
import numpy as np
from chainlit.input_widget import MultiSelect, Select, Slider
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

APP_DIR = Path(__file__).parent
PERSIST_DIRECTORY = str(APP_DIR.parent / "data" / "chroma_v3")
PROCESSED_DIRECTORY = APP_DIR.parent / "data" / "processed"
SECRETS_TOML_PATH = APP_DIR / ".streamlit" / "secrets.toml"

VECTORSTORE_DRIVE_FOLDER_ID = "1ovba37GozvlJOCHAuNchsoUIlXm8UMmW"


def _download_vectorstore():
    import shutil
    import tempfile

    import gdown

    staging = Path(tempfile.mkdtemp(prefix="chroma_dl_"))
    gdown.download_folder(id=VECTORSTORE_DRIVE_FOLDER_ID, output=str(staging), quiet=False, use_cookies=False)

    candidates = [staging] + [p for p in staging.iterdir() if p.is_dir()]
    source = next((c for c in candidates if (c / "chroma.sqlite3").exists()), None)
    if source is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("Downloaded vectorstore folder is missing chroma.sqlite3 — check the Drive folder contents.")

    Path(PERSIST_DIRECTORY).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), PERSIST_DIRECTORY)
    shutil.rmtree(staging, ignore_errors=True)


def _load_secret(name: str) -> str:
    if SECRETS_TOML_PATH.exists():
        try:
            with open(SECRETS_TOML_PATH, "rb") as f:
                data = tomllib.load(f)
            key = data.get(name)
            if key:
                return key
        except Exception:
            pass
    return os.environ.get(name, "")


NORMALIZED_SOURCES = {
    "sjc": (str(PROCESSED_DIRECTORY / "sjc_normalized.json"), "key"),
    "ccb": (str(PROCESSED_DIRECTORY / "ccb_normalized.json"), "case_id"),
}

ALL_SOURCES = {"lloc", "sjc", "ccb"}
SOURCE_LABELS = {"lloc": "التشريعات", "sjc": "السوابق القضائية", "ccb": "المحكمة الدستورية"}

# --- Session logging -------------------------------------------------------------------
# The client reported wrong answers during a live demo and none of it was recoverable
# afterwards -- there was no record of what was asked, what was retrieved, or what was
# answered. Every turn is now appended to a JSONL file so a session can be reconstructed
# and debugged later instead of relying on memory.
LOG_DIR = APP_DIR / "logs"
LOG_PATH = LOG_DIR / "sessions.jsonl"


def log_turn(kind, **fields):
    """Append one turn to the session log. Never raises -- logging must not break the app."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": cl.user_session.get("id") or "",
            "kind": kind,
            **fields,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def sources_for_log(sources):
    """Compact source record: enough to diagnose a bad answer without storing full texts."""
    return [
        {
            "source": s.get("source"),
            "doc_id": s.get("doc_id"),
            "article_no": s.get("article_no"),
            "cited": bool(s.get("cited")),
            "chars": len(s.get("text") or ""),
        }
        for s in sources
    ]

SEARCH_SCORE_THRESHOLD = 0.95

# --- Change 3: Search gets its own, slightly looser threshold. ---
SEARCH_SCORE_THRESHOLD_BROWSE = 1.05

# Chat's candidate pool. Raised from 20 to 300: with only 20 candidates, a well-worded
# question on a criminal-law topic fills the entire pool with Penal Code articles before a
# single judgment appears, so the source guarantee below has nothing to promote. 300 gives

CHAT_FETCH_K = 300


CHAT_FETCH_K_PER_SOURCE = 30
CHAT_RESULT_K = 8
# Its OWN threshold: SEARCH_SCORE_THRESHOLD is shared with Search in this file, so reusing
# it here would move Search's cutoff as a side effect. Looser than Search's because the
# per-source split already prevents one type from crowding the pool.
CHAT_SCORE_THRESHOLD = 1.05

# A law's own header, e.g. "مرسوم بقانون رقم (19) لسنة 2001". Used two ways: to read a
# retrieved law's (number, year), and to find the same pattern cited inside a judgment --
# which is the only way to tell WHICH law a judgment applied. Verified against the corpus:
# sjc records carry no linking metadata at all (title/categories/article_no/section_heading
# are empty for all 11,339 chunks), and 33.8% of them cite a decree number in their text.
LAW_CITATION = re.compile(r"(?:مرسوم\s+بقانون|قانون)\s+رقم\s*\(?\s*(\d+)\s*\)?\s*لسنة\s*(\d{4})")

# Search's base fetch pool -- kept as the floor.
SEARCH_FETCH_POOL = 300

# --- Change 2: Search's fetch pool now scales with the user's requested k_value. ---
SEARCH_FETCH_MULTIPLIER = 20   # fetch up to 20x the requested result count
SEARCH_FETCH_POOL_MAX = 2000   # hard cap so worst-case latency stays bounded


class ThresholdMMRRetriever(BaseRetriever):
    """Combines a relevance-score cutoff with MMR diversity for Chat -- LangChain's built-in
    retriever only supports one or the other (search_type is either "mmr" or
    "similarity_score_threshold", not both at once). Kept over plain similarity search (which
    scored HIGHER on the narrow single-answer benchmark, 75.0% vs ~65%) because a separate
    multi-source test in the eval notebook showed plain similarity search missing every relevant
    court case entirely on broader questions, while MMR correctly surfaced sources across both
    lloc and sjc -- see appchainlit_tuned.py's module docstring for the real numbers."""

    vectordb: Chroma
    k: int = 6
    fetch_k: int = CHAT_FETCH_K
    score_threshold: float = SEARCH_SCORE_THRESHOLD
    lambda_mult: float = 0.5
    filter: Optional[dict] = None
    max_chars: Optional[int] = None

    def _guarantee_both(self, idxs, kept):
        """If MMR chose only one source type but the other IS available, give it one slot.

        Deliberately displaces only the LAST (weakest) of MMR's picks, so the top of the
        result set is untouched and the answer keeps its best-matching document. Returns
        idxs unchanged when both types are already present, when only one type was
        selected, or when the other type genuinely has nothing above the threshold --
        nothing is invented.
        """
        wanted = self._wanted_sources()
        if not ("lloc" in wanted and ({"sjc", "ccb"} & wanted)):
            return idxs

        def is_leg(i):
            return kept[i][1].get("source") == "lloc"

        have_leg = any(is_leg(i) for i in idxs)
        have_jud = any(not is_leg(i) for i in idxs)
        if (have_leg and have_jud) or not idxs:
            return idxs

        need_leg = not have_leg          # which type is missing from the answer
        chosen = set(idxs)
        for cand in range(len(kept)):    # kept is in relevance order -- take the best one
            if cand not in chosen and is_leg(cand) == need_leg:
                return idxs[:-1] + [cand]
        return idxs

    def _fetch(self, query_vec, n_results, where):
        raw = self.vectordb._collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "embeddings"],
            where=where,
        )
        docs, metas, dists, embs = raw["documents"][0], raw["metadatas"][0], raw["distances"][0], raw["embeddings"][0]
        return [(d, m, e) for d, m, dist, e in zip(docs, metas, dists, embs) if dist <= self.score_threshold]

    def _wanted_sources(self):
        """Which sources the user selected, read back out of the Chroma filter that
        build_source_filter() produced. None means 'no filter', i.e. everything."""
        if not self.filter:
            return set(ALL_SOURCES)
        src = self.filter.get("source")
        if isinstance(src, dict):
            return set(src.get("$in") or [])
        return {src} if src else set(ALL_SOURCES)

    def _selected_source_names(self):
        """The source names to query separately, read back out of the Chroma filter."""
        if not self.filter:
            return sorted(ALL_SOURCES)
        src = self.filter.get("source")
        if isinstance(src, dict):
            return sorted(src.get("$in") or ALL_SOURCES)
        return [src] if src else sorted(ALL_SOURCES)

    def _query_source(self, query_vec, source):
        """CHAT_FETCH_K_PER_SOURCE candidates from ONE source, threshold applied here."""
        raw = self.vectordb._collection.query(
            query_embeddings=[query_vec],
            n_results=CHAT_FETCH_K_PER_SOURCE,
            include=["documents", "metadatas", "distances", "embeddings"],
            where={"source": source},
        )
        return [
            (d, m, e, float(dist))
            for d, m, dist, e in zip(raw["documents"][0], raw["metadatas"][0],
                                     raw["distances"][0], raw["embeddings"][0])
            if float(dist) <= CHAT_SCORE_THRESHOLD
        ]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        query_vec = self.vectordb.embeddings.embed_query(query)

        candidates = []
        for source in self._selected_source_names():
            candidates.extend(self._query_source(query_vec, source))
        if not candidates:
            return []

        # Deduplicate before MMR. article_no is part of the key so two different articles
        # of the same law stay distinct rather than collapsing into one.
        seen, unique = set(), []
        for d, m, e, dist in candidates:
            key = (m.get("source", ""), m.get("doc_id", ""), m.get("article_no", ""), d)
            if key not in seen:
                seen.add(key)
                unique.append((d, m, e, dist))
        unique.sort(key=lambda x: x[3])          # relevance order across all sources

        kept = [(d, m, e) for d, m, e, _ in unique]

        # --- source guarantee (Chat) -------------------------------------------------
        # Same failure Search had: a well-worded question about a criminal-law topic
        # returns six Penal Code articles and not one judgment, even though the client
        # asked for both sources. MMR diversifies in EMBEDDING space, which is not the
        # same as source-type diversity -- six near-identical articles of one law are
        # "diverse enough" for MMR while carrying no case law at all.
        #
        # Two distinct causes, handled separately:
        #   a) the type never entered the candidate pool  -> targeted top-up query
        #   b) it is in the pool but MMR did not pick it   -> displace one slot, below
        wanted = self._wanted_sources()
        want_legislation = "lloc" in wanted
        want_judgments = bool({"sjc", "ccb"} & wanted)
        if want_legislation and want_judgments:
            def _is_leg(meta):
                return meta.get("source") == "lloc"
            if not any(_is_leg(m) for _, m, _ in kept):
                kept += self._fetch(query_vec, max(self.k, 10), {"source": "lloc"})
            if not any(not _is_leg(m) for _, m, _ in kept):
                kept += self._fetch(query_vec, max(self.k, 10), {"source": {"$in": ["sjc", "ccb"]}})

        kept_embs = [e for _, _, e in kept]
        idxs = maximal_marginal_relevance(np.array(query_vec), kept_embs, lambda_mult=self.lambda_mult, k=min(self.k, len(kept)))
        idxs = self._guarantee_both(list(idxs), kept)
        results = [Document(page_content=kept[i][0], metadata=kept[i][1]) for i in idxs]
        _add_article_header(results)
        if self.max_chars:
            for d in results:
                if len(d.page_content) > self.max_chars:
                    d.page_content = d.page_content[: self.max_chars] + " […]"
        return results


# Article numbers are stored in metadata but are absent from the chunk text itself: of 600
# random legislation chunks, 2 mention their own article number in the prefix. A model
# quoting the passage therefore cannot cite it, and guesses -- which silently drops the
# source from قائمة المصادر even when the legal answer is exactly right.
_HAS_ARTICLE = re.compile(r"ماد[\u0629\u0647]\s*\(?\s*(\d+)")


def _add_article_header(docs):
    """Prefix each legislation chunk with its own article number, in the app's own format.

    Written as "مادة (N)" -- the same shape mark_cited() looks for -- so a model copying the
    heading produces a citation that matches. Judgments are left alone: their appeal number
    is already in their prefix, which is why they cite reliably today.
    """
    for d in docs:
        if d.metadata.get("source") != "lloc":
            continue
        article = str(d.metadata.get("article_no") or "").strip()
        if not article:
            continue
        # Skip when the opening already states this article, so laws that do carry the
        # heading in their text do not get a duplicate one.
        head = d.page_content[:400]
        if any(m.group(1) == article for m in _HAS_ARTICLE.finditer(head)):
            continue
        d.page_content = f"[\u0645\u0627\u062f\u0629 ({article})]\n" + d.page_content


def build_source_filter(selected_sources):
    selected = set(selected_sources)
    if not selected or selected == ALL_SOURCES:
        return None
    if len(selected) == 1:
        return {"source": next(iter(selected))}
    return {"source": {"$in": sorted(selected)}}


SYSTEM_TEMPLATE = """انت مساعد قانوني متخصص في القانون البحريني. استخدم المقاطع القانونية التالية فقط للاجابة على السؤال في نهاية النص.

قواعد صارمة يجب اتباعها:
- استند فقط الى النصوص المرفقة، ولا تخترع اي معلومة غير موجودة فيها.
- اذا لم تكن الاجابة موجودة في النصوص المرفقة، صرح بذلك بوضوح ولا تخمن.
- اذكر المصدر الدقيق لكل معلومة (رقم المادة او رقم القضية)، وانقل اسم القانون او القرار كما هو مكتوب حرفياً في النص المرفق فقط. لا تنسب اي معلومة الى اسم قانون لم يرد ذكره صراحة في النص المرفق.
- قبل استخدام اي مقطع، تحقق ان موضوعه يتعلق فعلاً بموضوع السؤال. اذا كان المقطع من مجال قانوني مختلف (مثل قرار اسكاني او اداري لا علاقة له بالسؤال) فلا تستخدمه ولا تذكره في الاجابة، حتى لو تشابهت بعض الكلمات.
- نظم اجابتك القانونية بالشكل التالي، حسب نوع المصادر المتوفرة فعلاً في النصوص المرفقة:

  ## الإجابة العامة
  ابدأ دائماً باجابة عامة موجزة (من ثلاثة الى خمسة اسطر) بلغة واضحة ومباشرة، تلخص موقف القانون البحريني من السؤال كما يظهر من النصوص المرفقة وحدها.
  - اكتبها بلغة مفهومة لغير المتخصص، دون الاغراق في المصطلحات، مع الحفاظ على الدقة القانونية.
  - يجب ان تكون خاصة بالقانون البحريني تحديداً، لا اجابة عامة عن القانون في المطلق ولا عن قوانين دول اخرى.
  - لا تذكر في هذا القسم ارقام المواد ولا ارقام القضايا، فموضعها الاقسام التالية.
  - لا تضف في هذا القسم اي حكم لا يستند الى النصوص المرفقة، فهو تلخيص لها وليس معرفة عامة.

  ## القوانين ذات الصلة
  لكل مادة قانونية تجيب على السؤال:
  - اذكر اسم القانون او القرار كما هو مكتوب حرفياً في النص المرفق، ورقم المادة.
  - اذكر نص الحكم القانوني نفسه (اقتباسا او تلخيصا دقيقا لما ورد في المادة، دون اضافة او تأويل).
  - وضح الشروط او الحالات التي ينطبق فيها هذا الحكم، ان وردت في النص (مثل: متى يسري، على من ينطبق، ما الاستثناءات المذكورة صراحة).

  ## تطبيقات قضائية
  لكل حكم من محكمة التمييز يتعلق فعلاً بموضوع السؤال:
  - اذكر رقم الطعن وسنته (او رقم القاعدة ان وجد)، كما ورد حرفياً في النص المرفق.
  - لخص وقائع القضية بايجاز: ما هو النزاع، ومن هم الاطراف، وما الذي حدث.
  - اشرح كيف طبقت المحكمة النص القانوني على هذه الوقائع تحديدا، وما التفسير الذي اعتمدته المحكمة للمادة القانونية (لا تكتفِ بذكر رقم القضية دون شرح كيف طبقت المحكمة القانون).
  - اذكر ما انتهت اليه المحكمة (قبول الطعن، رفضه، نقض الحكم، تاييده... الخ) وبأي أساس قانوني.

  قواعد التنظيم:
  - قسم "الإجابة العامة" يأتي دائماً اولاً، ويُكتب في كل الاحوال طالما وُجد في النصوص المرفقة ما يجيب على السؤال.
  - اذا توفر في النصوص المرفقة نص تشريعي فقط (بدون احكام قضائية)، اكتب "الإجابة العامة" ثم "القوانين ذات الصلة" فقط، ولا تكتب قسم "تطبيقات قضائية" ولا تشر الى غيابه.
  - اذا توفرت احكام قضائية فقط (بدون نص تشريعي مباشر)، اكتب "الإجابة العامة" ثم "تطبيقات قضائية" فقط، واذكر ضمن شرحك اي مادة قانونية ورد ذكرها صراحة داخل نص الحكم القضائي نفسه (فهي جزء من النص المرفق، لا اضافة من عندك).
  - اذا توفر النوعان معا، اكتب الاقسام الثلاثة بالترتيب: الإجابة العامة، ثم القوانين ذات الصلة، ثم تطبيقاتها القضائية، بحيث يفهم القارئ الخلاصة اولا، ثم الحكم القانوني، ثم كيف طبقته المحكمة عمليا.
  - اذا لم تجد في النصوص المرفقة ما يجيب على السؤال، فلا تكتب اي من هذه الاقسام، واكتفِ بجملة واضحة تفيد بعدم توفر نصوص ذات صلة في قاعدة البيانات.
  - لا تخترع قسما لا يوجد له سند في النصوص المرفقة تحت اي ظرف.
- اختم اجابتك دائماً بسطر منفصل يبدأ حرفياً بـ "المصادر_المستخدمة:" يليه فقط ارقام المواد و/او ارقام القضايا التي استندت اليها فعلاً في متن الاجابة (وليس تلك التي استبعدتها او ذكرتها لتوضيح عدم صلتها)، مفصولة بفواصل. اذا لم تستند الى اي نص مرفق فعلاً (سؤال غير متعلق بالقانون، او لا توجد اجابة في النصوص المرفقة)، اكتب "المصادر_المستخدمة: لا يوجد".

النصوص القانونية:
{context}

السؤال: {question}

الاجابة القانونية المدعومة بالمصادر:"""

CITED_SOURCES_MARKER = "المصادر_المستخدمة:"

# Every chunk arrives labelled "[n] المصدر=... | الوثيقة=... | المادة=..." (see
# docs_to_context). Asking for that number back is far more reliable than asking the model
# to restate an article or appeal number in a shape a regex can re-match: measured on this
# corpus, models write "مادة 2" for article 3, "material 202", bare "97 أ، 97 ب", and the
# court's own "الطعن رقم 2/00001/2023/35" for a doc_id stored as "1 M 2023 K 00". A copied
# integer has no such variants.
_INDEX_CITATION_RULE = (
    "\n- كل مقطع مرفق يبدأ برقم بين قوسين مثل [1] أو [2]. عند الاستشهاد بمقطع، اكتب رقمه"
    " بين قوسين مباشرة بعد المعلومة المأخوذة منه."
    "\n- في السطر الاخير اكتب: المصادر_المستخدمة: ثم ارقام المقاطع التي اعتمدت عليها فعلاً"
    " مفصولة بفواصل، مثل: المصادر_المستخدمة: [1], [4], [6]. لا تذكر رقماً لم يرد ضمن"
    " المقاطع المرفقة، ولا تكتب اسم القانون بدل الرقم."
)

SYSTEM_TEMPLATE = SYSTEM_TEMPLATE.replace(
    "- استند فقط الى النصوص المرفقة، ولا تخترع اي معلومة غير موجودة فيها.",
    "- استند فقط الى النصوص المرفقة، ولا تخترع اي معلومة غير موجودة فيها."
    + _INDEX_CITATION_RULE,
    1,
)

QA_CHAIN_PROMPT = PromptTemplate.from_template(SYSTEM_TEMPLATE)

# ---------------------------------------------------------------------------
# Nemotron Super 120B writes English words into Arabic answers. SYSTEM_TEMPLATE above is
# NOT modified -- the 550B obeys it and needs none of this.
#
# Three observed failures, all from this model on this app:
#   1. "material 113" instead of "مادة (113)". This is the damaging one: mark_cited()
#      looks for مادة followed by the article number, so the correct article -- retrieved
#      at rank 1 -- was recorded as uncited and vanished from قائمة المصادر.
#   2. "namely" and "merits of the claim" dropped mid-sentence into an Arabic answer.
#   3. "We need to respond in Arabic formal (Fus..." -- internal reasoning printed as the
#      answer, on a prompt that had explicitly asked for Arabic only (617 Latin chars).
#
# Built by WRAPPING SYSTEM_TEMPLATE rather than rewriting it: the legal formatting rules
# already work on this model, only the language discipline fails. The rules go first (read
# before the documents) and a one-line reminder goes last, immediately before the answer
# marker -- the final thing the model reads before generating.
#
# Note rule 3: appeal numbers genuinely contain Latin letters in the corpus
# ("192 M 1999 K 34"), so a blanket "no Latin characters" rule would break every judgment
# citation. The rule bans foreign WORDS, not the Latin script.
_NEMOTRON_AR_RULES = """تنبيه لغوي ملزم — اقرأه قبل كل شيء:

١. اكتب الاجابة كاملة بالعربية الفصحى. ممنوع منعاً باتاً استعمال اي كلمة اجنبية داخل النص.
٢. كلمة "مادة" تُكتب بالعربية دائماً. ممنوع كتابة material او article او اي مقابل اجنبي لها.
   مثال خاطئ وقع فعلاً: "المصادر_المستخدمة: material 113" — والصواب: "المصادر_المستخدمة: مادة 113".
   هذا الخطا يفقد الاستشهاد صلته بمصدره، فلا يظهر المصدر للمستخدم اطلاقاً.
٣. الاستثناء الوحيد: ارقام الطعون تُنقل كما وردت حرفياً في النص المرفق، ولو تضمنت حروفاً
   لاتينية (مثل: الطعن 192 M 1999 K 34). انقلها كما هي ولا تترجمها.
٤. لا تكتب تفكيرك ولا خطواتك الداخلية ولا مقدمات مثل "دعني افكر". ابدا الاجابة مباشرة
   بالعنوان الاول.

"""

_NEMOTRON_AR_REMINDER = (
    "\n\nتذكير اخير قبل الكتابة: كل كلمة في اجابتك بالعربية، وكلمة (مادة) بالعربية،"
    " ولا تكتب اي تفكير داخلي."
)

_ANSWER_MARKER = "\n\nالاجابة القانونية المدعومة بالمصادر:"

NEMOTRON_SUPER_TEMPLATE = (
    _NEMOTRON_AR_RULES
    + SYSTEM_TEMPLATE.replace(_ANSWER_MARKER, _NEMOTRON_AR_REMINDER + _ANSWER_MARKER)
)
NEMOTRON_SUPER_PROMPT = PromptTemplate.from_template(NEMOTRON_SUPER_TEMPLATE)


def prompt_for(provider):
    """The chat prompt for one model. Falls back to QA_CHAIN_PROMPT, so any provider
    without its own entry behaves exactly as before."""
    return (LLM_PROVIDERS.get(provider) or {}).get("prompt") or QA_CHAIN_PROMPT

# ---------------------------------------------------------------------------
# Attachment prompts.
#
# SYSTEM_TEMPLATE above is deliberately left untouched -- these are separate
# templates for the two attachment features, so the behaviour of an ordinary
# question is unchanged.
#
# REVIEW keeps the "المصادر_المستخدمة:" contract because it retrieves from the
# corpus and its sources feed the same panel. DOCQA has no retrieved sources at
# all, so it has no such line and no sources panel.
# ---------------------------------------------------------------------------

REVIEW_TEMPLATE = """انت مساعد قانوني متخصص في القانون البحريني. امامك مستند مرفق من المستخدم، بالاضافة الى نصوص قانونية بحرينية مسترجعة من قاعدة البيانات. مهمتك مراجعة المستند في ضوء هذه النصوص.

قواعد صارمة يجب اتباعها:
- ميّز دائماً بين ما ورد في **المستند المرفق** وما ورد في **النصوص القانونية**. لا تخلط بينهما ولا تنسب حكماً قانونياً الى المستند ولا بند عقد الى القانون.
- استند فقط الى المستند والنصوص القانونية المرفقة، ولا تخترع اي معلومة غير موجودة فيهما.
- اذكر اسم القانون ورقم المادة كما ورد حرفياً في النصوص المرفقة فقط، ولا تنسب معلومة الى قانون لم يرد ذكره صراحة.
- اذا لم تتضمن النصوص القانونية المسترجعة ما يغطي بنداً من بنود المستند، صرّح بذلك ولا تخمّن حكمه.
- انت لا تقدم استشارة قانونية نهائية، بل مراجعة اولية تستند الى النصوص المرفقة وحدها.

نظم اجابتك بالشكل التالي:

  ## ملخص المستند
  وصف موجز (من ثلاثة الى خمسة اسطر) لطبيعة المستند واطرافه وموضوعه، كما يظهر من المستند نفسه.

  ## النصوص القانونية ذات الصلة
  لكل مادة قانونية تتصل بموضوع المستند:
  - اذكر اسم القانون ورقم المادة كما ورد حرفياً في النص المرفق.
  - اذكر الحكم القانوني الذي تقرره المادة.

  ## الملاحظات على المستند
  لكل ملاحظة:
  - اذكر البند او العبارة محل الملاحظة من المستند (اقتباساً موجزاً).
  - اذكر المادة القانونية التي تتصل بها.
  - وضّح طبيعة الملاحظة: هل البند متوافق مع النص القانوني، ام يخالفه، ام ان المسألة غير مغطاة في النصوص المسترجعة.
  - اذا كان البند مخالفاً، اذكر ما تقرره المادة صراحة دون اقتراح صياغة بديلة من عندك.

  قواعد التنظيم:
  - اذا لم تتوفر نصوص قانونية ذات صلة في النصوص المسترجعة، اكتب "ملخص المستند" فقط، ثم صرّح بوضوح بعدم توفر نصوص ذات صلة في قاعدة البيانات.
  - لا تكتب قسماً لا يوجد له سند في المستند او في النصوص المرفقة.
- اختم اجابتك دائماً بسطر منفصل يبدأ حرفياً بـ "المصادر_المستخدمة:" يليه فقط ارقام المواد و/او ارقام القضايا التي استندت اليها فعلاً في متن الاجابة، مفصولة بفواصل. اذا لم تستند الى اي نص قانوني مرفق، اكتب "المصادر_المستخدمة: لا يوجد".

المستند المرفق ({doc_name}):
{document}

النصوص القانونية المسترجعة من قاعدة البيانات:
{context}

طلب المستخدم: {question}

المراجعة القانونية:"""

DOCQA_TEMPLATE = """انت مساعد قانوني. امامك مستند مرفق من المستخدم. اجب على سؤاله من هذا المستند وحده.

قواعد صارمة يجب اتباعها:
- استند فقط الى نص المستند المرفق ادناه. لا تستعن باي معرفة خارجية ولا باي قانون غير مذكور داخل المستند نفسه.
- اذا لم تكن الاجابة موجودة في المستند، صرّح بذلك بوضوح ولا تخمّن.
- عند نقل بند او عبارة، انقلها كما وردت حرفياً، واذكر موضعها (رقم البند او المادة او العنوان) ان كان مذكوراً في المستند.
- لا تُصدر حكماً على مدى مطابقة المستند للقانون البحريني في هذا الوضع، فالنصوص القانونية غير متاحة لك هنا. اذا سُئلت عن ذلك، وجّه المستخدم الى استخدام "{review_command}".

المستند المرفق ({doc_name}):
{document}

السؤال: {question}

الاجابة من المستند:"""

REVIEW_PROMPT = PromptTemplate.from_template(REVIEW_TEMPLATE)
DOCQA_PROMPT = PromptTemplate.from_template(DOCQA_TEMPLATE)

_vectordb = None
_vectordb_lock = Lock()


def get_vectorstore():
    global _vectordb
    if _vectordb is not None:
        return _vectordb
    with _vectordb_lock:
        if _vectordb is None:
            if not Path(PERSIST_DIRECTORY).exists():
                print(f"Vectorstore not found at {PERSIST_DIRECTORY} — downloading from Drive...")
                _download_vectorstore()
            embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
            _vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        return _vectordb


def _condense_question(chain, question, chat_history):
    """Rewrite a follow-up into a standalone question, using the chain's own question
    generator so behaviour matches ConversationalRetrievalChain exactly. Runs WITHOUT
    streaming -- only the final answer should appear in the chat."""
    try:
        return chain.question_generator.run(question=question, chat_history=chat_history)
    except Exception:
        return question


# Signatures of a transient upstream failure, matched against str(exception).lower().
#
# "provider returned error" was added after the free Nemotron endpoint spent an evening
# rejecting ~7 of every 8 requests under load. It returns the SAME overload as either 404
# or 502 depending on how the request was routed, so matching on the status code alone
# caught only some of them and the 404s fell straight through to the user as an error.
# Matching the message text covers both.
OVERLOAD_MARKERS = ("502", "overloaded", "provider returned error")
STREAM_MAX_ATTEMPTS = 5


def _invoke_with_retry(chain, payload, max_attempts=6, base_delay=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e).lower()
            if attempt == max_attempts or not any(marker in msg for marker in OVERLOAD_MARKERS):
                raise
            time.sleep(min(base_delay * (2 ** (attempt - 1)), 30))


# Every model lives in exactly one self-contained entry -- its own model string, endpoint,
# secret name, memory window and context budget. Nothing is shared between entries and
# nothing is inferred from the provider key, so adding or removing a model cannot change
# how any other model behaves. The two original entries below are byte-for-byte the same
# configuration they had before this table existed; only the surrounding plumbing moved.
#
# "kind" selects the client class: "groq" -> ChatGroq, "openai" -> ChatOpenAI against any
# OpenAI-compatible endpoint (which is what OpenRouter and most of the rest expose).
#
# NOTE ON THE ARABIC-NATIVE MODELS at the bottom: they are NOT on OpenRouter -- searched
# all 396 models in its live catalogue, and allam/fanar/jais/silma return zero hits. Each
# needs its own endpoint, so their base_url is left as None on purpose. Selecting one
# before you fill that in gives a clear Arabic message telling you what is missing; it
# does not raise and does not affect any other model.
LLM_PROVIDERS = {
    # --- the two originals, unchanged ------------------------------------------------
    "openrouter": {
        "label": "OpenRouter — Nemotron (الاساسي)",
        "kind": "openai",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "base_url": "https://openrouter.ai/api/v1",
        "secret": "OPENROUTER_API_KEY",
        "k": 6, "max_chars": 16000,
        "stream_timeout": 300,
    },
    # --- additional FREE model, via the existing OpenRouter key ----------------------
    # MiniMax M3. Added after GLM and Gemma were removed for upstream_provider_shared_pool
    # 429s, and it is worth stating why this one is not the same bet: GLM failed on one or
    # two tries, repeatedly, over hours; MiniMax answered on every check made today,
    # including one taken while nemotron_super was returning 502 from NVIDIA. It also won
    # the earlier free-model comparison on this corpus. Same shared-pool exposure in
    # principle -- it is still an OpenRouter :free endpoint -- so treat it as a second
    # option, not as the one to demo on.
    #
    # Verified free across every pricing field (prompt, completion, request) against
    # OpenRouter's live catalogue, and the window is 1,048,576 tokens -- four times
    # Nemotron Super's. max_chars = None because truncation is meaningless at that size:
    # the longest chunk in the corpus is 29,269 chars, so k=6 uncapped cannot approach it.
    "minimax": {
        "label": "MiniMax M3 — مجاني",
        "kind": "openai",
        "model": "minimax/minimax-m3:free",
        "base_url": "https://openrouter.ai/api/v1",
        "secret": "OPENROUTER_API_KEY",
        "k": 6, "max_chars": None,
        "stream_timeout": 300,
    },
    # --- DISABLED (kept for reference; strip the "# " to restore) ---
    # "nemotron_nano": {
        # "label": "Nemotron Nano 30B — مجاني (الأسرع)",
        # "kind": "openai",
        # "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        # "base_url": "https://openrouter.ai/api/v1",
        # "secret": "OPENROUTER_API_KEY",
        # "k": 6, "max_chars": None,
        # "stream_timeout": 300,
    # },
    # --- Arabic-native model: endpoint must be filled in before use ------------------
    # Fanar is the only one of the Arabic-native models kept here, because it is the only
    # one that needs neither a GPU nor a billing account: QCRI grants API keys free on
    # request at fanar.qa, and the API is OpenAI-compatible, so filling in base_url and
    # the secret below is the whole integration.
    #
    # ALLaM (watsonx/Azure) and Jais (Azure) were removed -- both are hosted but require a
    # paid cloud account -- and SILMA was removed as weights-only, needing a GPU this
    # machine does not have.
    #
    # base_url is deliberately left blank rather than guessed: a wrong one fails at request
    # time with an opaque network error instead of a message telling you what to fix.
    # Endpoint and model id read from QCRI's own OpenAPI document at api.fanar.qa, not
    # guessed: paths are /v1/chat/completions etc., so base_url carries the /v1. NOTE the
    # API's model id is NOT the HuggingFace repo name -- the card reads
    # "Fanar-2-27B-Instruct" but the API expects "Fanar-C-2-27B". Auth is a standard
    # Bearer token, which ChatOpenAI already sends. Rate limit: 50 requests/minute.
    # CONTEXT BUDGET -- why max_chars is far tighter here than for any other model.
    # Fanar's window is 16,000 tokens TOTAL; k=6 with uncapped documents sent 36,664 and
    # the API rejected the request outright with 413 too_large.
    #
    # The budget below is MEASURED, not estimated: Fanar's own /v1/tokens endpoint was
    # called with this app's real system prompt and four real corpus passages.
    #   SYSTEM_TEMPLATE   3,543 chars = 1,190 tokens   (2.98 chars/token)
    #   corpus passages                                 2.89-3.00 chars/token
    # Arabic legal text costs ~2.89 chars/token here -- roughly 20% more tokens per
    # character than the 3.50 measured for the OpenAI-family tokeniser on this corpus.
    #
    #   window                          16,000
    #   - system prompt                  1,190
    #   - answer headroom                2,500   (median answer is ~600 tokens)
    #   - slack                            200
    #   = documents                     12,110 tokens = ~34,900 chars at 2.89
    #   / k=6                           ~5,800 chars each
    #
    # k stays at 6 to match the Hit@6 retrieval benchmark -- the correct article often
    # sits at rank 3-6, so a smaller k costs recall directly. Truncating at 5,800 loses
    # every law reference in roughly 5% of judgments (measured), which is the unavoidable
    # cost of a 16k window; it is preferable to dropping a whole document.
    "fanar": {
        "label": "Fanar (QCRI) — عربي",
        "kind": "openai",
        "model": "Fanar-C-2-27B",
        "base_url": "https://api.fanar.qa/v1",
        "secret": "FANAR_API_KEY",
        # max_chars re-derived for k=8. Fanar's window is 16,000 tokens TOTAL and this
        # corpus costs ~2.89 chars/token (measured via Fanar's own /v1/tokens endpoint), so
        # 8 x 5,800 = 46,400 chars is ~16,000 tokens of context BEFORE the 1,190-token
        # system prompt -- rejected with 413 too_large. 8 x 4,300 = 34,400 chars ~= 11,900
        # tokens, which fits with the same headroom the previous k=6 budget had.
        "k": 6, "retrieval_k": 8, "max_chars": 4300,
    },
}
DEFAULT_LLM_PROVIDER = "openrouter"


def _build_llm(cfg):
    """Build one model's client from its own entry only.

    Every failure path raises RuntimeError, which both callers already handle: startup
    shows the message and leaves Search working, and switching model in الإعدادات shows it
    as a ⚠ message. So an unconfigured model never takes the app down.
    """
    api_key = _load_secret(cfg["secret"])
    if not api_key:
        raise RuntimeError(f"{cfg['secret']} missing — add it to .streamlit/secrets.toml")

    if cfg["kind"] == "groq":
        return ChatGroq(model=cfg["model"], temperature=0, api_key=api_key, streaming=True)

    if not cfg.get("base_url"):
        raise RuntimeError(
            f"{cfg['label']}: لم يتم ضبط عنوان الخدمة (base_url) لهذا النموذج بعد — "
            f"أضفه في LLM_PROVIDERS مع المفتاح {cfg['secret']}."
        )
    kwargs = dict(
        model=cfg["model"],
        temperature=0,
        api_key=api_key,
        base_url=cfg["base_url"],
        # Streaming does not make generation faster, but it removes the dead wait:
        # the lawyer sees the answer forming within ~2s instead of staring at a
        # spinner for the full response time. No accuracy cost.
        streaming=True,
    )
    # How long to wait for the FIRST streamed token before giving up. langchain-openai
    # defaults to 120s, which is too tight for a queued free endpoint: the logged
    # turns include successful answers taking up to 220.7s, so a request that would
    # have worked was being killed at 120 and surfaced to the user as an error.
    # Raising this does NOT make anything faster -- it stops a slow-but-alive request
    # being aborted. The cost is that a genuinely dead connection hangs longer.
    if cfg.get("stream_timeout"):
        kwargs["stream_chunk_timeout"] = cfg["stream_timeout"]
    return ChatOpenAI(**kwargs)


def build_qa_chain(vectordb, provider):
    cfg = LLM_PROVIDERS[provider]
    llm = _build_llm(cfg)

    memory = ConversationBufferWindowMemory(k=cfg["k"], memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=ThresholdMMRRetriever(vectordb=vectordb, k=cfg.get("retrieval_k", CHAT_RESULT_K),
                                        max_chars=cfg["max_chars"]),
        memory=memory,
        return_source_documents=True,
        # Kept in sync with the streaming path above, though chat does not go through
        # chain.invoke() -- this matters only for any code path that does.
        combine_docs_chain_kwargs={"prompt": prompt_for(provider)},
    )


@functools.lru_cache(maxsize=None)
def load_normalized_lookup(source):
    path_info = NORMALIZED_SOURCES.get(source)
    if not path_info:
        return {}
    path, id_key = path_info
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {rec.get(id_key): rec.get("normalized_text", "") for rec in records if rec.get(id_key)}


LLOC_SPLITS_PATH = str(PROCESSED_DIRECTORY / "document_splits_v2.json")


@functools.lru_cache(maxsize=1)
def load_lloc_article_lookup():
    if not Path(LLOC_SPLITS_PATH).exists():
        return {}
    with open(LLOC_SPLITS_PATH, encoding="utf-8") as f:
        records = json.load(f)
    lookup = {}
    for r in records:
        md = r.get("metadata", {})
        if md.get("source") != "lloc":
            continue
        lookup[(md.get("doc_id"), md.get("article_no"))] = r.get("page_content", "")
    return lookup


def full_text_for_doc(doc):
    source = doc.metadata.get("source", "")
    doc_id = doc.metadata.get("doc_id", "")
    if source in NORMALIZED_SOURCES:
        try:
            full_text = load_normalized_lookup(source).get(doc_id)
            if full_text:
                return full_text
        except MemoryError:
            pass
    if source == "lloc":
        full_text = load_lloc_article_lookup().get((doc_id, doc.metadata.get("article_no")))
        if full_text:
            return full_text
    return doc.page_content


def docs_to_sources(docs):
    """Retrieved documents -> source records carrying a stable 1-based source_id.

    That id is the single identity used everywhere downstream: it labels the chunk in the
    context, it is what the model cites as [n], and it numbers the sidebar entry. Nothing
    re-sorts or renumbers afterwards, so [3] always means the same chunk.
    """
    return [
        {
            "source_id": i,
            "source": doc.metadata.get("source", ""),
            "doc_id": doc.metadata.get("doc_id", ""),
            "article_no": doc.metadata.get("article_no", ""),
            "category": doc.metadata.get("category", ""),
            "text": doc.page_content,
            "cited": False,
        }
        for i, doc in enumerate(docs, start=1)
    ]


def docs_to_context(docs):
    """Label each chunk with its [n] and identity, so the model can cite by index.

    Citing by index removes an entire class of failure: the model no longer has to restate
    an article or appeal number in a form a regex can re-match. It copies a number it can
    see.
    """
    blocks = []
    for i, doc in enumerate(docs, start=1):
        md = doc.metadata
        head = (f"[{i}] المصدر={md.get('source','')} | الوثيقة={md.get('doc_id','')} "
                f"| المادة={md.get('article_no','') or '—'}")
        blocks.append(head + "\n" + doc.page_content)
    return "\n\n".join(blocks)


def _citation_scope(answer):
    """(scope, found_marker). found_marker decides how permissively the scope is read.

    Text after a real marker was written BY the model AS its source list, so an appeal
    number sitting next to its year is a citation there. The whole-answer fallback is
    ordinary prose, where the same two numbers mean nothing on their own.
    """
    idx = answer.find(CITED_SOURCES_MARKER)
    if idx == -1:
        return answer, False
    return answer[idx + len(CITED_SOURCES_MARKER):], True


_INDEX_REF = re.compile(r"\[(\d{1,2})\]")


def mark_cited(sources, answer):
    scope, in_citation_line = _citation_scope(answer)

    # Preferred path: the model cited by index, as the prompt asks. One integer identifies
    # one chunk exactly, so nothing has to be re-derived from prose.
    ids = {int(n) for n in _INDEX_REF.findall(scope) if 0 < int(n) <= len(sources)}
    if ids:
        for src in sources:
            src["cited"] = src.get("source_id") in ids
        return sources

    # Fallback: no [n] anywhere. A model that ignored the instruction should still get its
    # citations recognised rather than losing every source, so the article/appeal-number
    # matching below runs exactly as before.
    for src in sources:
        if src.get("source") == "lloc":
            article = src.get("article_no")
            src["cited"] = bool(article) and re.search(
                r"ماد[ةه]\s*\(?\s*" + re.escape(str(article)) + r"\s*\)?(?!\d)", scope
            ) is not None
        else:
            m = re.match(r"(\d+)\s+\S+\s+(\d{4})\s+K\s+(\d+)", src.get("doc_id") or "")
            if not m:
                src["cited"] = False
                continue
            appeal_no, year, rule_no = m.groups()
            cited = False
            for occ in re.finditer(re.escape(rule_no), scope):
                if "قاعدة" in scope[max(0, occ.start() - 30):occ.start() + 30]:
                    cited = True
                    break
            if not cited:
                for occ in re.finditer(re.escape(appeal_no), scope):
                    window = scope[max(0, occ.start() - 15):occ.start() + 40]
                    # Inside a real citation line الطعن is not required. Models copy the
                    # court's own reference format out of the judgment text -- e.g.
                    # "الطعن رقم 2/00001/2023/35" for a doc_id stored as "1 M 2023 K 00" -- and the
                    # 15-character lookbehind then misses الطعن by ONE character, dropping a
                    # citation the model made correctly and explicitly. In prose the word is
                    # still required, or any two matching numbers would mark a judgment.
                    if year in window and (in_citation_line or "الطعن" in window):
                        cited = True
                        break
            src["cited"] = cited
    return sources


SOURCE_LABELS_AR = {"lloc": "تشريع", "sjc": "حكم محكمة التمييز", "ccb": "حكم المحكمة الدستورية"}


def source_url(src):
    """Public URL for a source, when one genuinely exists.

    For legislation this is exact, not guessed: doc_id IS the lloc.gov.bh code
    (L1901 -> /Legislation/HTM/L1901/). The trailing slash matters -- without it the
    site returns 404. Judgments have no stable public URL, so they get none rather
    than a fabricated link.
    """
    if src.get("source") == "lloc" and src.get("doc_id"):
        return f"https://www.lloc.gov.bh/Legislation/HTM/{src['doc_id']}/"
    return None


def sources_to_elements(sources):
    labels = []
    # Legislation always precedes judgments in the numbered list, per the client's
    # request that the law itself -- not case law applying it -- comes first. Stable
    # sort: relative order within each group (retrieval order) is otherwise preserved.
    # NOT re-sorted and NOT renumbered. The previous version sorted cited sources
    # legislation-first and numbered them 1..n, so the panel's [1] was not necessarily the
    # model's [1] -- the two numbering schemes silently disagreed. Numbering now comes from
    # source_id, which is retrieval order and the same number the model was shown.
    cited = [s for s in sources if s.get("cited")]
    uncited = [s for s in sources if not s.get("cited")]

    def format_block(i, src, is_cited=True):
        article = src.get("article_no", "")
        article_part = f" — مادة {article}" if article else ""
        heading = f"[{i}] {src.get('source', '')} — {src.get('doc_id', '')}{article_part}"
        url = source_url(src)
        link_line = f"**الرابط:** [{url}]({url})  \n" if url else ""
        return (
            f"### {heading}\n\n"
            f"**النوع:** {SOURCE_LABELS_AR.get(src.get('source',''), src.get('source',''))}  \n"
            f"**رقم الوثيقة:** {src.get('doc_id', '')}  \n"
            f"**رقم المادة:** {src.get('article_no', '') or '—'}  \n"
            f"**التصنيف:** {src.get('category', '') or '—'}  \n"
            f"{link_line}"
            f"\n{src.get('text', '')}\n\n---\n"
        )

    sections = []

    cited_blocks = [format_block(src.get("source_id"), src, is_cited=True) for src in cited]
    if cited_blocks:
        labels.extend(f"[{src.get('source_id')}]" for src in cited)
        sections.append("## المصادر المستشهد بها في الاجابة\n\n" + "\n".join(cited_blocks))

    # Retrieved-but-uncited sources are DISPLAY-suppressed, not discarded. `uncited` is
    # still computed above and every one of those chunks is still retrieved, still sent to
    # the model, and still written to the session log with cited=False -- only the sidebar
    # section that listed them is gone, so the panel shows the cited sources alone.
    #
    # The cost of hiding them, worth remembering when debugging: an empty panel no longer
    # distinguishes "the model used nothing" from "the code failed to match what it used".
    # The log still tells them apart -- compare n_cited against len(sources) there.

    elements = []
    if sections:
        # NOT named "المصادر": Chainlit auto-links any exact occurrence of an element's
        # `name` found in the message text, turning it into a jump-to-panel link. The
        # model's own "المصادر_المستخدمة:" marker and our own content_suffix below both
        # contain that bare word, so naming the element "المصادر" split them into broken
        # mid-word links (visible in production: "[المصادر](#)_المستخدمة:"). This name
        # does not occur verbatim in either string, so nothing gets swallowed.
        elements.append(cl.Text(
            name="قائمة المصادر",
            content="\n\n".join(sections),
            display="side",
        ))

    # Deliberately contains "قائمة المصادر" verbatim -- Chainlit auto-links any exact
    # occurrence of an element's name to that element, so this is what makes the line
    # clickable and opens the side panel. The model's own "المصادر_المستخدمة:" marker
    # does not contain this longer phrase, so it is not affected by the same mechanism.
    content_suffix = ("\n\n**قائمة المصادر المستشهد بها في الاجابة:** " + " ".join(labels)) if labels else ""
    return elements, content_suffix


def build_export_text(messages):
    lines = ["# سجل المحادثة — المرجع (Al-Marja')", ""]
    turn = 0
    for msg in messages:
        if msg["role"] == "user":
            turn += 1
            lines.append(f"## سؤال {turn}")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")
        elif msg["role"] == "assistant":
            lines.append("### الاجابة")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")
            sources = msg.get("sources") or []
            if sources:
                lines.append("**المصادر:**")
                lines.append("")
                for i, src in enumerate(sources, start=1):
                    article = src.get("article_no", "")
                    article_part = f" — مادة {article}" if article else ""
                    lines.append(f"- [{i}] {src.get('source', '')} — {src.get('doc_id', '')}{article_part}")
                lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Attachments: text extraction.
#
# Two separate features, both driven off an attached file:
#   REVIEW_COMMAND_ID -- document + corpus retrieval ("does this comply?")
#   DOCQA_COMMAND_ID  -- document only, no corpus ("what does this say?")
#
# OCR is deliberately NOT attempted. Tesseract (both `ara` and tessdata_best)
# systematically corrupts Arabic-Indic numerals -- ٢٥ read as "79", 2002 as
# "3007" -- which in a legal document silently changes article numbers, dates
# and amounts. A scanned file is rejected with an explanation instead.
# ---------------------------------------------------------------------------

REVIEW_COMMAND_ID = "مراجعة مقابل القانون"
DOCQA_COMMAND_ID = "أسئلة عن المستند"

# Every question after the first is a follow-up by default: the app consults the last k
# exchanges and rewrites the question before searching (see the retrieval step below).
# This command lets the user override that for one question without touching مسح المحادثة
# -- memory is NOT cleared, so earlier context is still there and still exportable; this
# question just does not consult it. Deliberately does not clear memory, so it is not a
# substitute for مسح المحادثة -- that button erases accumulated context, this one skips it
# once.
FRESH_COMMAND_ID = "سؤال جديد"

# Moved out of the welcome message's action buttons and into the composer toolbar, next
# to بحث and the attachment icon, at the client's request.
#
# Chainlit commands are message *modifiers*, not one-shot buttons -- CommandDict has no
# "fire on click" flag. What makes this work anyway: the composer leaves its send button
# enabled when a command is selected with no text, and dispatches a command-only message
# (verified live against this app, not assumed). So these are handled ahead of the
# empty-query guard in on_message, and each runs on its own with nothing typed.
CLEAR_COMMAND_ID = "مسح المحادثة"
SOURCES_COMMAND_ID = "عرض كل المصادر"
EXPORT_COMMAND_ID = "تصدير المحادثة"

# Roughly 11k tokens of Arabic at the measured 3.50 chars/token. Leaves room for
# the retrieved corpus context and the answer inside a typical 32k window.
ATTACH_MAX_CHARS = 40000
SUPPORTED_ATTACH_EXT = (".pdf", ".docx", ".txt", ".md")

# Bidi controls and ZWNJ. During the corpus build these caused a law to be matched
# to the wrong decree number, so they are stripped here too.
_BIDI_MARKS = re.compile(r"[‌-‏‪-‮⁦-⁩]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def _normalize_extracted(text):
    """Repair the two things PDF text layers routinely get wrong.

    NFKC folds Arabic presentation forms (U+FB50-FDFF, U+FE70-FEFF) back to normal
    letters -- one sample file was 30% presentation forms, which the model would read
    as a different script entirely.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _BIDI_MARKS.sub("", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extraction_quality(text):
    """Return (single_char_ratio, word_like_ratio) for a rough legibility check."""
    tokens = text.split()
    if not tokens:
        return 1.0, 0.0
    single = sum(1 for t in tokens if len(t) == 1 and _LETTER.match(t))
    word_like = sum(1 for t in tokens if len(t) >= 2 and _LETTER.search(t))
    return single / len(tokens), word_like / len(tokens)


# Measured across the project's own sample documents. Readable files scored 0-8%
# single-char and 76-96% word-like; the two broken ones scored 86%/24% and 14%/43%.
# The thresholds sit in the gap, well clear of both groups.
MAX_SINGLE_CHAR_RATIO = 0.30
MIN_WORD_LIKE_RATIO = 0.55


def _extract_pdf(path):
    """Text layer of a PDF. Returns (text, note). No OCR -- see module comment."""
    import pymupdf

    doc = pymupdf.open(path)
    try:
        pages = [page.get_text() for page in doc]
        n_pages = doc.page_count
    finally:
        doc.close()

    text = "\n".join(pages).strip()
    # A born-digital page carries hundreds of characters; a scanned image carries
    # almost none. Below this the file has no usable text layer.
    if n_pages and len(text) < 50 * n_pages:
        return "", (
            f"الملف يبدو مصوراً (scanned) ولا يحتوي على نص قابل للاستخراج "
            f"({n_pages} صفحة، {len(text)} حرف فقط). "
            "استخراج النص من الصور غير مدعوم لأنه يشوّه الأرقام العربية في المستندات القانونية. "
            "الرجاء إرفاق نسخة نصية (PDF أصلي أو Word)."
        )
    return text, f"{n_pages} صفحة"


def _extract_docx(path):
    """Text of a .docx, including text boxes.

    python-docx walks paragraphs and tables only, so content inside
    <w:txbxContent> is invisible to it -- during the corpus build that silently
    returned 0 characters for 7 files and nearly lost an entire law. Walking the
    document part in document order picks up every <w:t> exactly once, text boxes
    included, while <w:p>/<w:br> supply the line breaks.
    """
    import zipfile

    from defusedxml import ElementTree as DET

    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = DET.fromstring(xml)

    parts = []
    for el in root.iter():
        if el.tag == f"{W}p" or el.tag == f"{W}br":
            parts.append("\n")
        elif el.tag == f"{W}tab":
            parts.append("\t")
        elif el.tag == f"{W}t":
            parts.append(el.text or "")
    text = re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()
    return text, f"{len(text):,} حرف"


def _extract_txt(path):
    raw = Path(path).read_bytes()
    encoding = "utf-8"
    try:
        import chardet

        detected = chardet.detect(raw).get("encoding")
        if detected:
            encoding = detected
    except Exception:
        pass
    text = raw.decode(encoding, errors="replace").strip()
    return text, f"{len(text):,} حرف"


def extract_document_text(path, name):
    """Return (text, note). `text` is empty when nothing usable could be read."""
    ext = Path(name).suffix.lower()
    try:
        if ext == ".pdf":
            text, note = _extract_pdf(path)
        elif ext == ".docx":
            text, note = _extract_docx(path)
        elif ext in (".txt", ".md"):
            text, note = _extract_txt(path)
        elif ext == ".doc":
            return "", (
                "صيغة .doc القديمة غير مدعومة. الرجاء حفظ الملف بصيغة .docx أو PDF ثم إعادة إرفاقه."
            )
        else:
            return "", (
                f"صيغة الملف ({ext or 'غير معروفة'}) غير مدعومة. "
                "الصيغ المدعومة: PDF، DOCX، TXT."
            )
    except Exception as e:
        return "", f"تعذر قراءة الملف: {type(e).__name__} — {str(e)[:150]}"

    if not text:
        return "", (note or "الملف فارغ أو لا يحتوي على نص قابل للاستخراج.")

    text = _normalize_extracted(text)

    # A PDF with a broken encoding map still yields "text" -- just unreadable text.
    # Passing it to the model would produce a confident answer about a document nobody
    # can actually read, which in a legal setting is worse than refusing.
    single_ratio, word_ratio = _extraction_quality(text)
    if single_ratio > MAX_SINGLE_CHAR_RATIO or word_ratio < MIN_WORD_LIKE_RATIO:
        return "", (
            "تعذّر استخراج نص سليم من هذا الملف — النص المستخرج مشوّه وغير قابل للقراءة "
            f"(كلمات مفهومة: {word_ratio:.0%}). "
            "غالباً ما يحدث هذا في ملفات PDF التي لا تتضمن خريطة ترميز صحيحة للخطوط العربية. "
            "الرجاء إرفاق نسخة Word أو نسخة PDF أخرى من المستند."
        )

    if len(text) > ATTACH_MAX_CHARS:
        text = text[:ATTACH_MAX_CHARS]
        note = f"{note} — تمت قراءة أول {ATTACH_MAX_CHARS:,} حرف فقط"
    return text, note


SEARCH_COMMAND_ID = "بحث"

# The two modes are exposed as Chat Profiles -- Chainlit's native equivalent of tabs -- so
# Search is a visible, first-class destination instead of a slash-command hidden in the
# composer. Each profile also shows only the settings that actually affect it (the
# "عدد النتائج" slider applies to Search only, which previously caused confusion).
PROFILE_CHAT = "المحادثة القانونية"
PROFILE_SEARCH = "البحث المباشر"


@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name=PROFILE_CHAT,
            markdown_description=(
                "اطرح سؤالاً قانونياً ويقوم النموذج اللغوي بصياغة إجابة منظمة "
                "(إجابة عامة، ثم القوانين ذات الصلة، ثم التطبيقات القضائية) مع ذكر المصادر."
            ),
        ),
        cl.ChatProfile(
            name=PROFILE_SEARCH,
            markdown_description=(
                "بحث مباشر في قاعدة البيانات القانونية **بدون نموذج لغوي** — "
                "نتائج فورية بالنصوص الأصلية، التشريعات أولاً ثم السوابق القضائية."
            ),
        ),
    ]


def build_document_actions():
    """Offered the moment a file is read, rather than living in the command bar.

    The client asked for these two to sit "inside the attachment icon". Chainlit's
    paperclip is the built-in spontaneous_file_upload -- an OS file picker with an
    accepted-MIME list and no API for attaching a menu to it, so there is no literal
    way to nest commands under it. Presenting the choice immediately after the upload
    is the closest equivalent: the question is asked at the moment the document exists,
    which is the point where the two features become meaningful at all.
    """
    return [
        cl.Action(name="doc_review", icon="scale", payload={}, label=REVIEW_COMMAND_ID),
        cl.Action(name="doc_qa", icon="file-text", payload={}, label=DOCQA_COMMAND_ID),
    ]


@cl.on_chat_start
async def start():
    profile = cl.user_session.get("chat_profile") or PROFILE_CHAT
    is_search = (profile == PROFILE_SEARCH)
    cl.user_session.set("mode", "search" if is_search else "chat")

    widgets = [
        MultiSelect(
            id="selected_sources",
            label="المصادر",
            initial=["lloc", "sjc", "ccb"],
            items={"التشريعات": "lloc", "السوابق القضائية": "sjc", "المحكمة الدستورية": "ccb"},
        ),
    ]
    # عدد النتائج is shown in BOTH profiles. Search is reachable from Chat too, via the
    # بحث command -- and while this slider was Search-profile-only, a search run that way
    # was permanently stuck at k_value=10 with no way to change it.
    widgets.append(Slider(id="k_value", label="عدد النتائج", initial=10, min=1, max=30, step=1))
    if not is_search:
        widgets.append(Select(
            id="llm_provider",
            label="مزود النموذج اللغوي",
            items={cfg["label"]: p for p, cfg in LLM_PROVIDERS.items()},
            initial=DEFAULT_LLM_PROVIDER,
        ))
    await cl.ChatSettings(widgets).send()
    cl.user_session.set("selected_sources", ["lloc", "sjc", "ccb"])
    cl.user_session.set("k_value", 10)
    cl.user_session.set("llm_provider", DEFAULT_LLM_PROVIDER)

    # The three toolbar actions below are registered in BOTH modes -- Search has a
    # transcript to clear and export just as Chat does, which is why they were on the
    # welcome message in both. عرض كل المصادر stays Chat-only: Search prints its sources
    # inline in every result, so there is no separate set to collect.
    #
    # The two attachment features are deliberately NOT here any more -- they are offered
    # as buttons the moment a file is read (build_document_actions).
    #
    # None of these are persistent: each is a one-off action that should clear itself
    # after firing rather than stay armed and re-fire on the next question.
    toolbar_commands = []
    if not is_search:
        # The slash-command stays available in Chat mode as a shortcut, but Search now has
        # its own profile so it no longer has to be discovered through the "/" menu.
        toolbar_commands += [
            {
                "id": SEARCH_COMMAND_ID,
                "description": "بحث مباشر في قاعدة البيانات القانونية بدون نموذج لغوي — نتائج فورية",
                "icon": "search",
                "button": True,
                "persistent": False,
            },
            {
                "id": FRESH_COMMAND_ID,
                "description": "تجاهل المحادثة السابقة عند البحث لهذا السؤال فقط -- المحادثة نفسها تبقى كما هي",
                "icon": "sparkles",
                "button": True,
                "persistent": False,
            },
        ]
    toolbar_commands.append({
        "id": CLEAR_COMMAND_ID,
        "description": "بدء سياق جديد -- يُمسح ما يتذكره النموذج، ويبقى نص المحادثة ظاهراً على الشاشة",
        "icon": "eraser",
        "button": True,
        "persistent": False,
    })
    if not is_search:
        toolbar_commands.append({
            "id": SOURCES_COMMAND_ID,
            "description": "عرض كل المصادر التي استُرجعت في هذه المحادثة، مرتبة حسب كل سؤال",
            "icon": "library",
            "button": True,
            "persistent": False,
        })
    toolbar_commands.append({
        "id": EXPORT_COMMAND_ID,
        "description": "تنزيل سجل المحادثة كاملاً بصيغة Markdown",
        "icon": "download",
        "button": True,
        "persistent": False,
    })
    await cl.context.emitter.set_commands(toolbar_commands)

    cl.user_session.set("history", [])

    try:
        vectordb = await cl.make_async(get_vectorstore)()
    except Exception as e:
        cl.user_session.set("vectordb_error", str(e))
        await cl.Message(content=f"تعذر تحميل قاعدة البيانات القانونية: {e}").send()
        return

    key_error = None
    if is_search:
        # Search never calls an LLM, so skip building the chain entirely: faster startup and
        # no spurious "API key missing" warning in a mode that does not need one.
        cl.user_session.set("qa_chain", None)
    else:
        try:
            qa_chain = build_qa_chain(vectordb, DEFAULT_LLM_PROVIDER)
            cl.user_session.set("qa_chain", qa_chain)
        except RuntimeError as e:
            key_error = str(e)
            cl.user_session.set("qa_chain", None)

    if is_search:
        welcome = (
            f"**{PROFILE_SEARCH}**\n\n"
            "اكتب كلمة أو عبارة قانونية (مثل *براءة الاختراع*) للبحث المباشر في قاعدة البيانات "
            "**بدون نموذج لغوي** — نتائج فورية بالنصوص الأصلية.\n\n"
            "تُعرض **التشريعات أولاً** ثم **السوابق القضائية**، مع رابط المصدر الرسمي لكل تشريع. "
            "استخدم لوحة الإعدادات (⚙) لتحديد المصادر وعدد النتائج."
        )
    else:
        welcome = (
            f"**{PROFILE_CHAT}**\n\n"
            "اكتب سؤالك القانوني مباشرة، وستحصل على إجابة منظمة: **إجابة عامة** موجزة، ثم "
            "**القوانين ذات الصلة**، ثم **التطبيقات القضائية** — مع ذكر المصادر وروابطها.\n\n"
            "للبحث المباشر في النصوص بدون نموذج لغوي، اختر **" + PROFILE_SEARCH + "** من أعلى الصفحة."
        )
    if key_error:
        welcome += (
            f"\n\n⚠️ {key_error} — المحادثة مع النموذج معطّلة حالياً، "
            "لكن **" + PROFILE_SEARCH + "** يعمل بدون الحاجة لمفتاح API."
        )
    # No action buttons here any more: مسح المحادثة / عرض كل المصادر / تصدير المحادثة now
    # live in the composer toolbar, so the welcome message is text only.
    await cl.Message(content=welcome).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("selected_sources", settings.get("selected_sources") or [])
    cl.user_session.set("k_value", int(settings.get("k_value") or 10))

    new_provider = settings.get("llm_provider") or DEFAULT_LLM_PROVIDER
    if new_provider != cl.user_session.get("llm_provider"):
        cl.user_session.set("llm_provider", new_provider)
        try:
            vectordb = await cl.make_async(get_vectorstore)()
            qa_chain = await cl.make_async(build_qa_chain)(vectordb, new_provider)
            cl.user_session.set("qa_chain", qa_chain)
            await cl.Message(content=f"تم التبديل إلى: {LLM_PROVIDERS[new_provider]['label']}").send()
        except RuntimeError as e:
            cl.user_session.set("qa_chain", None)
            await cl.Message(content=f"⚠️ {e}").send()


def _apply_source_guarantee(ranked, k_value, selected_sources):
    """Guarantee both legislation and judgments appear when both were selected.

    The failure this fixes: pure cosine ranking regularly fills all k_value slots with one
    source type -- "عقوبة السرقة" returns ten consecutive articles of L1576 and not one
    judgment -- so the client selected both sources and got only one back.

    Only applies when the user actually asked for both. Selecting a single source still
    returns that source alone, and if one type genuinely has no matches above the
    threshold, nothing is invented -- the results are returned as ranked.

    Relevance order is preserved: the guarantee decides WHICH documents make the cut, then
    the final set is re-sorted back into the retrieval order it came in.
    """
    selected = set(selected_sources)
    want_legislation = "lloc" in selected
    want_judgments = bool({"sjc", "ccb"} & selected)
    if not (want_legislation and want_judgments):
        return ranked[:k_value]

    legislation = [d for d in ranked if d.metadata.get("source") == "lloc"]
    judgments = [d for d in ranked if d.metadata.get("source") in ("sjc", "ccb")]
    if not legislation or not judgments:
        return ranked[:k_value]

    # Reserve a floor for each type rather than just one slot, so the scarcer type is
    # actually usable and not a single token result.
    reserve = max(1, k_value // 5)
    picked = legislation[: min(len(legislation), reserve)] + judgments[: min(len(judgments), reserve)]
    picked_ids = {id(d) for d in picked}
    for doc in ranked:                       # fill remaining slots in pure relevance order
        if len(picked) >= k_value:
            break
        if id(doc) not in picked_ids:
            picked.append(doc)
            picked_ids.add(id(doc))

    position = {id(d): i for i, d in enumerate(ranked)}
    return sorted(picked, key=lambda d: position[id(d)])[:k_value]


async def run_search(query: str):
    selected_sources = cl.user_session.get("selected_sources") or list(ALL_SOURCES)
    k_value = int(cl.user_session.get("k_value") or 10)

    if not selected_sources:
        await cl.Message(content="الرجاء اختيار مصدر واحد على الأقل من لوحة الإعدادات (⚙).").send()
        return 0

    try:
        vectordb = await cl.make_async(get_vectorstore)()
    except Exception as e:
        await cl.Message(content=f"تعذر الوصول لقاعدة البيانات القانونية: {e}").send()
        return 0

    # --- Change 2: fetch pool scales with k_value instead of a flat SEARCH_FETCH_POOL. ---
    fetch_pool = min(max(SEARCH_FETCH_POOL, k_value * SEARCH_FETCH_MULTIPLIER), SEARCH_FETCH_POOL_MAX)

    def _search():
        scored = vectordb.similarity_search_with_score(
            query, k=fetch_pool, filter=build_source_filter(selected_sources)
        )
        # --- Change 3: looser, Search-specific threshold (see module docstring). ---
        return [doc for doc, score in scored if score <= SEARCH_SCORE_THRESHOLD_BROWSE]

    raw_results = await cl.make_async(_search)()

    # --- Change 1: dedupe by (source, doc_id, article_no), not (source, doc_id) -- keeps
    # genuinely different articles of the same law as separate results, while still merging
    # true fragment-duplicates of the same article. ---
    seen = set()
    ranked = []
    for doc in raw_results:
        dedupe_key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("doc_id", ""),
            doc.metadata.get("article_no", ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ranked.append(doc)

    # NOTE: the truncation to k_value deliberately no longer happens in the loop above.
    # It has to come AFTER the source guarantee, or the guarantee would only ever see the
    # top k_value results -- which is exactly the set that was missing a whole source type.
    results = _apply_source_guarantee(ranked, k_value, selected_sources)

    if not results:
        await cl.Message(content="لا توجد نتائج مطابقة.").send()
        return 0

    # Group legislation first, then judgments -- a lawyer wants the governing text before
    # the cases applying it. Relevance order is preserved WITHIN each group.
    legislation = [d for d in results if d.metadata.get("source") == "lloc"]
    judgments = [d for d in results if d.metadata.get("source") in ("sjc", "ccb")]
    other = [d for d in results if d.metadata.get("source") not in ("lloc", "sjc", "ccb")]

    elements = []
    lines = [f"**{len(results)} نتيجة (بحث مباشر بدون نموذج لغوي):**", ""]
    idx = 0

    # --- attach each judgment to the law it applied -------------------------------
    # There is no metadata linking a judgment to a law (verified: every sjc record has
    # empty title/categories/article_no/section_heading), so the only available signal is
    # the decree citation inside the judgment's own text. Laws are keyed by the
    # (number, year) in their header, read from the retrieved laws themselves -- no corpus
    # scan needed. A judgment citing several of the retrieved laws is listed under each.
    law_key = {}
    for doc in legislation:
        did = doc.metadata.get("doc_id", "")
        if did and did not in law_key:
            m = LAW_CITATION.search(full_text_for_doc(doc)[:400])
            if m:
                law_key[did] = (int(m.group(1)), int(m.group(2)))

    cases_for, unattached = {}, []
    for doc in judgments:
        cites = {(int(a), int(b)) for a, b in LAW_CITATION.findall(full_text_for_doc(doc))}
        hits = [did for did, key in law_key.items() if key in cites]
        if hits:
            for did in hits:
                cases_for.setdefault(did, []).append(doc)
        else:
            # ~2/3 of judgments land here: they cite no decree number at all, so there is
            # no honest way to say which law they applied. Shown in their own block rather
            # than guessed into one.
            unattached.append(doc)

    def render_cases(group, indent="  "):
        """Renders a list of judgments as bullets. Used both under a law and standalone."""
        nonlocal idx
        for doc in group:
            idx += 1
            md = doc.metadata
            name = f"[{idx}] ({md.get('source','')}) {md.get('doc_id','')}"
            full_text = full_text_for_doc(doc)
            snippet = full_text[:160].strip()
            lines.append(f"{indent}- **{name}** — *{snippet}...*")
            elements.append(cl.Text(name=name, content=full_text, display="side"))

    def render_by_law(group, header):
        """Legislation: one entry per law, with that law's matching مواد underneath it.

        Several مواد of one law routinely match a single query -- "عقوبة السرقة" returns
        ten consecutive articles of L1576 -- and listing each as its own top-level hit
        filled the whole page with one law and pushed every other law out of the results.
        Grouping makes the law the unit the client scans while still showing every matched
        article, so nothing is dropped, only nested.

        Ordering is unchanged: laws appear in the order their best-ranked article appeared
        (dict preserves insertion order), and articles keep retrieval order within a law.
        """
        nonlocal idx
        if not group:
            return
        by_law = {}
        for doc in group:
            by_law.setdefault(doc.metadata.get("doc_id", ""), []).append(doc)
        lines.append(f"## {header} ({len(by_law)} قانون / {len(group)} مادة)")
        lines.append("")
        for doc_id, docs in by_law.items():
            idx += 1
            md = docs[0].metadata
            lines.append(f"**[{idx}] ({md.get('source','')}) {doc_id}** — {len(docs)} مادة")
            url = source_url({"source": md.get("source"), "doc_id": doc_id})
            if url:
                lines.append(f"[{url}]({url})")
            lines.append("")
            for doc in docs:
                article = doc.metadata.get("article_no", "")
                # doc_id is part of the name so two laws' "مادة 5" cannot collide -- the
                # name is what Chainlit matches on to make the side-panel link.
                name = f"مادة {article} ({doc_id})" if article else f"مقطع من {doc_id}"
                full_text = full_text_for_doc(doc)
                snippet = full_text[:160].strip()
                lines.append(f"- **{name}** — *{snippet}...*")
                elements.append(cl.Text(name=name, content=full_text, display="side"))
            # The law's own case law, immediately under it, before the next law.
            related = cases_for.get(doc_id) or []
            if related:
                lines.append("")
                lines.append(f"  **السوابق القضائية المتعلقة بـ {doc_id}:**")
                render_cases(related)
            lines.append("")

    def render_flat(group, header):
        """Judgments keep one entry each -- they are already one-per-judgment, because the
        dedupe key above collapses a judgment's sub-chunks (article_no is empty for sjc)."""
        nonlocal idx
        if not group:
            return
        lines.append(f"## {header} ({len(group)})")
        lines.append("")
        for doc in group:
            idx += 1
            md = doc.metadata
            doc_id = md.get("doc_id", "")
            article = md.get("article_no", "")
            article_part = f" — مادة {article}" if article else ""
            full_text = full_text_for_doc(doc)
            snippet = full_text[:200].strip()
            name = f"[{idx}] ({md.get('source','')}) {doc_id}{article_part}"
            lines.append(f"**{name}**")
            url = source_url({"source": md.get("source"), "doc_id": doc_id})
            if url:
                lines.append(f"[{url}]({url})")
            lines.append(f"*{snippet}...*")
            lines.append("")
            elements.append(cl.Text(name=name, content=full_text, display="side"))

    if legislation:
        # Interleaved view: each law, its مواد, then the cases that applied it.
        render_by_law(legislation, "التشريعات")
        if unattached:
            lines.append("## سوابق قضائية أخرى ذات صلة")
            lines.append("")
            render_cases(unattached, indent="")
            lines.append("")
    else:
        # Judgments-only result (or a judgments-only source selection): nothing to
        # interleave against, so they keep the plain flat listing.
        render_flat(judgments, "السوابق القضائية")
    render_flat(other, "مصادر أخرى")

    await cl.Message(content="\n".join(lines), elements=elements).send()
    return len(results)


async def ingest_attachments(message):
    """Read any files attached to this message into the session.

    Returns True if at least one file was read successfully. Extraction runs in a
    thread: a 200-page PDF takes seconds, and doing it inline would block the whole
    server for every connected user.
    """
    files = [e for e in (message.elements or []) if getattr(e, "path", None)]
    if not files:
        return False

    read_ok, notes = [], []
    for f in files:
        async with cl.Step(name=f"جارٍ قراءة {f.name}...", type="tool"):
            text, note = await cl.make_async(extract_document_text)(f.path, f.name)
        if text:
            read_ok.append((f.name, text))
            notes.append(f"- **{f.name}** — {note}")
        else:
            notes.append(f"- **{f.name}** — ⚠️ {note}")
        log_turn("attachment", filename=f.name, chars=len(text), note=note, ok=bool(text))

    if read_ok:
        # Several files are concatenated under labelled headers so the model can tell
        # them apart and name the right one in its answer.
        combined = "\n\n".join(f"=== {name} ===\n{text}" for name, text in read_ok)
        cl.user_session.set("doc_text", combined[:ATTACH_MAX_CHARS])
        cl.user_session.set("doc_name", "، ".join(name for name, _ in read_ok))
        # A new document re-asks which mode to use, rather than silently inheriting the
        # mode chosen for the previous one.
        cl.user_session.set("doc_mode", None)

    await cl.Message(content="**الملفات المرفقة:**\n" + "\n".join(notes)).send()
    return bool(read_ok)


async def answer_with_document(message, query, history, mode):
    """The two attachment features.

    mode "review" retrieves from the corpus and answers with both the document and the
    law; mode "docqa" answers from the document alone. They share this one function
    because everything except the prompt and the retrieval step is identical.
    """
    doc_text = cl.user_session.get("doc_text")
    doc_name = cl.user_session.get("doc_name") or "مستند"
    if not doc_text:
        await cl.Message(
            content=(
                f"لم يتم إرفاق أي مستند بعد. استخدم زر إرفاق الملفات (📎) لإرفاق ملف "
                f"(PDF أو DOCX أو TXT) ثم اطرح سؤالك.\n\n"
                f"للأسئلة القانونية العامة دون مستند، اترك الأمر واكتب سؤالك مباشرة."
            )
        ).send()
        return

    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain is None:
        await cl.Message(content="النموذج اللغوي غير متاح — تحقق من مفتاح API.").send()
        return

    t_start = time.time()
    sources, docs, error, answer = [], [], None, ""
    t_retrieval = None
    stream_msg = None

    try:
        if mode == "review":
            selected_sources = cl.user_session.get("selected_sources") or list(ALL_SOURCES)
            qa_chain.retriever.filter = build_source_filter(selected_sources)
            async with cl.Step(name="جارٍ البحث في المصادر...", type="retrieval"):
                # Retrieve against the question AND the document opening, so a bare
                # request like "راجع هذا العقد" still pulls relevant law rather than
                # matching on those two words alone.
                retrieval_query = f"{query}\n{doc_text[:1500]}"
                docs = await cl.make_async(qa_chain.retriever.invoke)(retrieval_query)
                t_retrieval = round(time.time() - t_start, 2)
            prompt_text = REVIEW_PROMPT.format(
                doc_name=doc_name,
                document=doc_text,
                context=docs_to_context(docs),
                question=query,
            )
        else:
            prompt_text = DOCQA_PROMPT.format(
                doc_name=doc_name,
                document=doc_text,
                question=query,
                review_command=REVIEW_COMMAND_ID,
            )

        stream_msg = cl.Message(content="")
        await stream_msg.send()

        llm = qa_chain.combine_docs_chain.llm_chain.llm
        for attempt in range(1, STREAM_MAX_ATTEMPTS + 1):
            try:
                answer = ""
                if attempt > 1:
                    stream_msg.content = ""
                    await stream_msg.update()
                async for chunk in llm.astream(prompt_text):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        answer += piece
                        await stream_msg.stream_token(piece)
                break
            except Exception as stream_err:
                msg = str(stream_err).lower()
                transient = any(m in msg for m in OVERLOAD_MARKERS) or "overload" in msg
                if attempt == STREAM_MAX_ATTEMPTS or not transient:
                    raise
                await asyncio.sleep(min(3 * (2 ** (attempt - 1)), 20))

        if mode == "review":
            sources = mark_cited(docs_to_sources(docs), answer)
    except Exception as e:
        error = str(e)[:300]
        if "rate_limit" in str(e).lower() or "429" in str(e):
            answer = (
                "تم تجاوز الحد المسموح به من عدد الطلبات لهذه الدقيقة او لهذا اليوم. "
                "الرجاء الانتظار قليلاً ثم إعادة المحاولة."
            )
        else:
            answer = f"حدث خطأ غير متوقع: {e}"

    log_turn(
        mode,
        question=query,
        answer=answer,
        seconds=round(time.time() - t_start, 2),
        retrieval_seconds=t_retrieval,
        doc_name=doc_name,
        doc_chars=len(doc_text),
        sources=sources_for_log(sources),
        error=error,
    )

    elements, content_suffix = sources_to_elements(sources) if sources else ([], "")
    if stream_msg is not None:
        stream_msg.content = answer + content_suffix
        stream_msg.elements = elements
        await stream_msg.update()
    else:
        await cl.Message(content=answer + content_suffix, elements=elements).send()

    history.append({"role": "assistant", "content": answer, "sources": sources})
    cl.user_session.set("history", history)


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()

    history = cl.user_session.get("history") or []

    # The three toolbar commands are handled before the empty-query guard below, because
    # they are sent with no text at all -- the whole message IS the command.
    if message.command == CLEAR_COMMAND_ID:
        await clear_conversation()
        return
    if message.command == SOURCES_COMMAND_ID:
        await show_sources()
        return
    if message.command == EXPORT_COMMAND_ID:
        await export_conversation()
        return

    # Attachments are ingested before anything else, so a file dropped in with no text
    # at all is still read and confirmed rather than silently discarded.
    had_attachment = await ingest_attachments(message)
    if not query:
        if had_attachment:
            # The two document features are offered here as buttons rather than as
            # commands the client has to know to pre-select -- see build_document_actions.
            await cl.Message(
                content="تمت قراءة المستند. ماذا تريد أن تفعل به؟",
                actions=build_document_actions(),
            ).send()
        return

    # Two ways in: the command form (kept working in case the two are ever restored to
    # the toolbar) and doc_mode, which is what the post-upload buttons set. doc_mode is
    # only honoured while a document is actually loaded, so it cannot silently swallow a
    # normal legal question after the document is gone.
    doc_mode = None
    if message.command == REVIEW_COMMAND_ID:
        doc_mode = "review"
    elif message.command == DOCQA_COMMAND_ID:
        doc_mode = "docqa"
    elif cl.user_session.get("doc_mode") and cl.user_session.get("doc_text"):
        doc_mode = cl.user_session.get("doc_mode")

    if doc_mode:
        label = REVIEW_COMMAND_ID if doc_mode == "review" else DOCQA_COMMAND_ID
        history.append({"role": "user", "content": f"[{label}] {query}"})
        cl.user_session.set("history", history)
        await answer_with_document(message, query, history, doc_mode)
        return

    # A file attached with no command at all defaults to review -- it is the feature
    # that uses the corpus, and answering a legal question without the law would be the
    # more surprising default. Only when no command was explicitly chosen -- if the user
    # picked "سؤال جديد" while attaching a file, that explicit choice must not be
    # silently overridden into review mode.
    if had_attachment and not message.command and cl.user_session.get("mode") != "search":
        history.append({"role": "user", "content": f"[{REVIEW_COMMAND_ID}] {query}"})
        cl.user_session.set("history", history)
        await answer_with_document(message, query, history, "review")
        return

    if message.command == SEARCH_COMMAND_ID or cl.user_session.get("mode") == "search":
        history.append({"role": "user", "content": f"[بحث] {query}"})
        t = time.time()
        n = await run_search(query)
        log_turn("search", question=query, seconds=round(time.time() - t, 2),
                 n_results=n, k_value=int(cl.user_session.get("k_value") or 10),
                 selected_sources=cl.user_session.get("selected_sources") or [])
        cl.user_session.set("history", history)
        return

    fresh = message.command == FRESH_COMMAND_ID
    history.append({"role": "user", "content": f"[{FRESH_COMMAND_ID}] {query}" if fresh else query})

    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain is None:
        provider = cl.user_session.get("llm_provider") or DEFAULT_LLM_PROVIDER
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
        answer = (
            f"{key_name} غير موجود — الرجاء إضافته الى .streamlit/secrets.toml "
            "(انظر .streamlit/secrets.toml.example) ثم إعادة تشغيل التطبيق. "
            "بامكانك استخدام أمر \"" + SEARCH_COMMAND_ID + "\" للبحث المباشر بدون نموذج لغوي في هذه الأثناء."
        )
        await cl.Message(content=answer).send()
        history.append({"role": "assistant", "content": answer, "sources": []})
        cl.user_session.set("history", history)
        return

    selected_sources = cl.user_session.get("selected_sources") or list(ALL_SOURCES)
    qa_chain.retriever.filter = build_source_filter(selected_sources)
    provider = cl.user_session.get("llm_provider") or DEFAULT_LLM_PROVIDER

    t_start = time.time()
    sources = []
    error = None
    answer = ""
    t_retrieval = None
    stream_msg = None
    standalone = query

    try:
        # Retrieval and generation are driven separately rather than through
        # chain.invoke() so that ONLY the final answer streams. ConversationalRetrievalChain
        # also calls the LLM to condense a follow-up question into a standalone one; streaming
        # that would print the rewritten question into the chat.
        async with cl.Step(name="جارٍ البحث في المصادر...", type="retrieval") as step:
            standalone = query
            # "سؤال جديد" skips consulting memory for retrieval on THIS question only --
            # it does not clear it. The answer is still saved into memory afterward
            # (below), so the next question -- without the command -- sees this turn as
            # part of its history again. مسح المحادثة is the one that actually erases it.
            history_pairs = (
                [] if fresh else
                (qa_chain.memory.load_memory_variables({}).get("chat_history") or [])
            )
            if history_pairs:
                standalone = await cl.make_async(_condense_question)(qa_chain, query, history_pairs)

            if fresh:
                step.output = (
                    "🆕 تم تجاهل المحادثة السابقة عند البحث لهذا السؤال فقط، بناءً على اختيارك. "
                    "المحادثة نفسها لم تُمسح، وستُستأنف تلقائياً مع السؤال التالي."
                )
            # Every question after the first is rewritten into a standalone question
            # before it searches the database, and that rewrite -- not what the user
            # typed -- decides what is retrieved. It used to be invisible, so a bad
            # answer caused by the rewrite pulling in an earlier topic was impossible
            # to diagnose from the chat. Shown only when it actually differs.
            elif standalone.strip() != query.strip():
                step.output = (
                    "أُعيدت صياغة سؤالك تلقائياً قبل البحث، بناءً على المحادثة السابقة. "
                    "هذه الصيغة هي التي بُحث بها فعلياً في قاعدة البيانات:\n\n"
                    f"> {standalone.strip()}\n\n"
                    "إذا كان هذا السؤال لا يعبّر عن قصدك، جرّب أمر \"" + FRESH_COMMAND_ID + "\" "
                    "للسؤال القادم، أو اضغط **مسح المحادثة** لمسح المحادثة كاملة."
                )

            docs = await cl.make_async(qa_chain.retriever.invoke)(standalone)
            t_retrieval = round(time.time() - t_start, 2)

        stream_msg = cl.Message(content="")
        await stream_msg.send()

        llm = qa_chain.combine_docs_chain.llm_chain.llm
        # Per-model prompt. This is the line that decides what the model actually sees --
        # chat streams through llm.astream() and never calls chain.invoke(), so the prompt
        # passed to build_qa_chain is not what generates the answer. Defaults to
        # QA_CHAIN_PROMPT, so every model without its own entry is unaffected.
        prompt_text = prompt_for(provider).format(
            context=docs_to_context(docs),
            question=query,
        )
        # The non-streaming path used _invoke_with_retry, which retries transient upstream
        # failures ("Service temporarily overloaded"). Streaming must retry too, otherwise a
        # blip that used to be invisible now reaches the user as an error message.
        for attempt in range(1, STREAM_MAX_ATTEMPTS + 1):
            try:
                answer = ""
                if attempt > 1:            # clear the partial text from the failed attempt
                    stream_msg.content = ""
                    await stream_msg.update()
                async for chunk in llm.astream(prompt_text):
                    piece = getattr(chunk, "content", "") or ""
                    if piece:
                        answer += piece
                        await stream_msg.stream_token(piece)
                break
            except Exception as stream_err:
                msg = str(stream_err).lower()
                transient = any(m in msg for m in OVERLOAD_MARKERS) or "overload" in msg
                if attempt == STREAM_MAX_ATTEMPTS or not transient:
                    raise
                await asyncio.sleep(min(3 * (2 ** (attempt - 1)), 20))

        sources = mark_cited(docs_to_sources(docs), answer)
        qa_chain.memory.save_context({"question": query}, {"answer": answer})
    except Exception as e:
        error = str(e)[:300]
        if "rate_limit" in str(e).lower() or "429" in str(e):
            # Two very different causes arrive as the same 429, and telling the client
            # "you used up your quota" when the real problem is a saturated shared
            # endpoint sends them to fix something that is not broken.
            #   upstream_provider_shared_pool -> the FREE model's provider is full for
            #       everyone on OpenRouter; a different key changes nothing, a different
            #       model does. Observed on z-ai/glm-5.2:free even for a 10-token request.
            #   otherwise -> this account's own per-minute / per-day allowance.
            low = str(e).lower()
            if "shared_pool" in low or "upstream_429" in low:
                answer = (
                    "هذا النموذج المجاني مزدحم حالياً لدى مزوّده، والحد مشترك بين جميع المستخدمين "
                    "وليس خاصاً بحسابك. اختر نموذجاً آخر من لوحة الإعدادات (⚙) أو أعد المحاولة لاحقاً."
                )
            else:
                answer = (
                    "تم تجاوز الحد المسموح به من عدد الطلبات لهذه الدقيقة او لهذا اليوم (حد مجاني من OpenRouter). "
                    "الرجاء الانتظار قليلاً ثم إعادة المحاولة."
                )
        else:
            answer = f"حدث خطأ غير متوقع: {e}"
    elapsed = round(time.time() - t_start, 2)

    log_turn(
        "chat",
        question=query,
        fresh=fresh,
        # The query actually sent to the retriever. Differs from `question` whenever the
        # follow-up rewriter fired, which is the usual cause of a puzzling answer.
        search_query=standalone if standalone.strip() != query.strip() else None,
        answer=answer,
        seconds=elapsed,
        retrieval_seconds=t_retrieval,
        provider=provider,
        selected_sources=selected_sources,
        sources=sources_for_log(sources),
        n_cited=sum(1 for s in sources if s.get("cited")),
        error=error,
    )

    elements, content_suffix = sources_to_elements(sources)
    if stream_msg is not None:
        # Update the message that was streamed rather than sending a second one, otherwise
        # the answer would appear twice.
        stream_msg.content = answer + content_suffix
        stream_msg.elements = elements
        await stream_msg.update()
    else:
        await cl.Message(content=answer + content_suffix, elements=elements).send()

    history.append({"role": "assistant", "content": answer, "sources": sources})
    cl.user_session.set("history", history)


@cl.action_callback("doc_review")
async def doc_review_action(action: cl.Action):
    """Arm مراجعة مقابل القانون for the attached document.

    Set as a session mode rather than run immediately, because both features need a
    question to answer -- and it stays armed across turns, matching the `persistent:
    True` behaviour the two commands had before they moved out of the toolbar.
    """
    cl.user_session.set("doc_mode", "review")
    await cl.Message(
        content=f"**{REVIEW_COMMAND_ID}** — اطرح سؤالك عن المستند وسيُراجَع في ضوء التشريعات والسوابق البحرينية."
    ).send()


@cl.action_callback("doc_qa")
async def doc_qa_action(action: cl.Action):
    cl.user_session.set("doc_mode", "docqa")
    await cl.Message(
        content=f"**{DOCQA_COMMAND_ID}** — اطرح سؤالك عن محتوى المستند وحده، دون الرجوع لقاعدة البيانات."
    ).send()


async def clear_conversation():
    """Break the conversational context without erasing the visible transcript.

    Deliberate: the messages stay on screen so the client can still read what was
    asked and answered. What is reset is the chain's memory, so the next question
    starts fresh with no carried-over context and no follow-up resolution against
    earlier turns.

    `history` is deliberately NOT cleared. It backs تصدير المحادثة and عرض كل المصادر,
    so clearing it would leave the client looking at a full transcript that the export
    button reports as empty.
    """
    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain is not None:
        qa_chain.memory.clear()
    # Releases the document mode too, so this doubles as the way out of مراجعة مقابل
    # القانون / أسئلة عن المستند. The document itself (doc_text) is deliberately kept --
    # only the routing flag is cleared, so nothing the client uploaded is thrown away.
    cl.user_session.set("doc_mode", None)
    await cl.Message(content="تم مسح المحادثة — يمكنك البدء بسؤال جديد.").send()


async def show_sources():
    history = cl.user_session.get("history") or []
    turns = [
        (i, msg) for i, msg in enumerate(history)
        if msg["role"] == "assistant" and msg.get("sources")
    ]
    if not turns:
        await cl.Message(content="لا توجد مصادر بعد — اطرح سؤالاً اولاً.").send()
        return

    all_elements = []
    lines = ["**المصادر المسترجعة لكل سؤال:**", ""]
    for i, msg in turns:
        question_text = history[i - 1]["content"] if i > 0 and history[i - 1]["role"] == "user" else ""
        lines.append(f"### السؤال: {question_text}" if question_text else "### سؤال")
        elements, content_suffix = sources_to_elements(msg["sources"])
        lines.append(content_suffix.strip() or "_(لا توجد مصادر مستشهد بها)_")
        lines.append("")
        all_elements.extend(elements)

    await cl.Message(content="\n".join(lines), elements=all_elements).send()


async def export_conversation():
    history = cl.user_session.get("history") or []
    if not history:
        await cl.Message(content="لا توجد محادثة لتصديرها بعد.").send()
        return
    export_text = build_export_text(history)
    file_element = cl.File(
        name="capital_legal_base_conversation_search_v2.md",
        content=export_text.encode("utf-8"),
        mime="text/markdown",
        display="inline",
    )
    await cl.Message(content="سجل المحادثة جاهز للتنزيل:", elements=[file_element]).send()
