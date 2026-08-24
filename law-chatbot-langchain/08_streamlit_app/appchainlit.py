"""Capital Legal Base — Chainlit port of app.py.

PARALLEL file: app.py (the deployed Streamlit app) is never modified, imported,
or executed by this module. Every piece of backend/RAG logic below (system
prompt, ConversationalRetrievalChain construction, the sjc/ccb/lloc full-text
lookup helpers, docs_to_sources) is a verbatim, line-for-line port of app.py's
logic. Only the UI-glue layer differs, because Chainlit has no tabs/widgets
API of its own — it was re-derived against Chainlit's actual current API
(2.x, community-maintained since May 2025), cross-checked against both
https://docs.chainlit.io and the source of the installed chainlit==2.11.1
package (chainlit/{message,element,chat_settings,input_widget,callbacks,
emitter}.py) before writing any of this file. See the final report for the
exact doc URLs and package-source lines used to confirm each API below.

Chainlit surfaces used here:
  - @cl.on_chat_start / @cl.on_message      lifecycle hooks (chainlit/callbacks.py)
  - cl.ChatSettings + chainlit.input_widget  settings panel: source filter + k
  - @cl.on_settings_update                   settings-change handler
  - cl.Text(..., display="side") elements    citations attached to a cl.Message
                                              (chainlit/element.py: Text/Element)
  - cl.context.emitter.set_commands([...])   registers a composer "/" command
    + Message.command                        (chainlit/types.py: CommandDict;
                                              chainlit/message.py: Message.command)
                                              used here for a "بحث" (Search) mode
  - cl.Action + @cl.action_callback          clear / export buttons
  - cl.File element                          conversation export download
  - cl.user_session                          per-session chain + memory + settings
  - cl.make_async                            runs the blocking embedding-model
                                              load off the event loop

Confirmed NOT needed:
  - No custom copy-to-clipboard button. The bundled frontend
    (chainlit/frontend/dist/assets/index-*.js) already ships a
    "copyToClipboard" action on every assistant message — verified by
    grepping the built JS bundle directly, since minified bundles aren't
    reliably covered by the docs. Building a duplicate would be redundant.
"""

import functools
import json
import os
import re
import tomllib
from pathlib import Path
from threading import Lock
from typing import List, Optional

import chainlit as cl
import numpy as np
from chainlit.input_widget import MultiSelect, Slider
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

APP_DIR = Path(__file__).parent
PERSIST_DIRECTORY = str(APP_DIR.parent / "data" / "chroma")
PROCESSED_DIRECTORY = APP_DIR.parent / "data" / "processed"
SECRETS_TOML_PATH = APP_DIR / ".streamlit" / "secrets.toml"

# ---------------------------------------------------------------------------
# Secrets. Chainlit's own convention is a .env file, auto-loaded into
# os.environ by python-dotenv the moment `chainlit` is imported (see
# chainlit/__init__.py: `load_dotenv(dotenv_path=os.path.join(os.getcwd(),
# os.getenv("CHAINLIT_ENV_FILE", ".env")))`) — it has no st.secrets
# equivalent. Rather than making the user duplicate OPENROUTER_API_KEY in a
# second file, read the SAME .streamlit/secrets.toml app.py already uses,
# with the stdlib tomllib (no new dependency, and this is plain file I/O —
# it doesn't fight or bypass Chainlit's own .env loading, which still runs
# and still populates os.environ normally). os.environ is checked second,
# mirroring app.py's own `st.secrets.get(...) or os.environ.get(...)` order,
# so a real .env would still work if someone adds one later.
# ---------------------------------------------------------------------------


def _load_openrouter_key() -> str:
    if SECRETS_TOML_PATH.exists():
        try:
            with open(SECRETS_TOML_PATH, "rb") as f:
                data = tomllib.load(f)
            key = data.get("OPENROUTER_API_KEY")
            if key:
                return key
        except Exception:
            pass
    return os.environ.get("OPENROUTER_API_KEY", "")


# A minority of sjc/ccb cases (~6% of sjc, ~28% of ccb) are split into multiple disconnected
# chunk fragments by a token-length fallback splitter, purely because the ruling text was too
# long — not a natural legal boundary (most cases are already a single whole-case chunk by
# design; see 03_document_splitting.ipynb). These *_normalized.json files hold the full,
# un-fragmented case/ruling text per doc_id, so Search can show the complete judgment instead of
# a floating fragment for that minority. lloc chunks are already whole articles (chunked per
# article_no), so no lookup is needed for that source.
NORMALIZED_SOURCES = {
    "sjc": (str(PROCESSED_DIRECTORY / "sjc_normalized.json"), "key"),
    "ccb": (str(PROCESSED_DIRECTORY / "ccb_normalized.json"), "case_id"),
}

ALL_SOURCES = {"lloc", "sjc", "ccb"}
SOURCE_LABELS = {"lloc": "التشريعات", "sjc": "السوابق القضائية", "ccb": "المحكمة الدستورية"}

# Chroma's default distance metric (L2) — lower is more similar, no fixed 0-1 range. Recalibrated
# against this vectorstore across several real legal terms (not just one): genuine matches range
# 0.65-0.86 (e.g. a case whose headnote literally quotes "التقادم المسقط" for that exact query
# scores 0.8592), while off-topic queries (cooking, football, unrelated topics) score 1.04-1.36 —
# still a clean gap, just narrower than first thought. 0.85 (the original value) sat inside the
# real-match range and silently discarded correct results — including an exact literal-phrase
# match. 0.95 sits in the actual gap. Important limit: this does NOT fix precision within the legal
# domain — a real but wrong-topic match (confirmed case: a housing-subsidy article wrongly surfaced
# for a spousal-maintenance question) scored 0.7267, BETTER than some genuinely relevant results in
# the same query, so no threshold can separate it. This only filters out results with no real
# relevance to the query at all, which Search has no other safety net against (unlike Chat, where
# the citation-discipline prompt is the actual guard for that case).
SEARCH_SCORE_THRESHOLD = 0.95

# Chroma's HNSW index has poor recall at small result counts on this collection: an unfiltered
# query for "الفصل التعسفي" with n_results=50 returned only lloc chunks (0 sjc, 0 ccb), even
# though a much better-scoring sjc match existed (0.7952, vs 0.86+ for the lloc results actually
# returned) — confirmed directly against this vectorstore. That match only appears once n_results
# is raised to ~200+. This isn't a filter-logic bug (per-source filtered queries return correct
# results) or a threshold bug (the missing result would have passed 0.85 easily) — it's the ANN
# search itself failing to explore the full collection at low k. Always fetching a large candidate
# pool before filtering/slicing works around it regardless of root cause inside Chroma/HNSW.
SEARCH_FETCH_POOL = 300


class ThresholdMMRRetriever(BaseRetriever):
    """Combines a relevance-score cutoff with MMR diversity for Chat — LangChain's built-in
    retriever only supports one or the other (search_type is either "mmr" or
    "similarity_score_threshold", not both at once). Same threshold and same honest limits as
    SEARCH_SCORE_THRESHOLD above (filters out queries with no real answer in the corpus at all;
    does not fix a real-but-wrong-topic match, since those can score just as well as a correct
    one — confirmed directly against this vectorstore). Fetches a larger candidate pool, drops
    anything past the threshold, then runs LangChain's own MMR selection on the survivors."""

    vectordb: Chroma
    k: int = 6
    fetch_k: int = SEARCH_FETCH_POOL
    score_threshold: float = SEARCH_SCORE_THRESHOLD
    lambda_mult: float = 0.5
    filter: Optional[dict] = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        query_vec = self.vectordb.embeddings.embed_query(query)
        raw = self.vectordb._collection.query(
            query_embeddings=[query_vec],
            n_results=self.fetch_k,
            include=["documents", "metadatas", "distances", "embeddings"],
            where=self.filter,
        )
        docs, metas, dists, embs = raw["documents"][0], raw["metadatas"][0], raw["distances"][0], raw["embeddings"][0]
        kept = [(d, m, e) for d, m, dist, e in zip(docs, metas, dists, embs) if dist <= self.score_threshold]
        if not kept:
            return []
        kept_embs = [e for _, _, e in kept]
        idxs = maximal_marginal_relevance(np.array(query_vec), kept_embs, lambda_mult=self.lambda_mult, k=min(self.k, len(kept)))
        return [Document(page_content=kept[i][0], metadata=kept[i][1]) for i in idxs]


def build_source_filter(selected_sources):
    """Build a Chroma metadata filter from a list of selected source codes. Returns None when
    all sources (or none specified) are selected, since that's equivalent to no filter. Shared
    helper: used by both Search and Chat's retriever (see on_message below)."""
    selected = set(selected_sources)
    if not selected or selected == ALL_SOURCES:
        return None
    if len(selected) == 1:
        return {"source": next(iter(selected))}
    return {"source": {"$in": sorted(selected)}}


# Same citation-discipline prompt validated in 06_question_answering / 07_chat.
SYSTEM_TEMPLATE = """انت مساعد قانوني متخصص في القانون البحريني. استخدم المقاطع القانونية التالية فقط للاجابة على السؤال في نهاية النص.

قواعد صارمة يجب اتباعها:
- استند فقط الى النصوص المرفقة، ولا تخترع اي معلومة غير موجودة فيها.
- اذا لم تكن الاجابة موجودة في النصوص المرفقة، صرح بذلك بوضوح ولا تخمن.
- اذكر المصدر الدقيق لكل معلومة (رقم المادة او رقم القضية)، وانقل اسم القانون او القرار كما هو مكتوب حرفياً في النص المرفق فقط. لا تنسب اي معلومة الى اسم قانون لم يرد ذكره صراحة في النص المرفق.
- قبل استخدام اي مقطع، تحقق ان موضوعه يتعلق فعلاً بموضوع السؤال. اذا كان المقطع من مجال قانوني مختلف (مثل قرار اسكاني او اداري لا علاقة له بالسؤال) فلا تستخدمه ولا تذكره في الاجابة، حتى لو تشابهت بعض الكلمات.
- اختم اجابتك دائماً بسطر منفصل يبدأ حرفياً بـ "المصادر_المستخدمة:" يليه فقط ارقام المواد و/او ارقام القضايا التي استندت اليها فعلاً في متن الاجابة (وليس تلك التي استبعدتها او ذكرتها لتوضيح عدم صلتها)، مفصولة بفواصل. اذا لم تستند الى اي نص مرفق فعلاً (سؤال غير متعلق بالقانون، او لا توجد اجابة في النصوص المرفقة)، اكتب "المصادر_المستخدمة: لا يوجد".

النصوص القانونية:
{context}

السؤال: {question}

الاجابة القانونية المدعومة بالمصادر:"""

CITED_SOURCES_MARKER = "المصادر_المستخدمة:"

QA_CHAIN_PROMPT = PromptTemplate.from_template(SYSTEM_TEMPLATE)


# ---------------------------------------------------------------------------
# Shared, expensive resources — module-level singletons (Chainlit's
# equivalent of st.cache_resource; there is no per-app cache decorator, so a
# plain lazily-initialized module global guarded by a lock is the standard
# pattern). Loaded once per worker process and reused by every session.
# ---------------------------------------------------------------------------
_vectordb = None
_vectordb_lock = Lock()


def get_vectorstore():
    """Load (once) and return the shared Chroma vectorstore. Synchronous/blocking —
    call through cl.make_async from async code so it doesn't block the event loop."""
    global _vectordb
    if _vectordb is not None:
        return _vectordb
    with _vectordb_lock:
        if _vectordb is None:
            if not Path(PERSIST_DIRECTORY).exists():
                raise FileNotFoundError(
                    f"Vectorstore not found at {PERSIST_DIRECTORY} — unzip the Drive-backed vectorstore there first."
                )
            embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
            _vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
        return _vectordb


def build_qa_chain(vectordb, openrouter_api_key):
    """Build a fresh ConversationalRetrievalChain + memory for one session. Unlike the shared
    vectorstore, this is cheap to construct (no model loading) and MUST be per-session — the
    memory object is per-conversation state, so it lives in cl.user_session, not as a module
    singleton."""
    # OpenRouter is OpenAI-API-compatible, so ChatOpenAI works against it with a custom base_url.
    llm = ChatOpenAI(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        temperature=0,
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    # k=6 for both retriever and memory. Groq's free tier forced k=2/k=2 because its limit was
    # tokens-per-minute (8,000) — a big constraint on how much text one request could carry. This
    # model's free tier limits requests-per-minute/day (20 RPM, 200 RPD) instead, and its 1M-token
    # context window comfortably absorbs far more retrieved context and conversation history than
    # Groq ever could — the old per-request token ceiling that drove k down to 2 doesn't apply here.
    # Kept at 6, not higher, since more retrieved chunks isn't free of cost to answer quality if the
    # extra context is less relevant — this is a deliberate middle ground, not the max possible.
    memory = ConversationBufferWindowMemory(k=6, memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=ThresholdMMRRetriever(vectordb=vectordb, k=6),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    )


@functools.lru_cache(maxsize=None)
def load_normalized_lookup(source):
    """Load a *_normalized.json file once and build a {doc_id: full_text} dict for O(1) lookup.
    Only applies to sjc/ccb — see NORMALIZED_SOURCES comment above. lru_cache stands in for
    st.cache_data here (module-level, process-lifetime, keyed by argument)."""
    path_info = NORMALIZED_SOURCES.get(source)
    if not path_info:
        return {}
    path, id_key = path_info
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return {rec.get(id_key): rec.get("normalized_text", "") for rec in records if rec.get(id_key)}


# For sjc/ccb, "the full document" is well-defined — one whole case, so a doc_id-keyed lookup can
# fall back to it safely. For lloc there's no equivalent: the raw source (lloc_normalized.json) is
# keyed by law, not by article, and its "full text" is the ENTIRE law (all of its articles) — falling
# back to that for a single article would replace a possibly-incomplete article with a wall of
# unrelated ones, which is worse, not safer. document_splits_v2.json already has every lloc chunk
# correctly split and labeled with its own article_no, so THAT is the right source for a lloc
# fallback: keyed by (doc_id, article_no), not doc_id alone.
LLOC_SPLITS_PATH = str(PROCESSED_DIRECTORY / "document_splits_v2.json")


@functools.lru_cache(maxsize=1)
def load_lloc_article_lookup():
    """{(doc_id, article_no): full_article_text} for lloc, built from the corrected v2 splitting
    output. A lloc chunk in the vectorstore is meant to already BE exactly one correctly-bounded
    article, so this is a safety net for the rare chunk that isn't."""
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
    """Return the complete case/article text to display for a Search result. For sjc/ccb, look up
    the full un-fragmented case by doc_id. For lloc, look up the correctly-bounded article by
    (doc_id, article_no). Falls back to the chunk's own text if no lookup match is found either way."""
    source = doc.metadata.get("source", "")
    doc_id = doc.metadata.get("doc_id", "")
    if source in NORMALIZED_SOURCES:
        full_text = load_normalized_lookup(source).get(doc_id)
        if full_text:
            return full_text
    if source == "lloc":
        full_text = load_lloc_article_lookup().get((doc_id, doc.metadata.get("article_no")))
        if full_text:
            return full_text
    return doc.page_content


def docs_to_sources(docs):
    """Build small serializable source dicts straight from retrieved Documents (not from LLM
    self-citation, which is unreliable — confirmed in prior testing)."""
    return [
        {
            "source": doc.metadata.get("source", ""),
            "doc_id": doc.metadata.get("doc_id", ""),
            "article_no": doc.metadata.get("article_no", ""),
            "category": doc.metadata.get("category", ""),
            "text": doc.page_content,
        }
        for doc in docs
    ]


def _citation_scope(answer):
    """Isolate the text to check for citation matches. The prompt requires every answer to end
    with a "المصادر_المستخدمة:" line listing ONLY the sources actually relied on — scanning just
    that line (instead of the whole answer) avoids false positives from the model naming an
    article elsewhere in the prose purely to explain why it's NOT relevant (e.g. "مادة 6 ... لا
    علاقة لها بالسؤال"), which the model does as good, transparent behavior but which a whole-text
    scan can't distinguish from an actual citation. Falls back to the full answer for older
    messages saved before this line existed, or if the model ever drops it."""
    idx = answer.find(CITED_SOURCES_MARKER)
    if idx == -1:
        return answer
    return answer[idx + len(CITED_SOURCES_MARKER):]


def mark_cited(sources, answer):
    """Flag which retrieved sources the model actually relied on, by matching article/case
    numbers against the citation scope (see _citation_scope) — not by trusting free-form
    self-citation (already confirmed unreliable elsewhere). The retriever routinely pulls in
    topically-adjacent-but-unused chunks (see SEARCH_SCORE_THRESHOLD comment); without this flag
    the source list presents all of them as if they backed the answer, which is misleading."""
    scope = _citation_scope(answer)
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
                    if "الطعن" in window and year in window:
                        cited = True
                        break
            src["cited"] = cited
    return sources


def sources_to_elements(sources):
    """Render a numbered source list as Chainlit citation elements: one cl.Text per source,
    display="side" so clicking the citation chip opens the full text in the side panel — this is
    Chainlit's actual mechanism for what app.py did with st.expander(...) + st.text_area(...).
    Only sources the model actually relied on (per mark_cited) are numbered and included in the
    content_suffix label line; retrieved-but-unused sources are still attached as elements (so
    nothing is silently hidden if the citation-matching heuristic misses a real reference) but
    are not presented as if they backed the answer.
    Returns (elements, content_suffix) where content_suffix is a "**المصادر:** [1] [2] ..." line
    to append to the message content, same convention as app.py's render_sources_list labels."""
    elements = []
    labels = []
    cited = [s for s in sources if s.get("cited")]
    uncited = [s for s in sources if not s.get("cited")]

    for i, src in enumerate(cited, start=1):
        article = src.get("article_no", "")
        article_part = f" — مادة {article}" if article else ""
        name = f"[{i}] {src.get('source', '')} — {src.get('doc_id', '')}{article_part}"
        labels.append(f"[{i}]")
        body = (
            f"المصدر: {src.get('source', '')}\n"
            f"رقم الوثيقة: {src.get('doc_id', '')}\n"
            f"رقم المادة: {src.get('article_no', '') or '—'}\n"
            f"التصنيف: {src.get('category', '') or '—'}\n\n"
            f"{src.get('text', '')}"
        )
        elements.append(cl.Text(name=name, content=body, display="side"))

    for j, src in enumerate(uncited, start=1):
        article = src.get("article_no", "")
        article_part = f" — مادة {article}" if article else ""
        name = f"(غير مستشهد به) {src.get('source', '')} — {src.get('doc_id', '')}{article_part}"
        body = (
            f"مصدر تم استرجاعه ولم يُستشهد به صراحة في الاجابة.\n\n"
            f"المصدر: {src.get('source', '')}\n"
            f"رقم الوثيقة: {src.get('doc_id', '')}\n"
            f"رقم المادة: {src.get('article_no', '') or '—'}\n"
            f"التصنيف: {src.get('category', '') or '—'}\n\n"
            f"{src.get('text', '')}"
        )
        elements.append(cl.Text(name=name, content=body, display="side"))

    content_suffix = ("\n\n**المصادر المستشهد بها في الاجابة:** " + " ".join(labels)) if labels else ""
    return elements, content_suffix


def build_export_text(messages):
    lines = ["# سجل المحادثة — Capital Legal Base", ""]
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
# The "بحث" (Search) command — Chainlit is fundamentally a single-thread chat
# UI with no tabs, so app.py's separate "البحث" tab (direct
# vectordb.similarity_search, no LLM call) needed a real equivalent. Chainlit
# 2.x's composer "/" command palette (cl.context.emitter.set_commands +
# message.command, chainlit/types.py CommandDict, message.py Message.command)
# is the closest native fit: the user picks "/بحث" once and every message
# sent while it's active is routed to plain vector search instead of the LLM
# chain — same low-friction, no-tab-switching feel as app.py's tab, and it
# doesn't require inventing a slash-command parser of our own. A cl.Action
# button was considered but rejected: actions fire a fixed payload on click,
# they don't cleanly accept new free-text queries turn after turn the way a
# composer command does.
# ---------------------------------------------------------------------------
SEARCH_COMMAND_ID = "بحث"


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            MultiSelect(
                id="selected_sources",
                label="المصادر",
                initial=["lloc", "sjc", "ccb"],
                # items (not values): lets each option show its Arabic label while the value
                # sent back to on_settings_update stays the short source code (lloc/sjc/ccb).
                # NOTE: MultiSelect.__post_init__ raises ValueError if both `values` and `items`
                # are given — confirmed live (Chainlit surfaced this exact pydantic validation
                # error in the UI on first run) and by reading chainlit/input_widget.py directly.
                items={"التشريعات": "lloc", "السوابق القضائية": "sjc", "المحكمة الدستورية": "ccb"},
            ),
            Slider(id="k_value", label="عدد النتائج", initial=10, min=1, max=30, step=1),
        ]
    ).send()
    cl.user_session.set("selected_sources", ["lloc", "sjc", "ccb"])
    cl.user_session.set("k_value", 10)

    await cl.context.emitter.set_commands(
        [
            {
                "id": SEARCH_COMMAND_ID,
                "description": "بحث مباشر في قاعدة البيانات القانونية بدون نموذج لغوي — نتائج فورية",
                "icon": "search",
                "button": True,
                "persistent": False,
            }
        ]
    )

    cl.user_session.set("history", [])

    # Vectorstore is required for both Chat and Search; load it eagerly (off the event loop) so
    # a missing/corrupt vectorstore fails fast and visibly instead of on the user's first query.
    try:
        vectordb = await cl.make_async(get_vectorstore)()
    except Exception as e:
        cl.user_session.set("vectordb_error", str(e))
        await cl.Message(content=f"تعذر تحميل قاعدة البيانات القانونية: {e}").send()
        return

    # Unlike app.py — which calls st.stop() and halts the ENTIRE app (Search tab included) when
    # OPENROUTER_API_KEY is missing — Search here has no dependency on the LLM/API key at all, so
    # it must keep working even without one. The QA chain is only built if a key is present; if
    # not, `qa_chain` stays None and on_message shows a clear Arabic error ONLY when the user
    # actually tries to use Chat mode, exactly mirroring app.py's st.error() wording/intent
    # without app.py's all-or-nothing blocking behavior.
    api_key = _load_openrouter_key()
    if api_key:
        qa_chain = build_qa_chain(vectordb, api_key)
        cl.user_session.set("qa_chain", qa_chain)
    else:
        cl.user_session.set("qa_chain", None)

    actions = [
        cl.Action(name="clear_conversation", icon="eraser", payload={}, label="مسح المحادثة"),
        cl.Action(name="export_conversation", icon="download", payload={}, label="تصدير المحادثة (Markdown)"),
    ]
    welcome = (
        "مرحباً بك في مساعد البحث القانوني الذكي — Capital Legal Base (نموذج أولي، نسخة Chainlit).\n\n"
        "اكتب سؤالك القانوني مباشرة للمحادثة مع النموذج، أو اختر الأمر **" + SEARCH_COMMAND_ID + "** "
        "من قائمة الأوامر (زر \"/\") لبحث مباشر بدون نموذج لغوي. استخدم لوحة الإعدادات (⚙) لاختيار "
        "المصادر وعدد النتائج."
    )
    if not api_key:
        welcome += (
            "\n\n⚠️ لم يتم العثور على OPENROUTER_API_KEY — المحادثة مع النموذج معطّلة حالياً، "
            "لكن البحث المباشر (" + SEARCH_COMMAND_ID + ") يعمل بدون الحاجة لمفتاح API."
        )
    await cl.Message(content=welcome, actions=actions).send()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("selected_sources", settings.get("selected_sources") or [])
    cl.user_session.set("k_value", int(settings.get("k_value") or 10))


async def run_search(query: str):
    selected_sources = cl.user_session.get("selected_sources") or list(ALL_SOURCES)
    k_value = cl.user_session.get("k_value") or 10

    if not selected_sources:
        await cl.Message(content="الرجاء اختيار مصدر واحد على الأقل من لوحة الإعدادات (⚙).").send()
        return

    try:
        vectordb = await cl.make_async(get_vectorstore)()
    except Exception as e:
        await cl.Message(content=f"تعذر الوصول لقاعدة البيانات القانونية: {e}").send()
        return

    def _search():
        # Always fetch a large candidate pool (SEARCH_FETCH_POOL), not just k_value — see that
        # constant's comment for why: Chroma's ANN search misses real matches from non-dominant
        # sources when n_results is small.
        scored = vectordb.similarity_search_with_score(
            query, k=SEARCH_FETCH_POOL, filter=build_source_filter(selected_sources)
        )
        # Drop anything with no real relevance to the query at all — see SEARCH_SCORE_THRESHOLD
        # comment above. Keep everything else in order, best first.
        return [doc for doc, score in scored if score <= SEARCH_SCORE_THRESHOLD]

    raw_results = await cl.make_async(_search)()

    # Dedupe by (source, doc_id): a single sjc/ccb case can be split into several chunk
    # fragments that each independently match the query, and after the full-text lookup below
    # they'd all render identical text — keep only the highest-ranked occurrence.
    seen = set()
    results = []
    for doc in raw_results:
        dedupe_key = (doc.metadata.get("source", ""), doc.metadata.get("doc_id", ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        results.append(doc)
        if len(results) >= int(k_value):
            break

    if not results:
        await cl.Message(content="لا توجد نتائج مطابقة.").send()
        return

    elements = []
    lines = [f"**{len(results)} نتيجة (بحث مباشر بدون نموذج لغوي):**", ""]
    for i, doc in enumerate(results, start=1):
        source_label = doc.metadata.get("source", "")
        doc_id = doc.metadata.get("doc_id", "")
        article = doc.metadata.get("article_no", "")
        article_part = f" — مادة {article}" if article else ""
        full_text = full_text_for_doc(doc)
        snippet = full_text[:200].strip()
        name = f"[{i}] ({source_label}) {doc_id}{article_part}"
        lines.append(f"**{name}**")
        lines.append(f"*{snippet}...*")
        lines.append("")
        elements.append(cl.Text(name=name, content=full_text, display="side"))

    await cl.Message(content="\n".join(lines), elements=elements).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()
    if not query:
        return

    history = cl.user_session.get("history") or []

    if message.command == SEARCH_COMMAND_ID:
        history.append({"role": "user", "content": f"[بحث] {query}"})
        await run_search(query)
        cl.user_session.set("history", history)
        return

    history.append({"role": "user", "content": query})

    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain is None:
        answer = (
            "OPENROUTER_API_KEY غير موجود — الرجاء إضافته الى .streamlit/secrets.toml "
            "(انظر .streamlit/secrets.toml.example) ثم إعادة تشغيل التطبيق. "
            "بامكانك استخدام أمر \"" + SEARCH_COMMAND_ID + "\" للبحث المباشر بدون نموذج لغوي في هذه الأثناء."
        )
        await cl.Message(content=answer).send()
        history.append({"role": "assistant", "content": answer, "sources": []})
        cl.user_session.set("history", history)
        return

    # Chat's retriever k stays hard-coded at 6 (see build_qa_chain's comment — a deliberate,
    # already-tuned middle ground, not meant to be user-adjustable). The source filter, however,
    # is cheap and safe to apply live: ThresholdMMRRetriever.filter is a plain attribute the
    # retriever reads fresh on every call, so updating it in place before invoke() is enough —
    # no need to rebuild the chain/retriever per settings change.
    selected_sources = cl.user_session.get("selected_sources") or list(ALL_SOURCES)
    qa_chain.retriever.filter = build_source_filter(selected_sources)

    async with cl.Step(name="جارٍ البحث والتحقق من المصادر...", type="run"):
        sources = []
        try:
            result = await cl.make_async(qa_chain.invoke)({"question": query})
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            sources = mark_cited(docs_to_sources(source_docs), answer)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                answer = (
                    "تم تجاوز الحد المسموح به من عدد الطلبات لهذه الدقيقة او لهذا اليوم (حد مجاني من OpenRouter). "
                    "الرجاء الانتظار قليلاً ثم إعادة المحاولة."
                )
            else:
                answer = f"حدث خطأ غير متوقع: {e}"

    elements, content_suffix = sources_to_elements(sources)
    await cl.Message(content=answer + content_suffix, elements=elements).send()

    history.append({"role": "assistant", "content": answer, "sources": sources})
    cl.user_session.set("history", history)


@cl.action_callback("clear_conversation")
async def clear_conversation(action: cl.Action):
    cl.user_session.set("history", [])
    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain is not None:
        # The whole point of this action: a UI-level reset alone would leave the
        # ConversationBufferWindowMemory object populated with stale turns, which
        # ConversationalRetrievalChain would keep feeding into every subsequent prompt.
        qa_chain.memory.clear()
    await cl.Message(content="تم مسح المحادثة — يمكنك البدء بسؤال جديد.").send()


@cl.action_callback("export_conversation")
async def export_conversation(action: cl.Action):
    history = cl.user_session.get("history") or []
    if not history:
        await cl.Message(content="لا توجد محادثة لتصديرها بعد.").send()
        return
    export_text = build_export_text(history)
    # mime must be set explicitly: chainlit/element.py only infers it from content bytes via
    # filetype.guess(), which reads binary file signatures (magic numbers) and returns None for
    # plain text like Markdown -- confirmed live, a null mime crashed the frontend's file-element
    # renderer with "Cannot read properties of null (reading 'startsWith')" when this was omitted.
    file_element = cl.File(
        name="capital_legal_base_conversation.md",
        content=export_text.encode("utf-8"),
        mime="text/markdown",
        display="inline",
    )
    await cl.Message(content="سجل المحادثة جاهز للتنزيل:", elements=[file_element]).send()
