import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores.utils import maximal_marginal_relevance

st.set_page_config(page_title="Capital Legal Base — مساعد البحث القانوني", layout="wide")

PERSIST_DIRECTORY = str(Path(__file__).parent.parent / "data" / "chroma")
PROCESSED_DIRECTORY = Path(__file__).parent.parent / "data" / "processed"

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
    helper: also usable by Chat's retriever later (as_retriever(search_kwargs={"filter": ...})),
    not Search-specific."""
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


@st.cache_resource
def load_vectorstore():
    if not Path(PERSIST_DIRECTORY).exists():
        st.error(f"Vectorstore not found at {PERSIST_DIRECTORY} — unzip the Drive-backed vectorstore there first.")
        st.stop()

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)


@st.cache_resource
def load_qa_chain():
    openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        st.error("OPENROUTER_API_KEY missing — add it to .streamlit/secrets.toml (see .streamlit/secrets.toml.example)")
        st.stop()

    vectordb = load_vectorstore()
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


@st.cache_data
def load_normalized_lookup(source):
    """Load a *_normalized.json file once and build a {doc_id: full_text} dict for O(1) lookup.
    Only applies to sjc/ccb — see NORMALIZED_SOURCES comment above."""
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
# correctly split and labeled with its own article_no (see 03_document_splitting_v2.ipynb), so THAT
# is the right source for a lloc fallback: keyed by (doc_id, article_no), not doc_id alone.
LLOC_SPLITS_PATH = str(PROCESSED_DIRECTORY / "document_splits_v2.json")


@st.cache_data
def load_lloc_article_lookup():
    """{(doc_id, article_no): full_article_text} for lloc, built from the corrected v2 splitting
    output. A lloc chunk in the vectorstore is meant to already BE exactly one correctly-bounded
    article, so this is a safety net for the rare chunk that isn't — not a routine fallback the way
    the sjc/ccb lookup is today (most of those routinely fall back; this one mostly won't)."""
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
    (doc_id, article_no) — see the comment above LLOC_SPLITS_PATH for why this differs from the
    sjc/ccb pattern. Falls back to the chunk's own text if no lookup match is found either way."""
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
    """Build small serializable source dicts straight from retrieved Documents
    (not from LLM self-citation, which is unreliable — confirmed in prior testing)."""
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
    the Sources panel presents all of them as if they backed the answer, which is misleading."""
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


def _render_source_expander(src, label, key):
    with st.expander(label):
        st.json(
            {
                "source": src.get("source", ""),
                "doc_id": src.get("doc_id", ""),
                "article_no": src.get("article_no", ""),
                "category": src.get("category", ""),
            }
        )
        st.caption(f"{len(src.get('text', ''))} characters total — full text below (this is exactly what the model saw)")
        st.text_area("النص الكامل", src.get("text", ""), height=300, label_visibility="collapsed", key=key)


def render_sources_list(sources, key_prefix):
    """Render a numbered, expandable source list under a chat answer or in the Sources tab.
    Only sources the model actually referenced in its answer (per mark_cited) are numbered and
    shown up front — the retriever routinely returns topically-adjacent-but-unused chunks
    alongside the real ones, and showing all of them as equal-weight "sources" is misleading.
    Anything retrieved but not cited is still shown, just collapsed separately, so nothing is
    silently hidden if the citation-matching heuristic misses a real reference."""
    cited = [s for s in sources if s.get("cited")]
    uncited = [s for s in sources if not s.get("cited")]

    labels = []
    for i, src in enumerate(cited, start=1):
        article = src.get("article_no", "")
        article_part = f" — مادة {article}" if article else ""
        label = f"[{i}] {src.get('source', '')} — {src.get('doc_id', '')}{article_part}"
        labels.append(f"[{i}]")
        _render_source_expander(src, label, key=f"{key_prefix}_cited_{i}")
    if labels:
        st.markdown("**المصادر المستشهد بها في الاجابة:** " + " ".join(labels))

    if uncited:
        with st.expander(f"مصادر أخرى تم استرجاعها ولم يُستشهد بها صراحة في الاجابة ({len(uncited)})"):
            for j, src in enumerate(uncited, start=1):
                article = src.get("article_no", "")
                article_part = f" — مادة {article}" if article else ""
                label = f"{src.get('source', '')} — {src.get('doc_id', '')}{article_part}"
                _render_source_expander(src, label, key=f"{key_prefix}_uncited_{j}")


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


if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    with st.spinner("جارٍ تحميل قاعدة البيانات القانونية..."):
        st.session_state.qa_chain = load_qa_chain()

st.title("مساعد البحث القانوني الذكي")
st.caption("Capital Legal Base — نموذج أولي")

tab_chat, tab_search, tab_sources, tab_settings = st.tabs(["المحادثة", "البحث", "المصادر", "الإعدادات"])

with tab_chat:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.code(msg["content"], language=None)
                sources = msg.get("sources") or []
                if sources:
                    render_sources_list(sources, key_prefix=f"hist_{i}")
            else:
                st.markdown(msg["content"])

    query = st.chat_input("اكتب سؤالك القانوني هنا...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("جارٍ البحث والتحقق من المصادر..."):
                sources = []
                try:
                    result = st.session_state.qa_chain.invoke({"question": query})
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])
                    sources = mark_cited(docs_to_sources(source_docs), answer)
                    st.session_state.last_sources = source_docs
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        answer = (
                            "تم تجاوز الحد المسموح به من عدد الطلبات لهذه الدقيقة او لهذا اليوم (حد مجاني من OpenRouter). "
                            "الرجاء الانتظار قليلاً ثم إعادة المحاولة."
                        )
                    else:
                        answer = f"حدث خطأ غير متوقع: {e}"
                st.code(answer, language=None)
                new_index = len(st.session_state.messages) + 1
                if sources:
                    render_sources_list(sources, key_prefix=f"new_{new_index}")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with tab_search:
    st.markdown("**بحث مباشر في قاعدة البيانات القانونية (بدون نموذج لغوي — نتائج فورية)**")

    # Wrapped in a form so adjusting the source filter or k doesn't itself trigger a fresh vector
    # search — without this, Streamlit reruns the whole script (and re-embeds the query) on every
    # single widget change, not just when "بحث" is pressed. Results are stashed in session_state
    # so they survive unrelated reruns elsewhere in the app (e.g. asking a question in Chat) —
    # st.form_submit_button's return value is only True on the exact rerun that submitted it, so
    # relying on that alone would make results vanish the next time anything else in the app reruns.
    with st.form(key="search_form"):
        search_query = st.text_input("ابحث عن مصطلح او عبارة قانونية...", key="search_query_input")
        col_sources, col_k = st.columns([2, 1])
        with col_sources:
            selected_sources = st.multiselect(
                "المصادر",
                options=list(ALL_SOURCES),
                default=list(ALL_SOURCES),
                format_func=lambda code: SOURCE_LABELS.get(code, code),
                key="search_source_filter",
            )
        with col_k:
            k_value = st.number_input("عدد النتائج", min_value=1, value=10, step=1, key="search_k_input")
        submitted = st.form_submit_button("بحث")

    if submitted:
        if not selected_sources:
            st.session_state.search_results = None
            st.session_state.search_warning = "الرجاء اختيار مصدر واحد على الأقل للبحث."
        elif not search_query:
            st.session_state.search_results = None
            st.session_state.search_warning = None
        else:
            with st.spinner("جارٍ البحث..."):
                vectordb = load_vectorstore()
                # Always fetch a large candidate pool (SEARCH_FETCH_POOL), not just k_value — see
                # that constant's comment for why: Chroma's ANN search misses real matches from
                # non-dominant sources when n_results is small.
                raw_results = vectordb.similarity_search_with_score(
                    search_query, k=SEARCH_FETCH_POOL, filter=build_source_filter(selected_sources)
                )

            # Drop anything with no real relevance to the query at all (score above the
            # threshold — see SEARCH_SCORE_THRESHOLD comment). Keep the rest, best first.
            raw_results = [doc for doc, score in raw_results if score <= SEARCH_SCORE_THRESHOLD]

            # Dedupe by (source, doc_id): a single sjc/ccb case can be split into several chunk
            # fragments that each independently match the query, and after the full-text lookup
            # below they'd all render identical text — keep only the highest-ranked occurrence.
            seen = set()
            deduped = []
            for doc in raw_results:
                dedupe_key = (doc.metadata.get("source", ""), doc.metadata.get("doc_id", ""))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                deduped.append(doc)
                if len(deduped) >= int(k_value):
                    break
            st.session_state.search_results = deduped
            st.session_state.search_warning = None

    if st.session_state.get("search_warning"):
        st.info(st.session_state.search_warning)
    elif st.session_state.get("search_results") is not None:
        results = st.session_state.search_results
        if not results:
            st.info("لا توجد نتائج مطابقة.")
        else:
            st.markdown(f"**{len(results)} نتيجة:**")
            for i, doc in enumerate(results, start=1):
                source_label = doc.metadata.get("source", "")
                doc_id = doc.metadata.get("doc_id", "")
                article = doc.metadata.get("article_no", "")
                article_part = f" — مادة {article}" if article else ""
                full_text = full_text_for_doc(doc)
                snippet = full_text[:200].strip()
                label = f"[{i}] ({source_label}) {doc_id}{article_part}"
                with st.expander(label):
                    st.caption(f"المصدر: {source_label}")
                    st.markdown(f"*{snippet}...*")
                    st.json(doc.metadata)
                    # Key must depend on the actual document, not just its position: with a
                    # positional key like f"search_result_{i}", Streamlit reuses whatever value
                    # was first stored under that key for the lifetime of the session and ignores
                    # the new `full_text` argument on every later rerun — so result #2 kept
                    # showing content from a completely different document once the position had
                    # ever been rendered before (confirmed live: an sjc case bled through onto an
                    # unrelated lloc result). Keying on the document identity means a fresh
                    # doc at that position always gets a fresh widget.
                    st.text_area(
                        "النص الكامل",
                        full_text,
                        height=300,
                        label_visibility="collapsed",
                        key=f"search_result_{source_label}_{doc_id}_{article}_{i}",
                    )

with tab_sources:
    assistant_turns = [
        (i, msg) for i, msg in enumerate(st.session_state.messages)
        if msg["role"] == "assistant" and msg.get("sources")
    ]
    if not assistant_turns:
        st.info("لا توجد مصادر بعد — اطرح سؤالاً اولاً.")
    else:
        st.markdown("**المصادر المسترجعة لكل سؤال:**")
        for i, msg in assistant_turns:
            question_text = ""
            if i > 0 and st.session_state.messages[i - 1]["role"] == "user":
                question_text = st.session_state.messages[i - 1]["content"]
            st.markdown(f"#### السؤال: {question_text}" if question_text else "#### سؤال")
            render_sources_list(msg["sources"], key_prefix=f"src_tab_{i}")
            st.divider()

# Rendered last (after tab_chat) on purpose: Streamlit executes the whole script top-to-bottom
# on every rerun, and switching tabs in the browser does not itself trigger a rerun — so a block
# placed before tab_chat would render using session_state.messages as it stood BEFORE this turn's
# question/answer was appended, showing the export button one turn stale. Placing it last ensures
# it always reflects the freshest state within the same rerun.
with tab_settings:
    if st.button("مسح المحادثة", type="primary"):
        st.session_state.messages = []
        st.session_state.qa_chain.memory.clear()
        st.session_state.pop("last_sources", None)
        st.rerun()
    st.caption("يمسح سجل المحادثة لبدء موضوع جديد")

    st.divider()

    st.markdown("**تصدير المحادثة**")
    if st.session_state.messages:
        export_text = build_export_text(st.session_state.messages)
        st.download_button(
            "تصدير المحادثة (Markdown)",
            data=export_text,
            file_name="capital_legal_base_conversation.md",
            mime="text/markdown",
        )
    else:
        st.caption("لا توجد محادثة لتصديرها بعد.")
