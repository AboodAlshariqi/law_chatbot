
import functools
import json
import os
import re
import time
import tomllib
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
PERSIST_DIRECTORY = str(APP_DIR.parent / "data" / "chroma_v2")
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

# Chat's threshold -- unchanged from appchainlit.py/appchainlit_tuned.py.
SEARCH_SCORE_THRESHOLD = 0.95

# --- Change 3: Search gets its own, slightly looser threshold (see module docstring). ---
SEARCH_SCORE_THRESHOLD_BROWSE = 1.05

# Chat's candidate pool -- unchanged.
CHAT_FETCH_K = 20

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
        results = [Document(page_content=kept[i][0], metadata=kept[i][1]) for i in idxs]
        if self.max_chars:
            for d in results:
                if len(d.page_content) > self.max_chars:
                    d.page_content = d.page_content[: self.max_chars] + " […]"
        return results


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
  - اذا توفر في النصوص المرفقة نص تشريعي فقط (بدون احكام قضائية)، اكتب قسم "القوانين ذات الصلة" فقط، ولا تكتب قسم "تطبيقات قضائية" ولا تشر الى غيابه.
  - اذا توفرت احكام قضائية فقط (بدون نص تشريعي مباشر)، اكتب قسم "تطبيقات قضائية" فقط، واذكر ضمن شرحك اي مادة قانونية ورد ذكرها صراحة داخل نص الحكم القضائي نفسه (فهي جزء من النص المرفق، لا اضافة من عندك).
  - اذا توفر النوعان معا، اكتب القسمين، وابدأ بالقوانين ذات الصلة ثم تطبيقاتها القضائية، بحيث يفهم القارئ الحكم العام اولا ثم كيف طبقته المحكمة عمليا.
  - لا تخترع قسما لا يوجد له سند في النصوص المرفقة تحت اي ظرف.
- اختم اجابتك دائماً بسطر منفصل يبدأ حرفياً بـ "المصادر_المستخدمة:" يليه فقط ارقام المواد و/او ارقام القضايا التي استندت اليها فعلاً في متن الاجابة (وليس تلك التي استبعدتها او ذكرتها لتوضيح عدم صلتها)، مفصولة بفواصل. اذا لم تستند الى اي نص مرفق فعلاً (سؤال غير متعلق بالقانون، او لا توجد اجابة في النصوص المرفقة)، اكتب "المصادر_المستخدمة: لا يوجد".

النصوص القانونية:
{context}

السؤال: {question}

الاجابة القانونية المدعومة بالمصادر:"""

CITED_SOURCES_MARKER = "المصادر_المستخدمة:"

QA_CHAIN_PROMPT = PromptTemplate.from_template(SYSTEM_TEMPLATE)

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


OVERLOAD_MARKERS = ("502", "overloaded")


def _invoke_with_retry(chain, payload, max_attempts=6, base_delay=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e).lower()
            if attempt == max_attempts or not any(marker in msg for marker in OVERLOAD_MARKERS):
                raise
            time.sleep(min(base_delay * (2 ** (attempt - 1)), 30))


LLM_PROVIDERS = {
    "openrouter": {"label": "OpenRouter — Nemotron (الاساسي)", "k": 6, "max_chars": None},
    "groq": {"label": "Groq — GPT-OSS-120B (أسرع)", "k": 2, "max_chars": 6000},
}
DEFAULT_LLM_PROVIDER = "openrouter"


def build_qa_chain(vectordb, provider):
    cfg = LLM_PROVIDERS[provider]

    if provider == "groq":
        groq_api_key = _load_secret("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY missing — add it to .streamlit/secrets.toml")
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=groq_api_key)
    else:
        openrouter_api_key = _load_secret("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing — add it to .streamlit/secrets.toml")
        llm = ChatOpenAI(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            temperature=0,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    memory = ConversationBufferWindowMemory(k=cfg["k"], memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=ThresholdMMRRetriever(vectordb=vectordb, k=cfg["k"], max_chars=cfg["max_chars"]),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
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
    idx = answer.find(CITED_SOURCES_MARKER)
    if idx == -1:
        return answer
    return answer[idx + len(CITED_SOURCES_MARKER):]


def mark_cited(sources, answer):
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
    labels = []
    cited = [s for s in sources if s.get("cited")]
    uncited = [s for s in sources if not s.get("cited")]

    def format_block(i, src, is_cited):
        article = src.get("article_no", "")
        article_part = f" — مادة {article}" if article else ""
        heading = f"[{i}] {src.get('source', '')} — {src.get('doc_id', '')}{article_part}"
        if not is_cited:
            heading = f"(غير مستشهد به) {heading}"
        return (
            f"### {heading}\n\n"
            f"**المصدر:** {src.get('source', '')}  \n"
            f"**رقم الوثيقة:** {src.get('doc_id', '')}  \n"
            f"**رقم المادة:** {src.get('article_no', '') or '—'}  \n"
            f"**التصنيف:** {src.get('category', '') or '—'}\n\n"
            f"{src.get('text', '')}\n\n---\n"
        )

    sections = []

    cited_blocks = []
    for i, src in enumerate(cited, start=1):
        labels.append(f"[{i}]")
        cited_blocks.append(format_block(i, src, is_cited=True))
    if cited_blocks:
        sections.append("## المصادر المستشهد بها في الاجابة\n\n" + "\n".join(cited_blocks))

    uncited_blocks = [format_block(j, src, is_cited=False) for j, src in enumerate(uncited, start=1)]
    if uncited_blocks:
        sections.append("## مصادر تم استرجاعها ولم يُستشهد بها\n\n" + "\n".join(uncited_blocks))

    elements = []
    if sections:
        elements.append(cl.Text(
            name="المصادر",
            content="\n\n".join(sections),
            display="side",
        ))

    content_suffix = ("\n\n**المصادر المستشهد بها في الاجابة:** " + " ".join(labels)) if labels else ""
    return elements, content_suffix


def build_export_text(messages):
    lines = ["# سجل المحادثة — Capital Legal Base (نسخة بحث محسّنة)", ""]
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


SEARCH_COMMAND_ID = "بحث"


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            MultiSelect(
                id="selected_sources",
                label="المصادر",
                initial=["lloc", "sjc", "ccb"],
                items={"التشريعات": "lloc", "السوابق القضائية": "sjc", "المحكمة الدستورية": "ccb"},
            ),
            Slider(id="k_value", label="عدد النتائج", initial=10, min=1, max=30, step=1),
            Select(
                id="llm_provider",
                label="مزود النموذج اللغوي",
                items={cfg["label"]: p for p, cfg in LLM_PROVIDERS.items()},
                initial=DEFAULT_LLM_PROVIDER,
            ),
        ]
    ).send()
    cl.user_session.set("selected_sources", ["lloc", "sjc", "ccb"])
    cl.user_session.set("k_value", 10)
    cl.user_session.set("llm_provider", DEFAULT_LLM_PROVIDER)

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

    try:
        vectordb = await cl.make_async(get_vectorstore)()
    except Exception as e:
        cl.user_session.set("vectordb_error", str(e))
        await cl.Message(content=f"تعذر تحميل قاعدة البيانات القانونية: {e}").send()
        return

    key_error = None
    try:
        qa_chain = build_qa_chain(vectordb, DEFAULT_LLM_PROVIDER)
        cl.user_session.set("qa_chain", qa_chain)
    except RuntimeError as e:
        key_error = str(e)
        cl.user_session.set("qa_chain", None)

    actions = [
        cl.Action(name="clear_conversation", icon="eraser", payload={}, label="مسح المحادثة"),
        cl.Action(name="export_conversation", icon="download", payload={}, label="تصدير المحادثة (Markdown)"),
        cl.Action(name="show_sources", icon="library", payload={}, label="عرض كل المصادر"),
    ]
    welcome = (
        "مرحباً بك في مساعد البحث القانوني الذكي — Capital Legal Base "
        "(نموذج أولي، نسخة بحث محسّنة الاسترجاع).\n\n"
        "اكتب سؤالك القانوني مباشرة للمحادثة مع النموذج، أو اختر الأمر **" + SEARCH_COMMAND_ID + "** "
        "من قائمة الأوامر (زر \"/\") لبحث مباشر بدون نموذج لغوي. استخدم لوحة الإعدادات (⚙) لاختيار "
        "المصادر وعدد النتائج ومزود النموذج اللغوي (OpenRouter/Nemotron الافتراضي، أو Groq الأسرع)."
    )
    if key_error:
        welcome += (
            f"\n\n⚠️ {key_error} — المحادثة مع النموذج معطّلة حالياً، "
            "لكن البحث المباشر (" + SEARCH_COMMAND_ID + ") يعمل بدون الحاجة لمفتاح API."
        )
    await cl.Message(content=welcome, actions=actions).send()


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


async def run_search(query: str):
    selected_sources = cl.user_session.get("selected_sources") or list(ALL_SOURCES)
    k_value = int(cl.user_session.get("k_value") or 10)

    if not selected_sources:
        await cl.Message(content="الرجاء اختيار مصدر واحد على الأقل من لوحة الإعدادات (⚙).").send()
        return

    try:
        vectordb = await cl.make_async(get_vectorstore)()
    except Exception as e:
        await cl.Message(content=f"تعذر الوصول لقاعدة البيانات القانونية: {e}").send()
        return

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
    results = []
    for doc in raw_results:
        dedupe_key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("doc_id", ""),
            doc.metadata.get("article_no", ""),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        results.append(doc)
        if len(results) >= k_value:
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

    async with cl.Step(name="جارٍ البحث والتحقق من المصادر...", type="run"):
        sources = []
        try:
            result = await cl.make_async(_invoke_with_retry)(qa_chain, {"question": query})
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
        qa_chain.memory.clear()
    await cl.Message(content="تم مسح المحادثة — يمكنك البدء بسؤال جديد.").send()


@cl.action_callback("show_sources")
async def show_sources(action: cl.Action):
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


@cl.action_callback("export_conversation")
async def export_conversation(action: cl.Action):
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
