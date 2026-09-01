# المرجع — Al-Marja'

An Arabic legal research assistant for Bahraini law. Ask a question in Arabic and get a
structured answer that shows the article or judgment behind every claim.

Built as a Data Science capstone for **General Assembly × BIBF**.

---

## The problem

Bahraini law is public but not searchable. Legislation sits on one government portal,
Court of Cassation rulings on a second, Constitutional Court rulings on a third. None of them
searches across all three, and none links a judgment to the article it applied.

A lawyer researching one question opens document after document by hand — billable time that
cannot be charged to a client.

## What it does

Brings all three sources into one place you can simply ask, over **49,782 passages**:

| source | passages |
|---|---|
| `lloc` — legislation | 38,281 |
| `sjc` — Court of Cassation | 11,339 |
| `ccb` — Constitutional Court | 162 |
| | **12,177 distinct documents** |

Two modes:

- **المحادثة القانونية** — a structured answer: a plain summary, the relevant articles, then
  how courts applied them. Every claim carries a numbered citation.
- **البحث المباشر** — direct search of the texts with no language model at all. Instant
  results, original wording, legislation first, then the judgments that applied it.

**It answers only from passages it retrieved.** If it finds nothing, it says so rather than
guessing.

---

## How it works

**1 · Retrieval.** The question is embedded with `BAAI/bge-m3`. Each selected source is then
queried **separately** for 30 candidates, filtered by a distance threshold, deduplicated, and
reduced to **8 passages** by maximal marginal relevance.

Querying per source is the decision that matters. A single blended query lets one
well-matched source fill the entire candidate pool before another appears — which is how a
criminal-law question returns ten Penal Code articles and not one judgment. Per-source
querying makes that impossible rather than merely unlikely: in evaluation, **14 of 14**
questions retrieved both legislation and case law.

**2 · Generation.** Each passage is labelled with its index before the model sees it:

```
[1] المصدر=lloc | الوثيقة=L1901 | المادة=97
<the article text>
```

The model is instructed to answer only from these and to cite the numbers.

**3 · Citation.** The app reads those numbers back and marks exactly those sources. The
number in the answer is the number in the panel, and each links to the official text.

Citing by index removes a whole class of failure. Re-deriving citations by matching article
numbers in Arabic prose broke constantly — models wrote `مادة 2` for article 3, bare
`97 أ، 97 ب`, or a court's own `الطعن رقم 2/00001/2023/35` for a document stored as
`1 M 2023 K 00`. A copied integer has no variants.

---

## Results

Measured on `chroma_v3` over 14 questions with hand-verified answer keys
(`notebooks/08_evaluation.ipynb`):

| | |
|---|---|
| Hit@1 | **71.4%** |
| Hit@8 | **78.6%** |
| MRR | **0.723** |
| retrieved both source types | **14 / 14** |
| median retrieval time | 1.2 s |
| median answer time | ~20 s (Fanar) |
| cost | ~$0.005 per question |

`Hit@8` is the ceiling on the whole system: the model only ever sees 8 passages, so an
article ranked ninth cannot be cited however good the answer is.

---

## Running it

```bash
pip install -r requirements.txt
```

Download the corpus — see [`data/README.md`](data/README.md) — then add your API keys to
`src/.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "..."
FANAR_API_KEY = "..."
```

```bash
cd src
chainlit run app.py --port 8000 -w
```

البحث المباشر works without any key; only the chat needs a model.

### Models

| | context window | notes |
|---|---|---|
| **Fanar (QCRI)** | 16k | Arabic-native, fastest (~20 s), limits are your own |
| **Nemotron 550B** | 262k | most detailed, slowest (~56 s), free tier is shared |
| **MiniMax M3** | 1M | large context; free tier shares upstream capacity |

---

## Repository

```
notebooks/   01a-01c  scraping, one per source
             02       Arabic normalisation
             03       document splitting
             04       embedding → vector store (Colab)
             05-07    retrieval, QA and chat experiments
             08       evaluation on chroma_v3
src/         app.py and the Chainlit interface
data/        instructions only — the corpus is ~5 GB
```

---

## Limitations

Stated plainly, because they bound what the system can do:

- **Retrieval is the ceiling.** An article ranked ninth cannot be cited. `Hit@8 = 78.6%`.
- **Questions are not normalised.** The corpus was character-normalised during cleaning;
  incoming questions are not, so `إجارة` and the stored `اجارة` embed differently. The
  accuracy figures are therefore a floor.
- **Phrasing matters more than it should.** *"إذا قتل شاب أباه كم يرث؟"* returns nothing;
  *"ما هي موانع الإرث"* returns the exact provision.
- **Case law is thinner in chat than in search.** Chat draws 30 candidates per source; direct
  search draws 300–2,000. A relevant judgment can reach the search view and not the chat.
- **Constitutional Court coverage is small** — 162 passages — and its scanned text carries OCR
  damage.
- **Nothing in the source data links a judgment to the law it applied.** Search infers it by
  scanning judgment text for citations — an approximation, not ground truth.
- **The system constrains sources, not reasoning.** A model can still misread the passages it
  was given.
- **Some areas are not codified at all.** Inheritance shares are governed by Islamic
  jurisprudence under the Constitution, so no article exists to retrieve.

A research aid that shows its work — not a substitute for the official text or for a lawyer.

---

## App Demo
- `Demo.mp4`: https://drive.google.com/file/d/1LCjebHTD7QCWEIUqosYCGDVnzgbR9YTM/view?usp=sharing

## License

MIT — see [LICENSE](LICENSE). The legal texts themselves are Bahraini government publications
and are not covered by it.
