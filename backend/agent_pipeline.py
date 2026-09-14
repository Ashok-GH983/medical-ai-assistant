# =============================================================================
# agent_pipeline.py — Multi-agent LangGraph pipeline (hybrid model strategy)
# =============================================================================
# Model routing:
#   - llm_fast  (llama-3.1-8b-instant)  -> planner, screening, extraction, gaps
#   - llm_smart (openai/gpt-oss-20b)    -> synthesis, retry only
#
# This splits token consumption across TWO separate daily quotas, letting you
# run the pipeline ~2x more often without hitting a TPD wall.
# =============================================================================

import os
import re
import json
import uuid
import time
from typing import List, Dict, Any, TypedDict, Optional
from urllib.parse import urljoin

import requests
import xml.etree.ElementTree as ET
import redis
import pymupdf
import chromadb

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

load_dotenv()

# -----------------------------------------------------------------------------
# Redis cache
# -----------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=3)
    redis_client.ping()
    print("[INFO] Connected to Redis Cache.")
except Exception as e:
    redis_client = None
    print(f"[WARNING] Redis not available, running without cache: {e}")

PUBMED_CACHE_TTL = 60 * 60 * 24 * 7
LLM_CACHE_TTL = 60 * 60 * 24      # LLM outputs cached 24h (cuts repeat-query cost to zero)


def _cache_get(key: str) -> Optional[Any]:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _cache_set(key: str, value: Any, ttl: int = PUBMED_CACHE_TTL) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


# -----------------------------------------------------------------------------
# TWO models — cheap for volume tasks, smart for synthesis only
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""

# Fast + high-limit model — replacement for llama-3.1-8b-instant
llm_fast = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1,
    max_tokens=4096,
)

# Smart model reserved for the synthesis step — replacement for llama-3.3-70b-versatile
llm_smart = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=8192,
)


embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# -----------------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    run_id: str
    query: str
    keywords: List[str]
    sub_questions: List[str]
    pmids: List[str]
    raw_papers: List[Dict[str, Any]]
    screened_papers: List[Dict[str, Any]]
    extractions: List[Dict[str, Any]]
    dataset_comparison: List[Dict[str, Any]]
    method_comparison: List[Dict[str, Any]]
    metric_summary: List[Dict[str, Any]]
    research_gaps: List[str]
    synthesis: str
    references: List[Dict[str, Any]]
    review_status: str
    status: str


# =============================================================================
# safe_llm_invoke — works for either model, caches result, backs off on 429
# =============================================================================
def safe_llm_invoke(
    prompt: str,
    model: str = "fast",       # "fast" or "smart"
    max_retries: int = 4,
    use_cache: bool = True,
) -> str:
    """
    Invoke the LLM with:
      - optional Redis cache keyed on model + prompt hash
      - exponential backoff on 429 / TPD / TPM rate-limit errors
      - fail-soft on other errors
    """
    # --- Cache lookup ---
    import hashlib
    cache_key = None
    if use_cache and redis_client:
        h = hashlib.sha256(f"{model}||{prompt}".encode("utf-8")).hexdigest()[:32]
        cache_key = f"llm:{h}"
        cached = _cache_get(cache_key)
        if cached is not None:
            print(f"[LLM CACHE] Hit for {model} prompt ({len(prompt)} chars)")
            return cached

    client = llm_fast if model == "fast" else llm_smart
    last_err = ""

    for attempt in range(max_retries):
        try:
            res = client.invoke(prompt)
            text = str(res.content)

            if cache_key:
                _cache_set(cache_key, text, ttl=LLM_CACHE_TTL)
            return text

        except Exception as e:
            msg = str(e).lower()
            last_err = str(e)

            is_rate_limit = (
                "429" in msg
                or "rate limit" in msg
                or "too many requests" in msg
                or "tokens per minute" in msg
                or "tokens per day" in msg
            )

            if is_rate_limit and attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"[LLM] {model} rate limited (attempt {attempt+1}/{max_retries}). Sleeping {wait}s...")
                time.sleep(wait)
                continue

            print(f"LLM Error [{model}]: {e}")
            return ""

    print(f"[LLM] {model} exhausted retries. Last error: {last_err}")
    return ""


# =============================================================================
# 1. QUERY PLANNING AGENT  (fast model)
# =============================================================================
def planner_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")

    prompt = f"""You are a medical literature retrieval assistant.

Given the research question below, produce a JSON object with exactly two keys:
  - "pubmed_query": a concise, highly specific PubMed search string
  - "sub_questions": a list of 3 focused sub-questions to guide the review

Focus on: epilepsy, seizure prediction/detection, artificial intelligence,
deep learning, machine learning, EEG.

Research question: {query}

Return ONLY valid JSON. No prose, no markdown fences.
Example:
{{"pubmed_query": "(epilepsy OR seizure) AND (deep learning OR EEG)", "sub_questions": ["q1","q2","q3"]}}
"""

    raw = safe_llm_invoke(prompt, model="fast").strip()

    pubmed_query = ""
    sub_questions: List[str] = []
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        pubmed_query = str(parsed.get("pubmed_query", "")).strip()
        sub_questions = [
            str(q).strip() for q in parsed.get("sub_questions", []) if str(q).strip()
        ][:3]
    except Exception:
        pubmed_query = raw.split("\n")[0].strip() if raw else ""

    if not pubmed_query or len(pubmed_query) < 5:
        pubmed_query = (
            "(seizure prediction OR seizure forecasting OR seizure detection) "
            "AND (epilepsy OR EEG) AND (deep learning OR machine learning)"
        )

    if not sub_questions:
        sub_questions = [
            "Which datasets and patient populations are used?",
            "Which model architectures and validation strategies are applied?",
            "Which evaluation metrics and reported performance are common?",
        ]

    return {
        "keywords": [pubmed_query],
        "sub_questions": sub_questions,
        "status": "Searching",
    }


# =============================================================================
# 2. LITERATURE RETRIEVAL AGENT  (no LLM)
# =============================================================================
def fetch_fulltext_pdf_from_pmc(pmid: str) -> str:
    cache_key = f"pmc_pdf:{pmid}"
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"[CACHE] PMC PDF hit for PMID {pmid}")
        return cached

    try:
        pmc_search_url = (
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        )
        res = requests.get(pmc_search_url, timeout=5).json()
        records = res.get("records", [])
        if not records or "pmcid" not in records[0]:
            _cache_set(cache_key, "")
            return ""

        pmcid = records[0]["pmcid"]
        pmc_page_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        page_res = requests.get(
            pmc_page_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(page_res.text, "html.parser")

        pdf_href = None
        for a_tag in soup.find_all("a", href=True):
            if a_tag["href"].endswith(".pdf"):
                pdf_href = a_tag["href"]
                break

        if not pdf_href:
            _cache_set(cache_key, "")
            return ""

        if pdf_href.startswith("http"):
            full_pdf_url = pdf_href
        else:
            full_pdf_url = urljoin(pmc_page_url, pdf_href)

        print(f"[PDF ENGINE] Resolved PDF URL for PMID {pmid}: {full_pdf_url}")

        pdf_res = requests.get(
            full_pdf_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        if pdf_res.status_code != 200:
            print(f"[PDF ENGINE] HTTP {pdf_res.status_code} for PMID {pmid}")
            _cache_set(cache_key, "")
            return ""

        doc = pymupdf.open(stream=pdf_res.content, filetype="pdf")
        full_text = [page.get_text() for page in doc]
        extracted_str = " ".join(full_text).strip()

        sectioned = _extract_methodology_sections(extracted_str)
        final_text = sectioned if sectioned else extracted_str[:6000]

        print(f"[PDF ENGINE] Extracted {len(final_text)} chars for PMID {pmid}")
        _cache_set(cache_key, final_text)
        return final_text

    except Exception as e:
        print(f"[PDF ENGINE WARNING] Full PDF extraction failed for PMID {pmid}: {e}")
        return ""


def _extract_methodology_sections(text: str) -> str:
    patterns = [
        r"(?i)(methods?|materials? and methods?|methodology).{0,5000}",
        r"(?i)(results? and discussion).{0,5000}",
        r"(?i)(results?).{0,3000}",
    ]
    out: List[str] = []
    for p in patterns:
        m = re.search(p, text)
        if m:
            out.append(m.group(0))
        if sum(len(x) for x in out) > 6000:
            break
    return " ".join(out).strip()[:6000]


def retrieval_node(state: AgentState) -> Dict[str, Any]:
    keywords = state.get("keywords", [])
    search_term = keywords[0] if keywords else "(seizure prediction) AND (epilepsy)"

    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": search_term,
        "retmode": "json",
        "retmax": 10,               # 12 -> 10 (saves tokens in downstream LLM calls)
        "sort": "relevance",
    }

    try:
        res = requests.get(esearch_url, params=search_params, timeout=10).json()
        pmids = res.get("esearchresult", {}).get("idlist", [])
    except Exception:
        pmids = []

    if not pmids:
        return {"pmids": [], "raw_papers": [], "status": "Screening"}

    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}

    raw_papers: List[Dict[str, Any]] = []
    try:
        fetch_res = requests.get(efetch_url, params=fetch_params, timeout=12)
        root = ET.fromstring(fetch_res.content)

        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "N/A")
            abstract_texts = [
                ab.text for ab in article.findall(".//AbstractText") if ab.text
            ]
            abstract = " ".join(abstract_texts) if abstract_texts else "Abstract unavailable."
            pubdate = article.findtext(".//Journal/JournalIssue/PubDate/Year", "2024")
            journal = article.findtext(".//Journal/Title", "Unknown Journal")

            authors_list = []
            for author in article.findall(".//AuthorList/Author"):
                last_name = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")
                if last_name:
                    authors_list.append(f"{last_name} {initials}".strip())
            authors_str = ", ".join(authors_list) if authors_list else "Unknown Authors"

            pdf_content = fetch_fulltext_pdf_from_pmc(pmid)
            content_source = "Full-Text PDF" if pdf_content else "PubMed Abstract"
            text_payload = pdf_content if pdf_content else abstract

            raw_papers.append({
                "pmid": pmid,
                "title": title,
                "authors": authors_str,
                "journal": journal,
                "abstract": text_payload,
                "source_type": content_source,
                "pubdate": pubdate,
            })
    except Exception as e:
        print(f"Parsing error: {e}")

    return {"pmids": pmids, "raw_papers": raw_papers, "status": "Screening"}


# =============================================================================
# 3. SCREENING AGENT  (fast model, batched, permissive + keyword rescue)
# =============================================================================
def screening_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    raw = state.get("raw_papers", [])

    if not raw:
        return {"screened_papers": [], "status": "Extracting"}

    lines: List[str] = []
    for i, p in enumerate(raw):
        excerpt = (p.get("abstract") or "")[:400].replace("\n", " ")
        lines.append(f"[{i}] PMID {p['pmid']} — {p['title']}\n    {excerpt}")
    listing = "\n\n".join(lines)

    prompt = f"""You are screening papers for a medical literature review.

Research question: {query}

Your job is to DROP only papers that are CLEARLY unrelated to the question.
The default for every paper is KEEP. Be permissive — if the paper touches on
seizure detection, epilepsy, EEG, deep learning for medical signals, or any
closely related topic, KEEP it.

Output EXACTLY one line per paper, no other text:

  <index>: KEEP
  <index>: DROP

Papers:
{listing}
"""

    verdicts = safe_llm_invoke(prompt, model="fast")

    keep_indices: set = set()
    for line in verdicts.split("\n"):
        m = re.match(
            r"^\s*\[?(\d+)\]?\s*[:\-]\s*(KEEP|DROP)", line.strip(), re.IGNORECASE
        )
        if not m:
            continue
        idx = int(m.group(1))
        if m.group(2).upper() == "KEEP":
            keep_indices.add(idx)

    if not keep_indices:
        print("[SCREEN] No KEEP verdicts parsed — keeping all papers.")
        keep_indices = set(range(len(raw)))

    kept = [raw[i] for i in range(len(raw)) if i in keep_indices]
    dropped = [raw[i]["pmid"] for i in range(len(raw)) if i not in keep_indices]

    # Keyword-overlap rescue if too many were dropped
    MIN_KEEP = max(3, len(raw) // 2)
    if len(kept) < MIN_KEEP:
        print(
            f"[SCREEN] Over-aggressive filter ({len(kept)}/{len(raw)}). "
            f"Rescuing via keyword overlap..."
        )
        stop = {
            "the", "a", "an", "of", "and", "or", "for", "in", "on", "to",
            "with", "is", "are", "was", "were", "be", "been", "by", "as",
            "at", "from", "that", "this", "these", "those", "recent",
            "recently", "using", "used", "based", "compare", "comparison",
            "methods", "method", "research", "review",
        }
        query_words = {
            w.lower() for w in re.findall(r"[a-zA-Z]{4,}", query)
            if w.lower() not in stop
        }
        scored: List[tuple] = []
        for p in raw:
            haystack = (p["title"] + " " + p["abstract"][:600]).lower()
            score = sum(1 for w in query_words if w in haystack)
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        kept = [p for _, p in scored[: max(MIN_KEEP, len(raw) // 2)]]
        dropped = [p["pmid"] for p in raw if p not in kept]
        print(f"[SCREEN] Rescue kept top {len(kept)} papers by keyword overlap.")

    if dropped:
        print(f"[SCREEN] Dropped {len(dropped)} paper(s): {dropped}")
    print(f"[SCREEN] Kept {len(kept)}/{len(raw)} papers.")

    return {"screened_papers": kept, "status": "Extracting"}


# =============================================================================
# 4. EVIDENCE EXTRACTION AGENT  (fast model, batched)
# =============================================================================
def _parse_extraction_block(block_text: str, paper: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single paper's key-value extraction block (tolerant of markdown)."""
    fields = {
        "dataset": "Not reported",
        "sample_size": "Not reported",
        "validation": "Not reported",
        "model": "Not reported",
        "metrics": "Not reported",
        "key_findings": "Not reported",
    }
    label_map = {
        "dataset": "dataset",
        "sample size": "sample_size",
        "sample_size": "sample_size",
        "n": "sample_size",
        "validation": "validation",
        "model": "model",
        "metrics": "metrics",
        "findings": "key_findings",
        "key findings": "key_findings",
    }

    for raw_line in block_text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Strip markdown bold/italics/backticks and leading bullets
        cleaned = re.sub(r"[*_`#>\-•\s]+", " ", line).strip()
        # Also strip any leading [n] or 0) marker on the same line
        cleaned = re.sub(r"^\[?\d+\]?[\)\.\:]?\s*", "", cleaned)

        # Try "Label: value" pattern, case-insensitive
        m = re.match(r"^([A-Za-z _]+?)\s*[:\-]\s*(.+)$", cleaned)
        if not m:
            continue
        label_raw = m.group(1).strip().lower()
        value = m.group(2).strip()

        # Try exact and progressive prefix matches
        for lbl, key in label_map.items():
            if label_raw == lbl or label_raw.startswith(lbl):
                if value and value.lower() not in ("not reported", "n/a", "na", "unknown", ""):
                    fields[key] = value
                break

    return fields


def extractor_node(state: AgentState) -> Dict[str, Any]:
    papers = state.get("screened_papers", [])
    if not papers:
        return {"extractions": [], "status": "Comparing"}

    CHUNK_SIZE = 3
    all_extractions: List[Dict[str, Any]] = []

    # Process in small chunks so one failure doesn't sink the whole run
    for start in range(0, len(papers), CHUNK_SIZE):
        chunk = papers[start:start + CHUNK_SIZE]
        local_index = {i: p for i, p in enumerate(chunk)}

        blocks: List[str] = []
        for i, p in local_index.items():
            text = (p.get("abstract") or "")[:1500].replace("\n", " ")
            blocks.append(f"[{i}] PMID {p['pmid']}\nTitle: {p['title']}\nText: {text}")
        listing = "\n\n".join(blocks)

        prompt = f"""Extract structured evidence from each paper below.

For EACH paper, output a plain-text block in EXACTLY this format.
Do not use markdown bold, italics, backticks, or bullet points.

[{0 if chunk else 0}]
Dataset: <name or Not reported>
Sample Size: <N or Not reported>
Validation: <protocol or Not reported>
Model: <architecture or Not reported>
Metrics: <numbers or Not reported>
Findings: <one sentence>

Rules:
- Use "Not reported" whenever the text does not state a field.
- Do NOT invent numbers, dataset names, or results.
- Output every paper. Preserve the [index] markers exactly.
- Plain text only. No ** ** , no #, no backticks.

Papers:
{listing}
"""

        raw_out = safe_llm_invoke(prompt, model="fast")

        # Split by [index] markers, tolerant of spaces and markdown asterisks
        parts = re.split(r"\[\s*(\d+)\s*\]", raw_out or "")

        chunk_parsed = 0
        for i in range(1, len(parts) - 1, 2):
            try:
                idx = int(parts[i])
            except ValueError:
                continue
            if idx not in local_index:
                continue
            body = parts[i + 1] if i + 1 < len(parts) else ""
            paper = local_index[idx]
            fields = _parse_extraction_block(body, paper)
            all_extractions.append({
                "pmid": paper["pmid"],
                "title": paper["title"],
                "pubdate": paper.get("pubdate", "2024"),
                "journal": paper.get("journal", ""),
                "source_type": paper.get("source_type", "PubMed Abstract"),
                **fields,
            })
            chunk_parsed += 1

        # Fail-loud: if this chunk parsed nothing, log raw output for diagnosis
        if chunk_parsed == 0:
            preview = (raw_out or "")[:300].replace("\n", " | ")
            print(f"[EXTRACT] Chunk {start}–{start+len(chunk)-1}: ZERO parsed. Raw preview: {preview!r}")
            # Fill this chunk with default rows so papers aren't lost downstream
            for idx, paper in local_index.items():
                all_extractions.append({
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "pubdate": paper.get("pubdate", "2024"),
                    "journal": paper.get("journal", ""),
                    "source_type": paper.get("source_type", "PubMed Abstract"),
                    "dataset": "Not reported",
                    "sample_size": "Not reported",
                    "validation": "Not reported",
                    "model": "Not reported",
                    "metrics": "Not reported",
                    "key_findings": paper["title"],
                })
        else:
            print(f"[EXTRACT] Chunk {start}–{start+len(chunk)-1}: parsed {chunk_parsed}/{len(chunk)}.")

    # Ensure every paper has an extraction row
    covered = {e["pmid"] for e in all_extractions}
    for p in papers:
        if p["pmid"] not in covered:
            all_extractions.append({
                "pmid": p["pmid"],
                "title": p["title"],
                "pubdate": p.get("pubdate", "2024"),
                "journal": p.get("journal", ""),
                "source_type": p.get("source_type", "PubMed Abstract"),
                "dataset": "Not reported",
                "sample_size": "Not reported",
                "validation": "Not reported",
                "model": "Not reported",
                "metrics": "Not reported",
                "key_findings": p["title"],
            })

    print(f"[EXTRACT] Total parsed {len(all_extractions)} extraction(s) from {len(papers)} papers.")
    return {"extractions": all_extractions, "status": "Comparing"}


# =============================================================================
# 5. COMPARISON AGENT  (fast model for gaps only)
# =============================================================================
def comparison_node(state: AgentState) -> Dict[str, Any]:
    extractions = state.get("extractions", [])

    if not extractions:
        return {
            "dataset_comparison": [],
            "method_comparison": [],
            "metric_summary": [],
            "research_gaps": [
                "No papers were retrieved and screened for this query.",
                "Consider broadening the query or adjusting focus terms.",
                "Verify PubMed connectivity and API rate limits.",
            ],
            "status": "Synthesizing",
        }

    dataset_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in extractions:
        ds = (item.get("dataset") or "Not reported").strip()
        key = ds.split(",")[0].strip() or "Not reported"
        dataset_groups.setdefault(key, []).append({
            "pmid": item.get("pmid"),
            "dataset": ds,
            "sample_size": item.get("sample_size"),
            "validation": item.get("validation"),
            "metrics": item.get("metrics"),
        })

    dataset_comp = [
        {"dataset_family": k, "papers": v} for k, v in dataset_groups.items()
    ]

    method_comp = [
        {
            "pmid": item.get("pmid"),
            "model": item.get("model"),
            "validation": item.get("validation"),
            "key_findings": item.get("key_findings"),
        }
        for item in extractions
    ]

    metric_summary: List[Dict[str, Any]] = []
    for item in extractions:
        metrics_text = item.get("metrics") or ""
        numbers = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", metrics_text)
        metric_summary.append({
            "pmid": item.get("pmid"),
            "reported": metrics_text,
            "numeric_percentages": [float(n) for n in numbers],
        })

    gap_prompt = f"""Based on these study extractions, identify exactly 3 specific,
distinct research gaps or open challenges.

{json.dumps(extractions, indent=2)[:3500]}

STRICT RULES:
1. Do NOT write introductory sentences, markdown titles, or markdown tables.
2. Output EXACTLY 3 lines, each starting with a single dash (-).
3. Each gap must be grounded in an observed limitation across the studies.
"""
    gap_res = safe_llm_invoke(gap_prompt, model="fast")

    parsed_gaps: List[str] = []
    if gap_res:
        for line in gap_res.strip().split("\n"):
            clean = line.lstrip("-*123456789. ").strip()
            if clean and not clean.startswith("|") and not clean.startswith("#"):
                parsed_gaps.append(clean)

    if len(parsed_gaps) < 3:
        parsed_gaps = [
            "Limited external validation across multi-center, demographically diverse cohorts.",
            "Inconsistent validation protocols (patient-wise vs. record-wise) hinder cross-study comparability.",
            "Interpretability and real-time deployment on wearable hardware remain under-addressed.",
        ]

    return {
        "dataset_comparison": dataset_comp,
        "method_comparison": method_comp,
        "metric_summary": metric_summary,
        "research_gaps": parsed_gaps[:3],
        "status": "Synthesizing",
    }


# =============================================================================
# 6. SYNTHESIS AGENT  (SMART model — the only expensive call)
# =============================================================================
PMID_PATTERN = re.compile(r"\[PMID:\s*(\d+)\]")


def _check_grounding(synthesis: str, real_pmids: set) -> tuple:
    cited = set(PMID_PATTERN.findall(synthesis or ""))
    hallucinated = cited - real_pmids

    if hallucinated:
        return False, "hallucinated", hallucinated
    if not cited:
        return False, "no_citations", set()
    return True, "ok", cited


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    screened = state.get("screened_papers", [])
    extractions = state.get("extractions", [])
    sub_questions = state.get("sub_questions", [])
    research_gaps = state.get("research_gaps", [])
    run_id = state.get("run_id", str(uuid.uuid4()))

    if not screened:
        return {
            "run_id": run_id,
            "synthesis": (
                f"### Literature Review: {query}\n\n"
                "No relevant papers were retrieved from PubMed for this query."
            ),
            "references": [],
            "review_status": "Approved — empty corpus",
            "status": "Completed",
        }

    docs = [
        Document(
            page_content=f"PMID {p['pmid']} | {p['title']}. {p['abstract']}",
            metadata={"pmid": p["pmid"]},
        )
        for p in screened
    ]

    client = chromadb.EphemeralClient()
    collection_name = f"run_{run_id.replace('-', '_')}"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": len(docs), "fetch_k": len(docs)},
    )
    relevant_docs = retriever.invoke(query)
    rag_context = "\n\n".join(d.page_content for d in relevant_docs)

    real_pmids = {p["pmid"] for p in screened}
    valid_list = ", ".join(sorted(real_pmids))
    sub_q_text = "\n".join(f"- {q}" for q in sub_questions)

    extraction_table = "\n".join(
        f"- PMID {e['pmid']}: Dataset={e.get('dataset')} | N={e.get('sample_size')} | "
        f"Validation={e.get('validation')} | Model={e.get('model')} | "
        f"Metrics={e.get('metrics')} | Findings={e.get('key_findings')}"
        for e in extractions
    )
    gaps_block = "\n".join(f"- {g}" for g in research_gaps)

    base_prompt = f"""You are writing a detailed, source-grounded medical literature review.

STRICT RULES — VIOLATING ANY RULE INVALIDATES THE REVIEW:
1. Every factual sentence MUST end with a citation of the form [PMID: <id>]
   where <id> is one of these valid PMIDs: {valid_list}
2. You MUST cite at least one PMID in every section.
3. NEVER invent a PMID. If the retrieved context does not support a claim,
   omit the claim entirely.
4. Do NOT reference any paper outside the Retrieved Context or Structured Evidence.
5. Use this markdown structure exactly:
   ## Overview
   ## Datasets and Populations
   ## Methods and Models
   ## Validation and Evaluation Metrics
   ## Key Findings
   ## Research Gaps
   ## Conclusion

DEPTH AND LENGTH REQUIREMENTS (mandatory):
- Each section must be AT LEAST 3 substantial paragraphs.
- Target total length: 1200-2000 words.
- Cite at least 3 distinct PMIDs across the review.

Research question: {query}

Sub-questions to address:
{sub_q_text}

Retrieved Context:
{rag_context}

Structured Evidence (more reliable than free text):
{extraction_table}

Pre-identified Research Gaps:
{gaps_block}
"""

    # ---- THE ONLY SMART-MODEL CALL ----
    synthesis = safe_llm_invoke(base_prompt, model="smart", use_cache=True)
    is_grounded, reason, details = _check_grounding(synthesis, real_pmids)

    if not is_grounded:
        if reason == "hallucinated":
            print(f"[SYNTHESIS] Rejected: hallucinated PMIDs {sorted(details)}. Retrying...")
            reason_line = (
                f"Your previous draft cited these PMIDs that DO NOT EXIST: {sorted(details)}. "
                f"You may ONLY cite PMIDs from this list: {valid_list}"
            )
        else:
            print("[SYNTHESIS] Rejected: no [PMID: ...] citations found. Retrying...")
            reason_line = (
                "Your previous draft contained ZERO [PMID: ...] citations. "
                f"Every factual sentence MUST end with [PMID: <id>] using one of: {valid_list}"
            )

        retry_prompt = f"""{base_prompt}

--- PREVIOUS DRAFT REJECTED ---
{reason_line}

Rewrite the entire review from scratch. Include at least 3 distinct PMID citations.
"""
        retry = safe_llm_invoke(retry_prompt, model="smart", use_cache=False)
        if retry.strip():
            synthesis = retry
            _, retry_reason, _ = _check_grounding(retry, real_pmids)
            print(f"[SYNTHESIS] Retry result: {retry_reason}")

    return {
        "run_id": run_id,
        "synthesis": synthesis,
        "references": [
            {
                "pmid": p["pmid"],
                "title": p["title"],
                "authors": p.get("authors", "Unknown Authors"),
                "journal": p.get("journal", ""),
                "pubdate": p.get("pubdate", "N/A"),
            }
            for p in screened
        ],
        "status": "Completed",
    }


# =============================================================================
# 7. REVIEWER AGENT
# =============================================================================
def reviewer_node(state: AgentState) -> Dict[str, Any]:
    synthesis = state.get("synthesis", "")
    references = state.get("references", [])
    real_pmids = {r["pmid"] for r in references}

    is_grounded, reason, details = _check_grounding(synthesis, real_pmids)

    if not references:
        status = "Approved — no references to verify"
    elif reason == "hallucinated":
        status = f"REJECTED — hallucinated PMIDs cited: {sorted(details)}"
    elif reason == "no_citations":
        status = "REJECTED — synthesis contains no PMID citations"
    else:
        status = f"Approved — {len(details)} unique PMIDs cited, all grounded"

    return {"review_status": status, "status": "Completed"}


# =============================================================================
# Graph wiring
# =============================================================================
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("screener", screening_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("comparator", comparison_node)
workflow.add_node("synthesizer", synthesis_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "screener")
workflow.add_edge("screener", "extractor")
workflow.add_edge("extractor", "comparator")
workflow.add_edge("comparator", "synthesizer")
workflow.add_edge("synthesizer", "reviewer")
workflow.add_edge("reviewer", END)

app_pipeline = workflow.compile()