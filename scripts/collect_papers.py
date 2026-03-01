#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from contextlib import suppress
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set

import feedparser
import requests


ARXIV_API = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API = "https://api.openalex.org/works"


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE)
    name = name.strip().replace(" ", "_")
    return name[:120] if len(name) > 120 else name


def _build_query(search: str, mode: str) -> str:
    # arXiv search syntax: all/ti/abs
    mode_map = {
        "all": "all",
        "title": "ti",
        "abstract": "abs",
        "title-abstract": "ti,abs",
    }
    prefix = mode_map.get(mode, "all")
    if prefix == "ti,abs":
        # arXiv does not support combined prefix directly; use OR
        return f"ti:{search} OR abs:{search}"
    return f"{prefix}:{search}"


def _fetch_batch(search_query: str, start: int, max_results: int) -> feedparser.FeedParserDict:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=(10, 30))
    resp.raise_for_status()
    return feedparser.parse(resp.text)


def _extract_pdf_url(entry: feedparser.FeedParserDict) -> Optional[str]:
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            return link.get("href")
        if link.get("title") == "pdf":
            return link.get("href")
    return None


def _download_pdf(url: str, dest_path: str, timeout: int, max_retries: int) -> None:
    last_exc = None
    for _ in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=(10, timeout)) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
            return
        except requests.RequestException as exc:
            last_exc = exc
            with suppress(FileNotFoundError):
                os.remove(dest_path)
            time.sleep(2)
    if last_exc:
        raise last_exc


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _today() -> date:
    return datetime.utcnow().date()


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.lower()).strip()


def _select_pdf_url(location: Optional[dict]) -> Optional[str]:
    if not location or not isinstance(location, dict):
        return None
    return location.get("pdf_url") or None


def _dedupe_key(item: Dict) -> str:
    doi = item.get("doi")
    if doi:
        return f"doi:{doi.lower()}"
    title = item.get("title", "")
    return f"title:{_normalize_title(title)}"


def _iter_arxiv(
    query: str,
    since: Optional[date],
    sleep_seconds: float,
    query_mode: str,
) -> Iterable[Dict]:
    batch_size = 100
    search_query = _build_query(query, query_mode)
    start = 0
    while True:
        feed = _fetch_batch(search_query, start=start, max_results=batch_size)
        if not feed.entries:
            break

        for entry in feed.entries:
            pdf_url = _extract_pdf_url(entry)
            if not pdf_url:
                continue

            published_raw = entry.get("published")
            published_date = None
            if published_raw:
                published_date = datetime.strptime(published_raw[:10], "%Y-%m-%d").date()
            if since and published_date and published_date < since:
                continue

            arxiv_id = entry.get("id", "").split("/")[-1]
            title = entry.get("title", "").replace("\n", " ").strip()

            yield {
                "id": arxiv_id,
                "title": title,
                "pdf_url": pdf_url,
                "published": published_raw,
                "authors": [a.get("name") for a in entry.get("authors", [])],
                "summary": entry.get("summary", "").replace("\n", " ").strip(),
                "source": "arxiv",
                "doi": None,
            }

        start += batch_size
        time.sleep(sleep_seconds)


def _iter_semantic_scholar(
    query: str,
    since: Optional[date],
    year: Optional[str],
    min_citations: Optional[int],
    api_key: Optional[str],
    require_pdf: bool,
    oa_only: bool,
) -> Iterable[Dict]:
    params = {
        "query": query,
        "limit": 100,
        "fields": "title,authors,year,publicationDate,openAccessPdf,abstract,externalIds,venue,citationCount",
        "sort": "publicationDate:desc",
    }
    if year:
        params["year"] = year
    if since:
        params["publicationDateOrYear"] = f"{since.isoformat()}:{_today().isoformat()}"
    if min_citations is not None:
        params["minCitationCount"] = str(min_citations)
    if oa_only or require_pdf:
        params["openAccessPdf"] = "true"

    token = None
    while True:
        if token:
            params["token"] = token
        headers = {"x-api-key": api_key} if api_key else None
        resp = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        for paper in payload.get("data", []):
            pdf_info = paper.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url")
            if not pdf_url:
                continue
            title = (paper.get("title") or "").strip()
            doi = (paper.get("externalIds") or {}).get("DOI")
            yield {
                "id": paper.get("paperId"),
                "title": title,
                "pdf_url": pdf_url,
                "published": paper.get("publicationDate") or paper.get("year"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "summary": (paper.get("abstract") or "").strip(),
                "source": "semantic_scholar",
                "doi": doi,
            }

        token = payload.get("next")
        if not token:
            break


def _iter_openalex(
    query: str,
    since: Optional[date],
    mailto: Optional[str],
    oa_only: bool,
) -> Iterable[Dict]:
    page = 1
    per_page = 100
    filters = []
    if since:
        filters.append(f"from_publication_date:{since.isoformat()}")
    if oa_only:
        filters.append("is_oa:true")

    select_fields = [
        "id",
        "display_name",
        "publication_date",
        "best_oa_location",
        "primary_location",
        "open_access",
        "doi",
        "authorships",
    ]

    while True:
        params = {
            "search": query,
            "per-page": per_page,
            "page": page,
            "select": ",".join(select_fields),
        }
        if filters:
            params["filter"] = ",".join(filters)
        if mailto:
            params["mailto"] = mailto

        resp = requests.get(OPENALEX_API, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            pdf_url = _select_pdf_url(item.get("best_oa_location"))
            if not pdf_url:
                pdf_url = _select_pdf_url(item.get("primary_location"))
            if not pdf_url:
                oa = item.get("open_access") or {}
                pdf_url = oa.get("oa_url")
            if not pdf_url:
                continue

            authors = []
            for author in item.get("authorships", []) or []:
                author_name = (author.get("author") or {}).get("display_name")
                if author_name:
                    authors.append(author_name)

            yield {
                "id": item.get("id"),
                "title": (item.get("display_name") or "").strip(),
                "pdf_url": pdf_url,
                "published": item.get("publication_date"),
                "authors": authors,
                "summary": "",
                "source": "openalex",
                "doi": (item.get("doi") or "").replace("https://doi.org/", "") or None,
            }

        page += 1


def _split_terms(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def _count_occurrences(text: str, phrase: str) -> int:
    if not text or not phrase:
        return 0
    return text.lower().count(phrase.lower())


def _split_query_terms(query: str) -> List[str]:
    parts = re.split(r"[,\s]+", query.strip())
    return [p.lower() for p in parts if p]


def _matches_title_then_abstract(
    title: str,
    summary: str,
    query: str,
    min_abstract_hits: int,
    require_title_match: bool,
) -> bool:
    terms = _split_query_terms(query)
    if not terms:
        return True

    title_lower = title.lower()
    if require_title_match and not any(term in title_lower for term in terms):
        return False

    if min_abstract_hits <= 0:
        return True

    abstract_hits = 0
    summary_lower = summary.lower()
    for term in terms:
        abstract_hits += summary_lower.count(term)
    return abstract_hits >= min_abstract_hits


def collect_papers(
    query: str,
    max_papers: int,
    out_dir: str,
    sleep_seconds: float,
    sources: List[str],
    since: Optional[date],
    year: Optional[str],
    min_citations: Optional[int],
    oa_only: bool,
    mailto: Optional[str],
    require_pdf: bool,
    ss_api_key: Optional[str],
    download_timeout: int,
    max_retries: int,
    per_source_limit: Optional[int],
    log_every: int,
    match_query_min_hits: int,
    use_query_match_filter: bool,
    require_title_match: bool,
    fallback_abstract_only: bool,
    fallback_min_hits: int,
    query_mode: str,
) -> List[Dict]:
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")

    collected: List[Dict] = []
    seen: Set[str] = set()

    def build_iterators() -> List[Iterable[Dict]]:
        sources_iterators: List[Iterable[Dict]] = []
        if "arxiv" in sources:
            sources_iterators.append(_iter_arxiv(query, since, sleep_seconds, query_mode))
        if "semantic_scholar" in sources:
            sources_iterators.append(
                _iter_semantic_scholar(
                    query,
                    since,
                    year,
                    min_citations,
                    ss_api_key,
                    require_pdf=require_pdf,
                    oa_only=oa_only,
                )
            )
        if "openalex" in sources:
            sources_iterators.append(_iter_openalex(query, since, mailto, oa_only))
        return sources_iterators

    def run_pass(pass_name: str, title_required: bool, min_hits: int) -> None:
        nonlocal collected, seen
        print(f"Running pass: {pass_name} (title_required={title_required}, min_hits={min_hits})")
        for iterator in build_iterators():
            source_count = 0
            for item in iterator:
                if len(collected) >= max_papers:
                    return
                if per_source_limit is not None and source_count >= per_source_limit:
                    break
                if require_pdf and not item.get("pdf_url"):
                    continue

                key = _dedupe_key(item)
                if key in seen:
                    continue
                seen.add(key)

                title = item.get("title", "")
                summary = item.get("summary", "") or ""
                if use_query_match_filter:
                    if not _matches_title_then_abstract(
                        title, summary, query, min_hits, title_required
                    ):
                        continue

                safe_title = _sanitize_filename(title) or "paper"
                suffix = item.get("id") or str(len(collected) + 1)
                filename = f"{suffix}_{safe_title}.pdf"
                dest_path = os.path.join(out_dir, filename)

                if item.get("pdf_url") and not os.path.exists(dest_path):
                    _download_pdf(item["pdf_url"], dest_path, download_timeout, max_retries)

                item["file"] = dest_path
                collected.append(item)
                source_count += 1

                if log_every > 0 and len(collected) % log_every == 0:
                    print(f"Collected {len(collected)} papers so far...")

                with open(manifest_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")

    run_pass("primary", require_title_match, match_query_min_hits)

    if len(collected) < max_papers and use_query_match_filter and fallback_abstract_only:
        run_pass("fallback", title_required=False, min_hits=fallback_min_hits)

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect papers from arXiv by query.")
    parser.add_argument("--query", required=True, help="Research topic keywords.")
    parser.add_argument("--max-papers", type=int, default=50, help="Number of papers to download.")
    parser.add_argument("--out-dir", default="data/raw_papers", help="Output directory for PDFs.")
    parser.add_argument(
        "--sources",
        default="semantic_scholar,openalex,arxiv",
        help="Comma-separated sources: semantic_scholar,openalex,arxiv",
    )
    parser.add_argument(
        "--query-mode",
        default="all",
        choices=["all", "title", "abstract", "title-abstract"],
        help="arXiv search scope (all/title/abstract/title-abstract).",
    )
    parser.add_argument(
        "--query-min-hits",
        type=int,
        default=2,
        help="Require query terms to appear at least N times in abstract (after title match).",
    )
    parser.add_argument(
        "--use-query-filter",
        action="store_true",
        help="Enable filtering by query phrase hit count.",
    )
    parser.add_argument(
        "--require-title-match",
        action="store_true",
        help="Require any query term to appear in title (recommended).",
    )
    parser.add_argument(
        "--fallback-abstract-only",
        action="store_true",
        help="If not enough papers, relax to abstract-only match with --fallback-min-hits.",
    )
    parser.add_argument(
        "--fallback-min-hits",
        type=int,
        default=1,
        help="Fallback abstract hit threshold when --fallback-abstract-only is enabled.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include papers published on or after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--recent-years",
        type=int,
        default=2,
        help="Fallback window for recency when --since is not provided.",
    )
    parser.add_argument(
        "--year",
        default=None,
        help="Semantic Scholar year filter (e.g. 2024 or 2022-2025).",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=None,
        help="Semantic Scholar min citation count filter.",
    )
    parser.add_argument(
        "--oa-only",
        action="store_true",
        help="Only collect open-access papers (where supported).",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help="Contact email for OpenAlex polite pool.",
    )
    parser.add_argument(
        "--ss-api-key",
        default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        help="Semantic Scholar API key (or set SEMANTIC_SCHOLAR_API_KEY).",
    )
    parser.add_argument(
        "--allow-no-pdf",
        action="store_true",
        help="Allow metadata-only entries without PDFs.",
    )
    parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between API calls.")
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="Per-download read timeout seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for API and PDF download failures.",
    )
    parser.add_argument(
        "--per-source-limit",
        type=int,
        default=None,
        help="Max papers per source before moving to next source.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print progress every N downloads (0 to disable).",
    )
    args = parser.parse_args()

    since = _parse_date(args.since)
    if since is None and args.recent_years > 0:
        since = _today() - timedelta(days=365 * args.recent_years)

    collected = collect_papers(
        query=args.query,
        max_papers=args.max_papers,
        out_dir=args.out_dir,
        sleep_seconds=args.sleep,
        sources=[s.strip().lower() for s in args.sources.split(",") if s.strip()],
        since=since,
        year=args.year,
        min_citations=args.min_citations,
        oa_only=args.oa_only,
        mailto=args.mailto,
        require_pdf=not args.allow_no_pdf,
        ss_api_key=args.ss_api_key,
        download_timeout=args.download_timeout,
        max_retries=args.max_retries,
        per_source_limit=args.per_source_limit,
        log_every=args.log_every,
        match_query_min_hits=args.query_min_hits,
        use_query_match_filter=args.use_query_filter,
        require_title_match=args.require_title_match,
        fallback_abstract_only=args.fallback_abstract_only,
        fallback_min_hits=args.fallback_min_hits,
        query_mode=args.query_mode,
    )
    print(f"Collected {len(collected)} papers into {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
