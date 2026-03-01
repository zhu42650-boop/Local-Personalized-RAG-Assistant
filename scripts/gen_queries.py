#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict, Iterable, List, Set

import requests


DEFAULT_SYSTEM_PROMPT = (
    "You are a scientific search assistant. Generate short retrieval queries "
    "that a researcher would type to find the passage. Avoid copying sentences. "
    "Use academic phrasing. Return only the queries."
)


def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_existing_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    ids = set()
    for row in _read_jsonl(path):
        chunk_id = row.get("chunk_id")
        if chunk_id is not None:
            ids.add(str(chunk_id))
    return ids


def _build_prompt(text: str, num_queries: int) -> str:
    return (
        "Generate {n} concise search queries for the following passage. "
        "Do not quote the passage. Focus on key methods, tasks, or findings.\n\n"
        "Passage:\n{text}\n\n"
        "Output format: one query per line."
    ).format(n=num_queries, text=text)


def _call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 429:
        raise requests.HTTPError("Rate limited", response=resp)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _parse_queries(text: str) -> List[str]:
    lines = [line.strip(" -\t") for line in text.splitlines()]
    queries = [line for line in lines if line]
    return queries


def generate_queries(
    chunks_path: str,
    output_path: str,
    base_url: str,
    api_key: str,
    model: str,
    num_queries: int,
    max_chunks: int,
    sleep_seconds: float,
    temperature: float,
    max_tokens: int,
    timeout: int,
    system_prompt: str,
    max_retries: int,
    backoff_base: float,
) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed = _load_existing_ids(output_path)
    total = 0

    with open(output_path, "a", encoding="utf-8") as out:
        for chunk in _read_jsonl(chunks_path):
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id in processed:
                continue
            text = chunk.get("text", "").strip()
            if not text:
                continue

            prompt = _build_prompt(text, num_queries)
            raw = None
            for attempt in range(max_retries + 1):
                try:
                    raw = _call_chat_completion(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    break
                except requests.RequestException as exc:
                    if getattr(exc, "response", None) is not None and exc.response is not None:
                        status = exc.response.status_code
                        if status == 429 and attempt < max_retries:
                            wait = backoff_base * (2**attempt)
                            print(f"[WARN] chunk {chunk_id} rate limited, retry in {wait:.1f}s")
                            time.sleep(wait)
                            continue
                    print(f"[WARN] chunk {chunk_id} failed: {exc}")
                    raw = None
                    break

            if raw is None:
                time.sleep(sleep_seconds)
                continue

            queries = _parse_queries(raw)
            if not queries:
                continue

            for q in queries[:num_queries]:
                record = {
                    "query": q,
                    "positive": text,
                    "paper_id": chunk.get("paper_id"),
                    "chunk_id": chunk.get("chunk_id"),
                }
                out.write(json.dumps(record, ensure_ascii=True) + "\n")

            processed.add(chunk_id)
            total += 1
            if max_chunks and total >= max_chunks:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training queries for chunks via LLM API.")
    parser.add_argument("--chunks", default="data/chunks.jsonl", help="Input chunks JSONL.")
    parser.add_argument("--output", default="data/pairs.jsonl", help="Output pairs JSONL.")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", ""), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="API key.")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"), help="Model name.")
    parser.add_argument("--num-queries", type=int, default=2, help="Queries per chunk.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit number of chunks (0=all).")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max tokens for completion.")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout seconds.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt override.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries on 429 errors.")
    parser.add_argument("--backoff-base", type=float, default=1.5, help="Backoff base seconds.")
    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        raise SystemExit("Missing --base-url or --api-key (or set LLM_BASE_URL/LLM_API_KEY).")

    count = generate_queries(
        chunks_path=args.chunks,
        output_path=args.output,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        num_queries=args.num_queries,
        max_chunks=args.max_chunks,
        sleep_seconds=args.sleep,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        system_prompt=args.system_prompt,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
    )
    print(f"Processed {count} chunks. Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
