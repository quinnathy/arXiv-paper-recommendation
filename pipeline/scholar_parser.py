"""Google Scholar profile parser for user onboarding.

Fetches paper metadata from a public Google Scholar profile URL in a
single HTTP request. Extracts title, authors, venue, citation count,
and publication year for each paper, then keeps a small representative
subset for embedding: the top cited papers and the most recent papers.

Returns a list of dicts compatible with EmbeddingModel.embed_papers().
"""

from __future__ import annotations

import re
from html import unescape
from urllib.parse import urlparse, parse_qs

import requests

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:
    BeautifulSoup = None


SCHOLAR_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def parse_scholar_url(url: str) -> str | None:
    """Extract the Google Scholar user ID from a profile URL.

    Args:
        url: A Google Scholar profile URL.

    Returns:
        The user ID string, or None if the URL is not a valid profile.
    """
    parsed = urlparse(url)
    if not parsed.hostname or "scholar.google" not in parsed.hostname:
        return None
    params = parse_qs(parsed.query)
    user_ids = params.get("user")
    return user_ids[0] if user_ids else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_int(text: str) -> int:
    """Parse an integer from text, returning 0 for empty or non-numeric."""
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _parse_rows_bs4(soup, max_papers: int) -> list[dict]:
    """Extract paper metadata from Scholar HTML using BeautifulSoup."""
    papers: list[dict] = []
    for row in soup.select("tr.gsc_a_tr")[:max_papers]:
        title_el = row.select_one("a.gsc_a_at")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)

        gray_divs = row.select("td.gsc_a_t div.gs_gray")
        authors = gray_divs[0].get_text(strip=True) if len(gray_divs) > 0 else ""
        venue = gray_divs[1].get_text(strip=True) if len(gray_divs) > 1 else ""

        cite_el = row.select_one("td.gsc_a_c")
        citations = _parse_int(cite_el.get_text(strip=True)) if cite_el else 0

        year_el = row.select_one("td.gsc_a_y span.gsc_a_h")
        year = _parse_int(year_el.get_text(strip=True)) if year_el else 0

        papers.append({
            "title": title,
            "authors": authors,
            "venue": venue,
            "citations": citations,
            "year": year,
            "abstract": "",
        })
    return papers


def _parse_rows_regex(html: str, max_papers: int) -> list[dict]:
    """Extract paper metadata from Scholar HTML using regex (fallback)."""
    papers: list[dict] = []
    row_matches = re.findall(
        r'<tr[^>]*class="[^"]*gsc_a_tr[^"]*"[^>]*>(.*?)</tr>',
        html,
        re.DOTALL,
    )
    for row_html in row_matches[:max_papers]:
        title_m = re.search(
            r'<a[^>]*class="[^"]*gsc_a_at[^"]*"[^>]*>(.*?)</a>', row_html
        )
        if not title_m:
            continue
        title = unescape(re.sub(r"<[^>]+>", "", title_m.group(1))).strip()

        gray = re.findall(
            r'<div[^>]*class="[^"]*gs_gray[^"]*"[^>]*>(.*?)</div>', row_html
        )
        authors = unescape(re.sub(r"<[^>]+>", "", gray[0])).strip() if len(gray) > 0 else ""
        venue = unescape(re.sub(r"<[^>]+>", "", gray[1])).strip() if len(gray) > 1 else ""

        cite_m = re.search(
            r'<td[^>]*class="[^"]*gsc_a_c[^"]*"[^>]*>(.*?)</td>', row_html, re.DOTALL
        )
        citations = _parse_int(cite_m.group(1)) if cite_m else 0

        year_m = re.search(
            r'<td[^>]*class="[^"]*gsc_a_y[^"]*"[^>]*>.*?(\d{4}).*?</td>',
            row_html,
            re.DOTALL,
        )
        year = int(year_m.group(1)) if year_m else 0

        papers.append({
            "title": title,
            "authors": authors,
            "venue": venue,
            "citations": citations,
            "year": year,
            "abstract": "",
        })
    return papers


def fetch_scholar_papers(
    user_id: str,
    max_papers: int = 30,
) -> list[dict]:
    """Scrape paper metadata from a Google Scholar profile.

    Makes exactly one HTTP request.

    Args:
        user_id: The Google Scholar user ID.
        max_papers: Maximum papers to retrieve. Default 30.

    Returns:
        Paper dicts with keys: title, authors, venue, citations (int),
        year (int), abstract ("").
    """
    url = (
        f"https://scholar.google.com/citations"
        f"?user={user_id}&cstart=0&pagesize={max_papers}&sortby=pubdate"
    )
    resp = requests.get(url, headers=SCHOLAR_HEADERS, timeout=15)
    resp.raise_for_status()

    if BeautifulSoup is not None:
        soup = BeautifulSoup(resp.text, "html.parser")
        papers = _parse_rows_bs4(soup, max_papers)
    else:
        papers = _parse_rows_regex(resp.text, max_papers)

    return papers


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_papers(
    papers: list[dict],
    max_n: int = 3,
) -> list[dict]:
    """Select top-cited and most-recent papers for embedding.

    The selector keeps up to ``max_n`` papers by citation count plus up to
    ``max_n`` papers by publication year. Overlaps are deduplicated by
    normalized title, preserving the top-cited group first.

    Args:
        papers: Paper dicts from :func:`fetch_scholar_papers`.
        max_n: Number of papers to take from each ranking. Default 3.

    Returns:
        Selected paper dicts, length <= ``2 * max_n``.
    """
    if not papers or max_n <= 0:
        return []

    def normalized_title(paper: dict) -> str:
        return re.sub(r"\s+", " ", str(paper.get("title", ""))).casefold().strip()

    def citations_key(item: tuple[int, dict]) -> tuple[int, int]:
        idx, paper = item
        return (int(paper.get("citations") or 0), -idx)

    def recency_key(item: tuple[int, dict]) -> tuple[int, int]:
        idx, paper = item
        return (int(paper.get("year") or 0), -idx)

    indexed = list(enumerate(papers))
    top_cited = [
        paper
        for _, paper in sorted(indexed, key=citations_key, reverse=True)[:max_n]
    ]
    top_recent = [
        paper
        for _, paper in sorted(indexed, key=recency_key, reverse=True)[:max_n]
    ]

    selected: list[dict] = []
    seen_titles: set[str] = set()
    for paper in top_cited + top_recent:
        title_key = normalized_title(paper)
        if title_key and title_key in seen_titles:
            continue
        if title_key:
            seen_titles.add(title_key)
        selected.append(paper)

    return selected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_scholar_papers(
    scholar_url: str,
    max_papers: int = 30,
) -> list[dict] | None:
    """End-to-end: URL -> filtered list of paper dicts.

    Args:
        scholar_url: A Google Scholar profile URL.
        max_papers: Maximum papers to scrape from the page.

    Returns:
        List of selected paper dicts, or None if the URL is
        invalid or scraping fails.  Each dict has at least "title" and
        "abstract" keys for compatibility with
        :meth:`EmbeddingModel.embed_papers`.
    """
    user_id = parse_scholar_url(scholar_url)
    if user_id is None:
        return None

    try:
        papers = fetch_scholar_papers(user_id, max_papers)
    except requests.RequestException:
        return None

    if not papers:
        return None

    return filter_papers(papers)
