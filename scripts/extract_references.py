#!/usr/bin/env python3
"""Extract unique links from an HTML file and generate formatted academic references.

Usage:
    python scripts/extract_references.py <path-to-html-file>

For each link found in the HTML, the script queries the Semantic Scholar API to
retrieve paper metadata and prints a numbered reference list.  Two citation
formats are supported (conference preferred over arXiv preprint):

  Conference:
    [N] First Author, et al. ["Title."](arXiv-abstract-url). Conference Year.

  arXiv preprint:
    [N] First Author, et al. ["Title."](arXiv-abstract-url). arXiv preprint arXiv:XXXX.XXXXX (YYYY).
"""

import argparse
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
SEMANTIC_SCHOLAR_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"
FIELDS = "title,authors,venue,year,externalIds,publicationVenue"
REQUEST_TIMEOUT = 15  # seconds per request
RATE_LIMIT_DELAY = 3.5  # seconds between API calls (S2 free tier ≈ 1 req/s but be polite)

# Domains that are very unlikely to be academic papers – skip them early.
SKIP_DOMAINS = {
    "localhost",
    "127.0.0.1",
    "cdn.jsdelivr.net",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "gohugo.io",
    "github.com",           # repo pages, not papers
    "www.github.com",
    "yingqianwang.github.io",
    "mrtanke.github.io",
    "twitter.com",
    "x.com",
    "www.youtube.com",
    "youtube.com",
    "www.bilibili.com",
    "bilibili.com",
    "huggingface.co",
    "www.huggingface.co",
    # Company / product / blog sites – not papers
    "pinecone.io",
    "www.pinecone.io",
    "medium.com",
    "towardsdatascience.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "stackoverflow.com",
    "docs.python.org",
    "pytorch.org",
    "www.tensorflow.org",
    "keras.io",
    "colab.research.google.com",
    "kaggle.com",
    "www.kaggle.com",
    "wandb.ai",
    "neptune.ai",
    "mlflow.org",
    "paperswithcode.com",
    "www.paperswithcode.com",
}

# Domains to keep even though they look "non-paper-ish" at first glance.
PAPER_DOMAINS = {
    "arxiv.org",
    "www.arxiv.org",
    "doi.org",
    "dx.doi.org",
    "openreview.net",
    "www.openreview.net",
    "proceedings.neurips.cc",
    "papers.nips.cc",
    "proceedings.mlr.press",
    "aclanthology.org",
    "www.aclanthology.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "www.nature.com",
    "www.science.org",
    "openaccess.thecvf.com",
    "semanticscholar.org",
    "www.semanticscholar.org",
    "api.semanticscholar.org",
}

# ---------------------------------------------------------------------------
# HTML link extraction
# ---------------------------------------------------------------------------

class LinkExtractor(HTMLParser):
    """Simple HTML parser that collects all href values from <a> tags."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        if tag == "a":
            for attr_name, attr_value in attrs:
                if attr_name == "href" and attr_value:
                    self.links.append(attr_value)


def extract_links(html: str) -> list[str]:
    """Return deduplicated, ordered list of external http(s) links from *html*."""
    parser = LinkExtractor()
    parser.feed(html)
    seen: set[str] = set()
    unique: list[str] = []
    for link in parser.links:
        link = link.strip()
        if not link.startswith(("http://", "https://")):
            continue
        # Normalise trailing slashes for dedup but keep original form.
        normalised = link.rstrip("/")
        if normalised in seen:
            continue
        seen.add(normalised)
        unique.append(link)
    return unique


# ---------------------------------------------------------------------------
# Link → Semantic Scholar identifier mapping
# ---------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
_DOI_RE = re.compile(r"(?:doi\.org|dx\.doi\.org)/(.+)", re.I)
_S2_ID_RE = re.compile(r"semanticscholar\.org/paper/[^/]*/([0-9a-f]{40})", re.I)
_OPENREVIEW_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([A-Za-z0-9_-]+)", re.I)

# ACL Anthology: e.g. https://aclanthology.org/2023.nlposs-1.24/
_ACL_ANTHOLOGY_RE = re.compile(r"aclanthology\.org/([A-Za-z0-9._-]+?)/?$", re.I)

# IEEE Xplore: e.g. https://ieeexplore.ieee.org/document/9296658
_IEEE_RE = re.compile(r"ieeexplore\.ieee\.org/document/(\d+)", re.I)

# ACM DL: e.g. https://dl.acm.org/doi/10.1145/3292500.3330701
_ACM_DOI_RE = re.compile(r"dl\.acm\.org/doi/(10\..+)", re.I)

# Springer: e.g. https://link.springer.com/article/10.1007/...
_SPRINGER_DOI_RE = re.compile(r"link\.springer\.com/(?:article|chapter)/(10\..+)", re.I)

# CVF open-access: e.g. https://openaccess.thecvf.com/content/CVPR2023/papers/...
_CVF_RE = re.compile(r"openaccess\.thecvf\.com/", re.I)

# NeurIPS proceedings: e.g. https://proceedings.neurips.cc/paper_files/paper/2022/hash/...
_NEURIPS_RE = re.compile(r"proceedings\.neurips\.cc/", re.I)

# PMLR proceedings: e.g. https://proceedings.mlr.press/v162/chen22a.html
_PMLR_RE = re.compile(r"proceedings\.mlr\.press/", re.I)


def _link_to_query_id(url: str) -> str | None:
    """Convert a URL to a Semantic Scholar paper identifier string, or *None*
    if the URL doesn't look like a paper link we can resolve."""

    # arXiv
    m = _ARXIV_ID_RE.search(url)
    if m:
        aid = re.sub(r"v\d+$", "", m.group(1))  # strip version
        return f"arXiv:{aid}"

    # DOI via doi.org / dx.doi.org
    m = _DOI_RE.search(url)
    if m:
        doi = m.group(1).rstrip("/")
        return f"DOI:{doi}"

    # Semantic Scholar direct ID
    m = _S2_ID_RE.search(url)
    if m:
        return m.group(1)

    # OpenReview – use URL-based lookup
    m = _OPENREVIEW_RE.search(url)
    if m:
        return f"URL:{url}"

    # ACL Anthology – map to DOI  (10.18653/v1/<id>)
    m = _ACL_ANTHOLOGY_RE.search(url)
    if m:
        acl_id = m.group(1)
        return f"DOI:10.18653/v1/{acl_id}"

    # ACM Digital Library – extract embedded DOI
    m = _ACM_DOI_RE.search(url)
    if m:
        doi = m.group(1).rstrip("/")
        return f"DOI:{doi}"

    # Springer – extract embedded DOI
    m = _SPRINGER_DOI_RE.search(url)
    if m:
        doi = m.group(1).rstrip("/")
        return f"DOI:{doi}"

    # IEEE Xplore – use URL-based lookup (DOI not always in URL)
    m = _IEEE_RE.search(url)
    if m:
        return f"URL:{url}"

    # CVF open-access, NeurIPS proceedings, PMLR – URL-based lookup
    if _CVF_RE.search(url) or _NEURIPS_RE.search(url) or _PMLR_RE.search(url):
        return f"URL:{url}"

    # Fallback: try known paper domains.
    domain = urlparse(url).netloc.lower()
    if domain in PAPER_DOMAINS:
        return f"URL:{url}"

    return None


def _should_skip_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    # Strip port for localhost check
    host = domain.split(":")[0]
    return host in SKIP_DOMAINS


# ---------------------------------------------------------------------------
# Semantic Scholar API helpers
# ---------------------------------------------------------------------------

def _s2_get(url: str, params: dict) -> dict | None:
    """GET *url* with rate-limit retry.  Returns parsed JSON or None."""
    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            time.sleep(5)
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
        return None
    except requests.RequestException:
        return None


def _query_paper(query_id: str) -> dict | None:
    """Query S2 API by identifier and return the paper dict, or *None*."""
    url = f"{SEMANTIC_SCHOLAR_API}/{query_id}"
    params = {"fields": FIELDS}
    return _s2_get(url, params)


def _search_paper_by_title(title: str) -> dict | None:
    """Fallback: search S2 by title string and return the best match."""
    params = {"query": title, "limit": "1", "fields": FIELDS}
    data = _s2_get(SEMANTIC_SCHOLAR_SEARCH_API, params)
    if data and data.get("data"):
        return data["data"][0]
    return None


# ---------------------------------------------------------------------------
# CrossRef API helpers
# ---------------------------------------------------------------------------

def _crossref_lookup(doi: str) -> dict | None:
    """Query CrossRef for *doi* and return a normalised paper dict
    compatible with our format_reference(), or *None*."""
    url = f"{CROSSREF_API}/{doi}"
    try:
        resp = requests.get(
            url,
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        item = resp.json().get("message", {})
    except (requests.RequestException, ValueError):
        return None

    title_list = item.get("title", [])
    title = title_list[0] if title_list else None
    if not title:
        return None

    # Build an S2-compatible dict so format_reference works unchanged.
    authors = []
    for a in item.get("author", []):
        given = a.get("given", "")
        family = a.get("family", "")
        authors.append({"name": f"{given} {family}".strip()})

    # Determine year from published-print or published-online.
    year = None
    for key in ("published-print", "published-online", "created"):
        parts = (item.get(key) or {}).get("date-parts", [[]])[0]
        if parts:
            year = parts[0]
            break

    venue = ""
    for key in ("container-title", "short-container-title"):
        ct = item.get(key, [])
        if ct:
            venue = ct[0]
            break

    # Try to find an arXiv ID in the "alternative-id" or link fields.
    external_ids: dict = {}
    alt_ids = item.get("alternative-id", [])
    for aid in alt_ids:
        if re.match(r"\d{4}\.\d{4,5}", aid):
            external_ids["ArXiv"] = aid
            break

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "venue": venue,
        "publicationVenue": None,
        "externalIds": external_ids,
    }


# ---------------------------------------------------------------------------
# Scrape HTML <title> from a URL (lightweight fallback)
# ---------------------------------------------------------------------------

def _scrape_title(url: str) -> str | None:
    """Fetch *url* and extract the <title> text.  Returns None on failure."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={
            "User-Agent": "Mozilla/5.0 (compatible; RefExtractor/1.0)"
        })
        if resp.status_code != 200:
            return None
        m = re.search(r"<title[^>]*>([^<]+)</title>", resp.text, re.I)
        if m:
            import html as html_mod
            title = html_mod.unescape(m.group(1)).strip()
            # Many sites append " - ACL Anthology" etc.; strip that.
            title = re.sub(r"\s*[-|–]\s*(ACL Anthology|IEEE Xplore|Springer.*)$", "", title)
            return title
    except requests.RequestException:
        pass
    return None


# ---------------------------------------------------------------------------
# Reference formatting
# ---------------------------------------------------------------------------

# Venue names we treat as "real" conferences / journals.
# If the venue field contains one of these substrings (case-insensitive),
# we consider it a conference/journal citation instead of arXiv.
_VENUE_KEYWORDS = [
    "NeurIPS", "NIPS", "ICML", "ICLR", "AAAI", "IJCAI",
    "CVPR", "ICCV", "ECCV", "ACL", "EMNLP", "NAACL", "COLING",
    "SIGIR", "KDD", "WWW", "ICRA", "IROS", "RSS",
    "IEEE", "ACM", "Springer", "Nature", "Science",
    "TPAMI", "TIP", "TCSVT", "JMLR", "TMLR",
    "SIGGRAPH", "Transactions", "Journal", "Conference",
    "Symposium", "Workshop", "Proceedings",
    "AISTATS", "UAI", "COLT", "ISIT",
    "WACV", "BMVC", "ACCV", "3DV", "FG",
    "INTERSPEECH", "ICASSP",
]


def _is_conference_venue(venue: str | None, pub_venue: dict | None) -> bool:
    """Return True if *venue* looks like a real conference / journal."""
    names_to_check: list[str] = []
    if venue:
        names_to_check.append(venue)
    if pub_venue:
        for key in ("name", "alternate_names", "type"):
            val = pub_venue.get(key)
            if isinstance(val, str):
                names_to_check.append(val)
            elif isinstance(val, list):
                names_to_check.extend(str(v) for v in val)
    combined = " ".join(names_to_check).strip()
    if not combined:
        return False
    combined_lower = combined.lower()
    for kw in _VENUE_KEYWORDS:
        if kw.lower() in combined_lower:
            return True
    return False


def _short_venue(venue: str | None, pub_venue: dict | None) -> str:
    """Return a short venue name suitable for the citation."""
    # Prefer publicationVenue.name (usually cleaner).
    if pub_venue and pub_venue.get("name"):
        return pub_venue["name"]
    if venue:
        return venue
    return ""


def _first_author_surname(authors: list[dict]) -> str:
    if not authors:
        return "Unknown"
    name = authors[0].get("name", "Unknown")
    return name


def _arxiv_abstract_url(external_ids: dict) -> str | None:
    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


def format_reference(idx: int, paper: dict, original_url: str) -> str:
    """Return a single formatted reference line."""
    title = paper.get("title") or "Untitled"
    authors = paper.get("authors") or []
    year = paper.get("year")
    venue = paper.get("venue")
    pub_venue = paper.get("publicationVenue")
    external_ids = paper.get("externalIds") or {}

    first_author = _first_author_surname(authors)
    if len(authors) > 1:
        author_str = f"{first_author}, et al."
    else:
        author_str = f"{first_author}."

    # Determine the best link (prefer arXiv abstract).
    link = _arxiv_abstract_url(external_ids) or original_url

    if _is_conference_venue(venue, pub_venue):
        short = _short_venue(venue, pub_venue)
        year_str = f" {year}" if year else ""
        return f'[{idx}] {author_str} ["{title}."]({link}). {short}{year_str}.'
    else:
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id and year:
            return (
                f'[{idx}] {author_str} ["{title}."]({link}). '
                f"arXiv preprint arXiv:{arxiv_id} ({year})."
            )
        elif arxiv_id:
            return (
                f'[{idx}] {author_str} ["{title}."]({link}). '
                f"arXiv preprint arXiv:{arxiv_id}."
            )
        else:
            # Fallback – no arXiv ID, no recognised venue.
            year_str = f" ({year})" if year else ""
            v = _short_venue(venue, pub_venue) or "preprint"
            return f'[{idx}] {author_str} ["{title}."]({link}). {v}{year_str}.'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract links from an HTML file and generate formatted references."
    )
    parser.add_argument("html_file", type=str, help="Path to the HTML file.")
    parser.add_argument(
        "--delay",
        type=float,
        default=RATE_LIMIT_DELAY,
        help=f"Seconds between API calls (default: {RATE_LIMIT_DELAY}).",
    )
    args = parser.parse_args()

    html_path = Path(args.html_file)
    if not html_path.is_file():
        print(f"Error: file not found: {html_path}", file=sys.stderr)
        sys.exit(1)

    html = html_path.read_text(encoding="utf-8", errors="replace")
    links = extract_links(html)

    print(f"Found {len(links)} unique links in {html_path.name}.\n")

    references: list[str] = []
    ref_idx = 1

    for link in links:
        if _should_skip_domain(link):
            continue

        query_id = _link_to_query_id(link)
        if query_id is None:
            print(f"  [skip] {link}  (not a recognised paper URL)")
            continue

        print(f"  Querying: {link}  →  {query_id} ...", end=" ", flush=True)
        paper = _query_paper(query_id)

        # ── Fallback chain when primary S2 lookup fails ──────────────
        if paper is None or not paper.get("title"):
            # 1) If the query_id was a DOI, try CrossRef.
            if query_id.startswith("DOI:"):
                doi = query_id[4:]
                print("S2 miss → trying CrossRef ...", end=" ", flush=True)
                paper = _crossref_lookup(doi)
                time.sleep(1)

            # 2) Scrape the page <title> and search S2 by title.
            if paper is None or not paper.get("title"):
                print("trying title search ...", end=" ", flush=True)
                scraped = _scrape_title(link)
                if scraped:
                    paper = _search_paper_by_title(scraped)
                    time.sleep(args.delay)

        if paper is None or not paper.get("title"):
            print("NOT FOUND")
        else:
            ref = format_reference(ref_idx, paper, link)
            references.append(ref)
            print("OK")
            ref_idx += 1

        time.sleep(args.delay)

    print("\n" + "=" * 72)
    print("REFERENCES")
    print("=" * 72 + "\n")
    for ref in references:
        print(ref)
    print()


if __name__ == "__main__":
    main()
