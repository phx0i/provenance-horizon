"""SEC EDGAR 10-K/10-Q download and risk-factor / MD&A section extraction."""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
FORMS = ("10-K", "10-Q")


@dataclass
class FilingRef:
    cik: int
    company: str
    form: str
    accession: str
    filing_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return SEC_ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document)

    @property
    def filing_id(self) -> str:
        return f"{self.cik}_{self.accession}"


@dataclass
class SectionRecord:
    filing_id: str
    cik: int
    form: str
    filing_date: str
    section: str
    text: str


class SECClient:
    def __init__(self, user_agent: str, min_interval: float = 0.11, timeout: int = 60):
        import requests

        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent of the form 'Org Name contact@example.org'")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self.timeout = timeout
        self._last = 0.0

    def get(self, url: str, retries: int = 4):
        from requests import RequestException

        err = ""
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(url, timeout=self.timeout)
            except RequestException as e:
                err = repr(e)
            else:
                if r.status_code == 200:
                    self._last = time.time()
                    return r
                err = f"status {r.status_code}"
            self._last = time.time()
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed: {err}")


def list_companies(client: SECClient) -> list[tuple[int, str]]:
    data = client.get(SEC_TICKERS_URL).json()
    seen, out = set(), []
    for v in data.values():
        cik = int(v["cik_str"])
        if cik not in seen:
            seen.add(cik)
            out.append((cik, v.get("title", "")))
    return out


def company_filings(client: SECClient, cik: int, company: str, years: tuple[int, int],
                    forms: Iterable[str] = FORMS) -> list[FilingRef]:
    sub = client.get(SEC_SUBMISSIONS_URL.format(cik=cik)).json()
    rec = sub.get("filings", {}).get("recent", {})
    out = []
    for form, acc, date, doc in zip(rec.get("form", []), rec.get("accessionNumber", []),
                                    rec.get("filingDate", []), rec.get("primaryDocument", [])):
        if form in forms and years[0] <= int(date[:4]) <= years[1] and doc:
            out.append(FilingRef(cik=cik, company=company, form=form, accession=acc,
                                 filing_date=date, primary_document=doc))
    out.sort(key=lambda f: f.filing_date, reverse=True)
    return out


def sample_filings(client: SECClient, n_filings: int, years: tuple[int, int] = (2020, 2025),
                   forms: Iterable[str] = FORMS, max_per_company: int = 2, seed: int = 0,
                   log=print) -> list[FilingRef]:
    """Seeded sample of up to max_per_company qualifying filings per company until n_filings are collected."""
    companies = list_companies(client)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(companies))
    refs: list[FilingRef] = []
    for i in order:
        cik, name = companies[i]
        try:
            fs = company_filings(client, cik, name, years, forms=forms)
        except Exception as e:
            log(f"skip CIK {cik}: {e}")
            continue
        refs.extend(fs[:max_per_company])
        if len(refs) >= n_filings:
            break
    return refs[:n_filings]


def html_to_text(html: str) -> str:
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    for t in soup.find_all(_BLOCK_TAGS):
        t.insert_before("\n")
        t.insert_after("\n")
    for t in soup.find_all(["td", "th"]):
        t.insert_after(" ")
    text = soup.get_text("")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


_BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol",
               "section", "article", "hr", "pre", "blockquote"]


_ITEM = r"item\s*{num}(?![0-9a-z])\s*[\.:\-–—]?"


def _heading_positions(text: str, num: str) -> list[int]:
    """Start offsets of lines beginning with the item heading."""
    pat = re.compile(r"(?im)^[ \t]*" + _ITEM.format(num=num) + r"[ \t]*(?:\S.{0,300})?$")
    return [m.start() for m in pat.finditer(text)]


def _heading_title(text: str, start: int, num: str) -> str:
    """Title of the item heading at start: the rest of its line, or the next line when that rest is empty."""
    lines = text[start:start + 350].split("\n")
    title = re.sub(r"(?i)^[ \t]*" + _ITEM.format(num=num), "", lines[0]).strip()
    if not title and len(lines) > 1:
        title = lines[1]
    return title.lower()


def _item_span(text: str, start_num: str, end_nums: list[str], min_chars: int = 500,
               start_require: Optional[str] = None) -> Optional[str]:
    starts = _heading_positions(text, start_num)
    if start_require:
        terms = start_require.lower().split("|")
        starts = [s for s in starts if any(t in _heading_title(text, s, start_num) for t in terms)]
    ends = sorted(p for n in end_nums for p in _heading_positions(text, n))
    for s in reversed(starts):
        e = next((p for p in ends if p > s), None)
        if e is None:
            continue
        span = text[s:e].strip()
        if len(span) < min_chars:
            continue
        if any(p > 0 for p in _heading_positions(text[s:e], start_num)):
            continue
        return span
    return None


def extract_sections(html: str, form: str) -> dict[str, str]:
    text = html_to_text(html)
    out: dict[str, str] = {}
    if form.startswith("10-K"):
        rf = _item_span(text, "1A", ["1B", "1C", "2"])
        md = _item_span(text, "7", ["7A", "8"])
    else:
        rf = _item_span(text, "1A", ["2", "3", "4", "5", "6"])
        md = _item_span(text, "2", ["3", "4"], start_require="management|discussion")
    if rf:
        out["risk_factors"] = rf
    if md:
        out["mdna"] = md
    return out


def download_and_extract(client: SECClient, refs: Iterable[FilingRef], out_path: Path,
                         raw_dir: Optional[Path] = None, log=print) -> int:
    """Downloads each primary document and appends its section records to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = Path(str(out_path) + ".done")
    n = 0
    with out_path.open("a", encoding="utf-8") as fh, done_path.open("a", encoding="utf-8") as done:
        for ref in refs:
            try:
                html = client.get(ref.url).text
            except Exception as e:
                log(f"skip {ref.filing_id}: {e}")
                continue
            if raw_dir is not None:
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{ref.filing_id}.html").write_text(html, encoding="utf-8")
            secs = extract_sections(html, ref.form)
            for name, txt in secs.items():
                rec = SectionRecord(filing_id=ref.filing_id, cik=ref.cik, form=ref.form,
                                    filing_date=ref.filing_date, section=name, text=txt)
                fh.write(json.dumps(asdict(rec)) + "\n")
                n += 1
            done.write(ref.filing_id + "\n")
            log(f"{ref.filing_id} {ref.form} {ref.filing_date}: sections={list(secs)}")
    return n


def processed_filings(out_path: Path) -> set[str]:
    p = Path(str(out_path) + ".done")
    return {line.strip() for line in p.read_text().splitlines() if line.strip()} if p.exists() else set()


def read_sections(path: Path) -> Iterator[SectionRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield SectionRecord(**json.loads(line))
