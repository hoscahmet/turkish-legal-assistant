"""Fetch law texts from mevzuat.gov.tr and split them into one chunk per article.

Run: uv run python -m turkish_legal_assistant.ingest
"""

import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from .models import Article

# shortcut: repo-relative path, only works from a source checkout; move to config when deployed.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SOURCE_URL = "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo={n}&MevzuatTur=1&MevzuatTertip=5"
# The public page loads the full text from this iframe URL.
TEXT_URL = "https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=1&MevzuatNo={n}&MevzuatTertip=5"

# law number -> (name, article numbers to keep; None keeps all)
LAWS: dict[str, tuple[str, range | None]] = {
    "4857": ("İş Kanunu", None),
    "6698": ("Kişisel Verilerin Korunması Kanunu", None),
    "6502": ("Tüketicinin Korunması Hakkında Kanun", None),
    "6098": ("Türk Borçlar Kanunu", range(299, 379)),  # lease contracts section only
}

# "Madde 1 -", "MADDE 4/A- (1)", "GEÇİCİ MADDE 1 –", "Ek Madde 2-", "Madde 87 (Mülga...)"
HEADING_RE = re.compile(r"^(?:(GEÇİCİ|Geçici|EK|Ek) )?(?:MADDE|Madde) (\d+(?:/[A-Z])?)\b\s*[-–,]?\s*(.*)$")
SECTION_RE = re.compile(r"\b(KISIM|BÖLÜM|AYIRIM)$")
SECTION_LEVELS = ["KISIM", "BÖLÜM", "AYIRIM"]
PREFIXES = {"GEÇİCİ": "Geçici", "Geçici": "Geçici", "EK": "Ek", "Ek": "Ek"}
# The amendment table and footnotes after the last article start with e.g. "4857 SAYILI KANUNA ...".
END_RE = re.compile(r"^\d+ SAYILI KANUNA")
FOOTNOTE_RE = re.compile(r"\[\d+\]")


class _TextExtractor(HTMLParser):
    """Collect visible text, one line per block element."""

    BLOCKS = {"p", "br", "tr", "div", "li", "h1", "h2", "h3"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0  # depth inside <style>, <script> or a footnote reference

    def handle_starttag(self, tag, attrs):
        if self.skip or tag in ("style", "script") or ("class", "MsoFootnoteReference") in attrs:
            self.skip += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            # The source is Word-exported HTML: line breaks inside a paragraph are just wrapping.
            self.parts.append(re.sub(r"\s+", " ", data.replace("\xad", "")))


def html_to_lines(html: str) -> list[str]:
    parser = _TextExtractor()
    parser.feed(html)
    lines = (line.strip() for line in "".join(parser.parts).split("\n"))
    return [line for line in lines if line]


def _is_header(line: str) -> bool:
    """Titles and section names are short and have no sentence punctuation; body lines do."""
    return len(line) <= 150 and not line.startswith("(") and not line.endswith((".", ";", ":", ","))


def parse_articles(
    lines: list[str], law_number: str, law_name: str, keep: range | None = None
) -> list[Article]:
    articles: list[Article] = []
    sections: dict[int, str] = {}
    pending: list[str] = []  # lines seen since the last article heading
    current: dict | None = None

    def flush_body(body: list[str]) -> None:
        if current is not None:
            current["text"] = FOOTNOTE_RE.sub("", "\n".join(body)).strip()
            articles.append(Article(**current))

    for line in lines:
        if END_RE.match(line):
            break
        m = HEADING_RE.match(line)
        if not m:
            pending.append(line)
            continue

        # Trailing header-like lines belong to the next article (section names, title).
        split = len(pending)
        while split > 0 and _is_header(pending[split - 1]):
            split -= 1
        flush_body(pending[:split])
        headers = pending[split:]

        for i, h in enumerate(headers):
            if s := SECTION_RE.search(h):
                level = SECTION_LEVELS.index(s[1])
                sections = {k: v for k, v in sections.items() if k < level}
                name = headers[i + 1] if i + 1 < len(headers) else ""
                sections[level] = f"{h}: {name}" if name else h
        title = headers[-1] if headers and not SECTION_RE.search(headers[-1]) else ""

        prefix, number, rest = m.groups()
        label = f"{PREFIXES[prefix]} {number}" if prefix else number
        wanted = keep is None or (prefix is None and int(number.split("/")[0]) in keep)
        current = (
            dict(
                law_number=law_number,
                law_name=law_name,
                article_number=label,
                title=title,
                section=" > ".join(sections[k] for k in sorted(sections)),
                source_url=SOURCE_URL.format(n=law_number),
            )
            if wanted
            else None
        )
        pending = [rest] if rest else []

    flush_body(pending)
    return articles


def fetch_html(law_number: str) -> str:
    """Download the law text, caching the raw HTML under data/raw/."""
    cache = DATA_DIR / "raw" / f"{law_number}.html"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    req = urllib.request.Request(TEXT_URL.format(n=law_number), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(html, encoding="utf-8")
    return html


def main() -> None:
    articles: list[Article] = []
    for number, (name, keep) in LAWS.items():
        found = parse_articles(html_to_lines(fetch_html(number)), number, name, keep)
        print(f"{number} {name}: {len(found)} articles")
        articles += found
    out = DATA_DIR / "articles.json"
    out.write_text(
        json.dumps([a.model_dump() for a in articles], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"Wrote {len(articles)} articles to {out}")


if __name__ == "__main__":
    main()
