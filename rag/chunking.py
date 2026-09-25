"""
Heading-aware Markdown chunking.

Policy docs are short and well structured, so each `##` section becomes one
chunk. Every chunk is prefixed with "<doc title> > <section>" (contextual
header) so it is self-explanatory when retrieved alone. Oversized sections are
split on paragraph boundaries with one paragraph of overlap.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 1200


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    content: str  # includes contextual header
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return hashlib.sha256(f"{self.doc_id}|{self.content}".encode()).hexdigest()[:32]

    @property
    def citation(self) -> str:
        return f"{self.doc_title} > {self.section}"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def _split_long(body: str) -> list[str]:
    if len(body) <= MAX_CHARS:
        return [body]
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    parts, cur = [], []
    for p in paras:
        if cur and len("\n\n".join(cur + [p])) > MAX_CHARS:
            parts.append("\n\n".join(cur))
            cur = [cur[-1]]  # overlap
        cur.append(p)
    if cur:
        parts.append("\n\n".join(cur))
    return parts


def chunk_markdown(text: str, doc_id: str) -> list[Chunk]:
    meta, body = parse_frontmatter(text)
    title = meta.get("title") or doc_id
    base_meta = {"doc_type": meta.get("doc_type", "unknown"),
                 "applies_to": meta.get("applies_to", "all")}

    chunks: list[Chunk] = []
    for sec in re.split(r"^## ", body, flags=re.M)[1:]:
        heading, _, sec_body = sec.partition("\n")
        heading, sec_body = heading.strip(), sec_body.strip()
        if not sec_body:
            continue
        for part in _split_long(sec_body):
            content = f"{title} > {heading}\n\n{part}"
            chunks.append(Chunk(doc_id, title, heading, content, dict(base_meta)))
    return chunks


def chunk_directory(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for f in sorted(path.glob("*.md")):
        out.extend(chunk_markdown(f.read_text(encoding="utf-8"), f.stem))
    return out
