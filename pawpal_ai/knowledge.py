"""Knowledge base loading and retrieval (the "R" in RAG).

The knowledge base is a folder of Markdown files (``knowledge/*.md``). Each file has a
small front-matter header (title, species) and is split into one chunk per ``## `` section.
Sections may contain ``TASK:`` lines, which are machine-readable task suggestions that the
planner can turn into real :class:`pawpal_system.Task` objects.

Retrieval is a dependency-free TF-IDF + cosine-similarity search with a species boost, so
it runs anywhere, is deterministic, and is easy to test and explain.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "knowledge"

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it
    its me my of on or our should so than that the their them then there these they this
    to too up was we what when where which who why will with you your yours about need
    needs much many often day days get give per""".split()
)
TASK_LINE = re.compile(r"^TASK:\s*(.+)$", re.MULTILINE)
# Query expansion: everyday words -> the species names used in the knowledge base.
SYNONYMS = {"puppy": "dog", "pup": "dog", "kitten": "cat", "kitty": "cat", "bunny": "rabbit",
            "parrot": "bird", "parakeet": "bird", "budgie": "bird", "cockatiel": "bird"}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop stopwords, and apply a tiny plural stemmer."""
    words = re.findall(r"[a-z]+", text.lower())
    out = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"  # puppies -> puppy
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # walks -> walk
        out.append(w)
    return out


@dataclass(frozen=True)
class TaskTemplate:
    """A suggested task parsed from a ``TASK:`` line in the knowledge base."""

    description: str
    time: str
    duration_minutes: int
    priority: str
    frequency: str

    @classmethod
    def parse(cls, line: str) -> TaskTemplate:
        """Parse 'Description | HH:MM | minutes | priority | frequency'."""
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 5:
            raise ValueError(f"Malformed TASK line: {line!r}")
        desc, time, minutes, priority, frequency = parts
        return cls(desc, time, int(minutes), priority.lower(), frequency.lower())


@dataclass
class Chunk:
    """One retrievable section of a knowledge-base document."""

    id: str  # e.g. "dogs#exercise-and-walks"
    doc_title: str
    heading: str
    species: str  # "dog", "cat", ..., or "all"
    text: str  # prose only (TASK lines removed)
    tasks: list[TaskTemplate] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Human-readable citation label."""
        return f"{self.doc_title} > {self.heading}"


@dataclass
class RetrievedChunk:
    """A chunk plus its similarity score for one query."""

    chunk: Chunk
    score: float


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        for line in header.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        return meta, body
    return meta, raw


def load_chunks(kb_dir: str | Path = DEFAULT_KB_DIR) -> list[Chunk]:
    """Load every Markdown file in kb_dir and split it into section chunks."""
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_dir}")
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        title = meta.get("title", path.stem)
        species = meta.get("species", "all").lower()
        for section in re.split(r"^## ", body, flags=re.MULTILINE)[1:]:
            heading, _, content = section.partition("\n")
            tasks = []
            for line in TASK_LINE.findall(content):
                try:
                    tasks.append(TaskTemplate.parse(line))
                except ValueError as exc:  # bad KB line: skip it, but say so
                    log.warning("Skipping %s in %s: %s", line, path.name, exc)
            prose = TASK_LINE.sub("", content).strip()
            chunks.append(Chunk(f"{path.stem}#{_slug(heading)}", title, heading.strip(),
                                species, prose, tasks))
    log.info("Loaded %d chunks from %s", len(chunks), kb_dir)
    return chunks


class Retriever:
    """TF-IDF retriever over knowledge-base chunks."""

    SPECIES_BOOST = 1.5  # chunks written for the pet's species rank higher
    OTHER_SPECIES_PENALTY = 0.3  # chunks for a *different* species rank much lower

    def __init__(self, chunks: list[Chunk]) -> None:
        """Index the chunks (heading words count double)."""
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        docs = [tokenize(f"{c.heading} {c.heading} {c.doc_title} {c.text}") for c in chunks]
        df = Counter(term for doc in docs for term in set(doc))
        n = len(docs)
        self.idf = {term: math.log((1 + n) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vectorize(doc) for doc in docs]

    @classmethod
    def from_folder(cls, kb_dir: str | Path = DEFAULT_KB_DIR) -> Retriever:
        """Build a retriever from a knowledge-base folder."""
        return cls(load_chunks(kb_dir))

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, species: str | None = None, k: int = 4,
               min_score: float = 0.0) -> list[RetrievedChunk]:
        """Return up to k chunks most similar to the query, best first."""
        tokens = tokenize(query)
        tokens += [SYNONYMS[t] for t in tokens if t in SYNONYMS]
        qvec = self._vectorize(tokens)
        species = species.lower() if species else None
        results = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = sum(w * vec.get(t, 0.0) for t, w in qvec.items())
            if species and chunk.species == species:
                score *= self.SPECIES_BOOST
            elif species and chunk.species not in ("all", species):
                score *= self.OTHER_SPECIES_PENALTY
            if score > min_score:
                results.append(RetrievedChunk(chunk, round(score, 4)))
        results.sort(key=lambda r: r.score, reverse=True)
        log.info("search(%r, species=%s) -> %s", query, species,
                 [(r.chunk.id, r.score) for r in results[:k]])
        return results[:k]

    def get(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by id."""
        return next((c for c in self.chunks if c.id == chunk_id), None)
