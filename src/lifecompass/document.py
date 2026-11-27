"""
The source document -- the single source of truth for one video.

It is what the speaker SAID, as heard from the audio: a list of `Unit`s (one paragraph each). A principle is a passage the
model copied from this text; the guard in principle_extractor.py checks that it really is one.

Pure Python, no I/O, no network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

DOC_VERSION = 3


@dataclass
class Unit:
    index: int
    text: str


@dataclass
class SourceDocument:
    video_id: str
    units: list
    stats: dict = field(default_factory=dict)
    version: int = DOC_VERSION

    @property
    def doc_hash(self) -> str:
        blob = "\n".join(u.text for u in self.units).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    def full_text(self) -> str:
        return "\n".join(u.text for u in self.units)

    def to_json(self) -> str:
        return json.dumps({
            "version": self.version,
            "video_id": self.video_id,
            "stats": self.stats,
            "units": [asdict(u) for u in self.units],
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, blob: str) -> "SourceDocument":
        d = json.loads(blob)
        return cls(
            video_id=d["video_id"],
            units=[Unit(u["index"], u["text"]) for u in d["units"]],
            stats=d.get("stats", {}),
            version=d.get("version", DOC_VERSION),
        )
