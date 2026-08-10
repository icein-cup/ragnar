import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ingestion.parser import Block, ParsedDocument


class Storage:
    """Owns the on-disk layout.

    data/ is the source of truth; Qdrant is a rebuildable index. Originals
    are never deleted automatically.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.originals = self.root / "originals"
        self.converted = self.root / "converted"
        for directory in (self.inbox, self.originals, self.converted):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def doc_id(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()

    def archive(self, path: Path, doc_id: str) -> Path:
        target = self.originals / f"{path.stem}.{doc_id[:8]}{path.suffix}"
        shutil.move(str(path), str(target))
        return target

    def archived_path(self, filename: str, doc_id: str) -> Path:
        """Where archive() put the original for this (filename, doc_id)."""
        name = Path(filename)
        return self.originals / f"{name.stem}.{doc_id[:8]}{name.suffix}"

    def restore_to_inbox(self, filename: str, doc_id: str) -> bool:
        """Copy an archived original back into the inbox for re-ingestion.

        Returns False if the archived original is missing (e.g. removed by
        hand), so the caller can report which documents can't be re-chunked.
        """
        src = self.archived_path(filename, doc_id)
        if not src.exists():
            return False
        shutil.copy(str(src), str(self.inbox / filename))
        return True

    def write_converted(self, doc_id: str, markdown: str) -> None:
        (self.converted / f"{doc_id}.md").write_text(markdown)

    def read_markdown(self, doc_id: str) -> str | None:
        path = self.converted / f"{doc_id}.md"
        return path.read_text() if path.exists() else None

    def remove_converted(self, doc_id: str) -> None:
        (self.converted / f"{doc_id}.md").unlink(missing_ok=True)
        (self.converted / f"{doc_id}.blocks.json").unlink(missing_ok=True)

    def write_parsed(self, doc_id: str, parsed: ParsedDocument) -> None:
        """Cache the parsed blocks so a chunking-only change can skip the
        (expensive) Docling parse. Markdown is not duplicated here — it's
        already on disk via write_converted, keyed by the same doc_id."""
        payload = {
            "low_confidence": parsed.low_confidence,
            "blocks": [asdict(b) for b in parsed.blocks],
        }
        (self.converted / f"{doc_id}.blocks.json").write_text(json.dumps(payload))

    def read_parsed(self, doc_id: str) -> ParsedDocument | None:
        path = self.converted / f"{doc_id}.blocks.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        blocks = [Block(**b) for b in payload["blocks"]]
        return ParsedDocument(markdown="", blocks=blocks,
                              low_confidence=payload["low_confidence"])

    def pending_files(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())
