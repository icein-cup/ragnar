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

    @staticmethod
    def _stamped(filename: str, doc_id: str) -> str:
        """`report.pdf` + doc_id -> `report.1a2b3c4d.pdf`.

        doc_id is content-derived but a filename is not, so the stamp is what
        keeps two different files that happen to share a name apart on disk.
        """
        name = Path(filename)
        return f"{name.stem}.{doc_id[:8]}{name.suffix}"

    def archive(self, path: Path, doc_id: str,
                filename: str | None = None) -> Path:
        """Move a file out of the inbox into the originals archive.

        `filename` is the document's logical name as the registry knows it,
        defaulting to the path's own name. Passing it matters because `path`
        may already carry stage()'s stamp, and stamping a stamped name again
        would file the original somewhere archived_path() never looks.
        """
        target = self.archived_path(filename or path.name, doc_id)
        shutil.move(str(path), str(target))
        return target

    def archived_path(self, filename: str, doc_id: str) -> Path:
        """Where archive() put the original for this (filename, doc_id)."""
        return self.originals / self._stamped(filename, doc_id)

    def stage(self, filename: str, data: bytes) -> tuple[str, Path]:
        """Write uploaded bytes into the inbox; return (doc_id, path).

        The inbox name carries the content stamp. Staging under the bare
        filename meant two different documents both called `report.pdf`
        produced two registry rows but one inbox file — the second write
        clobbered the first, so the first doc_id was ingested from the second
        file's bytes and its citations pointed at the wrong original.
        """
        doc_id = hashlib.sha256(data).hexdigest()
        target = self.inbox / self._stamped(filename, doc_id)
        target.write_bytes(data)
        return doc_id, target

    def inbox_path(self, filename: str, doc_id: str) -> Path:
        """Where the worker should look for this document's source bytes.

        Prefers the stamped name written by stage(), and falls back to the
        bare filename for files dropped straight into the watched folder,
        which never went through stage() and so carry no stamp.
        """
        stamped = self.inbox / self._stamped(filename, doc_id)
        return stamped if stamped.exists() else self.inbox / filename

    def restore_to_inbox(self, filename: str, doc_id: str) -> bool:
        """Copy an archived original back into the inbox for re-ingestion.

        Returns False if the archived original is missing (e.g. removed by
        hand), so the caller can report which documents can't be re-chunked.
        """
        src = self.archived_path(filename, doc_id)
        if not src.exists():
            return False
        shutil.copy(str(src), str(self.inbox / self._stamped(filename, doc_id)))
        return True

    def write_converted(self, doc_id: str, markdown: str) -> None:
        (self.converted / f"{doc_id}.md").write_text(markdown, encoding="utf-8")

    def read_markdown(self, doc_id: str) -> str | None:
        path = self.converted / f"{doc_id}.md"
        return path.read_text(encoding="utf-8") if path.exists() else None

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
        (self.converted / f"{doc_id}.blocks.json").write_text(
            json.dumps(payload), encoding="utf-8")

    def read_parsed(self, doc_id: str) -> ParsedDocument | None:
        path = self.converted / f"{doc_id}.blocks.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        blocks = [Block(**b) for b in payload["blocks"]]
        return ParsedDocument(markdown="", blocks=blocks,
                              low_confidence=payload["low_confidence"])

    def pending_files(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())
