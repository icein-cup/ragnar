import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, MatchAny,
)

from core.models import Chunk, SearchResult

NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class QdrantStore:
    def __init__(self, url: str, collection: str, dim: int = 1024,
                 client: QdrantClient | None = None):
        self.collection = collection
        self.dim = dim
        self._client = client or QdrantClient(url=url)

    def ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim,
                                            distance=Distance.COSINE),
            )

    def drop_collection(self) -> None:
        self._client.delete_collection(self.collection)

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        # Deterministic: re-upserting the same chunk overwrites rather than
        # duplicating.
        return str(uuid.uuid5(NAMESPACE, f"{chunk.doc_id}:{chunk.chunk_index}"))

    def upsert(self, chunks: list[Chunk],
               vectors: list[list[float]]) -> None:
        if not chunks:
            return
        points = [
            PointStruct(
                id=self._point_id(chunk),
                vector=vector,
                payload={
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page,
                    "sheet": chunk.sheet,
                    "is_table": chunk.is_table,
                    "low_confidence": chunk.low_confidence,
                    "is_summary": chunk.is_summary,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], limit: int,
               doc_ids: list[str] | None = None) -> list[SearchResult]:
        # doc_ids=None means unfiltered (search everything). An explicit
        # empty list means "nothing selected" - short-circuit rather than
        # ask Qdrant to match against zero ids, which is a degenerate query.
        if doc_ids is not None and not doc_ids:
            return []

        query_filter = None
        if doc_ids is not None:
            query_filter = Filter(must=[
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))
            ])

        hits = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points

        return [
            SearchResult(
                chunk=Chunk(
                    doc_id=h.payload["doc_id"],
                    filename=h.payload["filename"],
                    text=h.payload["text"],
                    chunk_index=h.payload["chunk_index"],
                    page=h.payload.get("page"),
                    sheet=h.payload.get("sheet"),
                    is_table=h.payload.get("is_table", False),
                    low_confidence=h.payload.get("low_confidence", False),
                    is_summary=h.payload.get("is_summary", False),
                ),
                score=h.score,
            )
            for h in hits
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ]),
        )

    def delete_stale(self, doc_id: str, keep_indices: set[int]) -> None:
        """Remove points for ``doc_id`` whose chunk index is not in ``keep_indices``.

        Because point IDs are deterministic (doc_id:chunk_index), upserting the
        new chunk set already overwrites any old chunk with the same index.
        This call deletes the old chunks whose indices are no longer present,
        without touching the newly-written points.

        Filtered server-side, not scrolled — a scroll+Python-diff approach
        caps out at whatever page limit is chosen and silently stops there.
        """
        if not keep_indices:
            self.delete_by_doc(doc_id)
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))],
                must_not=[FieldCondition(
                    key="chunk_index", match=MatchAny(any=sorted(keep_indices)),
                )],
            ),
        )
