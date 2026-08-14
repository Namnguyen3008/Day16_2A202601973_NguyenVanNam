"""Shared provenance-safe evidence helpers for Agent Arena middleware."""

from __future__ import annotations


def line_supports(body: str, text: str) -> bool:
    """True only when *text* is a literal substring of one document line."""
    return (
        isinstance(body, str)
        and isinstance(text, str)
        and bool(text)
        and any(text in line for line in body.splitlines())
    )


def observed_docs(ctx):
    """Yield documents whose full body actually appeared in observed tool output."""
    corpus = getattr(ctx, "corpus", None)
    observed = getattr(ctx, "observed_text", "") or ""
    if corpus is None:
        return

    for doc in getattr(corpus, "docs", ()):
        body = getattr(doc, "body", "")
        if isinstance(body, str) and body and body in observed:
            yield doc


def observed_source_ids(ctx, text: str) -> list[str]:
    """Return fully observed document ids with single-line support for *text*."""
    if not isinstance(text, str) or not text:
        return []

    return [
        doc.doc_id
        for doc in observed_docs(ctx)
        if line_supports(getattr(doc, "body", ""), text)
    ]


def find_observed_source(
    ctx,
    text: str,
    *,
    exclude_doc_id: str | None = None,
) -> str | None:
    """Return the first fully observed source for *text*, optionally excluding one id."""
    for doc_id in observed_source_ids(ctx, text):
        if doc_id != exclude_doc_id:
            return doc_id
    return None


def citations_from_claims(claims) -> list[str]:
    """Build the informational citations list without touching claim text."""
    return sorted(
        {
            claim["doc_id"]
            for claim in claims
            if isinstance(claim, dict)
            and isinstance(claim.get("doc_id"), str)
            and claim["doc_id"]
        }
    )
