"""Orchestration: turn one requested URL into stored, catalogued artefacts.

This is the seam between the *download* phase and future *processing* phases.
:func:`ingest_document` performs, in order:

    resolve (India Code) -> select English -> download -> validate -> hash
        -> store -> manifest

and returns a :class:`DownloadResult`. Later phases (text extraction, chunking,
embeddings) can be added as new functions that consume the manifest / stored
PDFs without touching this download path.

The corpus is **English-only**. The language gate sits *before* the download, so
a Hindi (or unverifiable) file is never fetched, never written to ``data/raw/``,
never given a ``metadata.json`` and never listed in ``data/manifest.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from . import config, indiacode, language
from .errors import IngestionError
from .http_client import download_stream
from .manifest import Manifest
from .models import BitstreamRef, DownloadResult, Outcome, ParsedItem
from .pdf_validation import validate_pdf
from .storage import DocumentStore
from .utils import (
    make_document_id,
    make_subordinate_document_id,
    sha256_file,
    utcnow_iso,
)

log = logging.getLogger(__name__)


def ingest_document(
    session: requests.Session,
    store: DocumentStore,
    manifest: Manifest,
    url: str,
    category: str,
    *,
    parent_url: str | None = None,
    all_bitstreams: bool = False,
    force: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> DownloadResult:
    """Ingest a single India Code URL into the corpus. Never raises for a
    per-document failure — errors are captured into a ``FAILED`` result so the
    caller can continue with the next URL.

    ``parent_url`` is the act page a Rules/Regulations file is listed on; it is
    what makes that file's language verifiable (see :mod:`ingestion.indiacode`).
    """
    try:
        return _ingest(
            session, store, manifest, url, category,
            parent_url=parent_url,
            all_bitstreams=all_bitstreams, force=force,
            skip_existing=skip_existing, dry_run=dry_run,
        )
    except IngestionError as exc:
        log.error("FAILED %s: %s", url, exc)
        return DownloadResult(
            source_url=url, outcome=Outcome.FAILED, message=str(exc),
            error_type=type(exc).__name__,
            http_status=getattr(exc, "status_code", None),
        )
    except Exception as exc:  # defensive: never let one URL kill the batch
        log.exception("Unexpected error while ingesting %s", url)
        return DownloadResult(
            source_url=url, outcome=Outcome.FAILED, message=f"Unexpected error: {exc}",
            error_type=type(exc).__name__,
        )


def _ingest(session, store, manifest, url, category, *, parent_url=None,
            all_bitstreams=False, force=False, skip_existing=False, dry_run=False):
    item: ParsedItem = indiacode.resolve(session, url, parent_url=parent_url)

    # English-only gate. Raises LanguageError (-> FAILED) when the item has no
    # English file or its language cannot be established; nothing is fetched.
    english: BitstreamRef = language.select_english_bitstream(item)
    document_id = _document_id_for(item, english)
    primary_url = english.url
    log.debug(
        "Selected English bitstream %s for %s (language_source=%s: %s)",
        english.filename, document_id, english.language_source, english.language_evidence,
    )

    existing = manifest.get(document_id)

    if dry_run:
        msg = (
            f"Would download English PDF {primary_url} "
            f"(language_source={english.language_source}) -> "
            f"{store.relpath(store.pdf_path(category, document_id))}"
        )
        log.info("DRY-RUN %s (%s)", document_id, msg)
        return DownloadResult(url, Outcome.DRY_RUN, document_id, category, message=msg)

    if existing and skip_existing and not force:
        log.info("SKIP %s (already in manifest; --skip-existing)", document_id)
        _touch_checked(manifest, existing)
        manifest.save()
        return DownloadResult(
            url, Outcome.UNCHANGED, document_id, category,
            sha256=existing.get("sha256"), pdf_path=existing.get("pdf_relpath"),
            message="Already present; skipped without downloading.",
        )

    document_dir = store.document_dir(category, document_id)
    document_dir.mkdir(parents=True, exist_ok=True)

    # --- download primary PDF to a .part file, then validate & hash -----------
    part = store.part_path(category, document_id)
    log.info("Downloading %s -> %s", primary_url, store.relpath(store.pdf_path(category, document_id)))
    try:
        fetched = download_stream(
            session, primary_url, part,
            # Used only if India Code's own lookup cannot resolve the file; the
            # bytes still have to be a PDF served from an India Code host.
            fallback_url=indiacode.showfile_url(primary_url),
        )
        content_type, nbytes = fetched.content_type, fetched.bytes
        validate_pdf(part, content_type=content_type, source_url=primary_url)
        new_sha = sha256_file(part)
    except Exception:
        # A failed document leaves nothing behind — not even the directory that
        # was made to stage its download into.
        part.unlink(missing_ok=True)
        _remove_if_empty(document_dir)
        raise

    now = utcnow_iso()

    # --- unchanged fast-path: identical content already stored ----------------
    # Do not re-write the PDF, metadata.json or version history; just record
    # that we checked. --force skips this short-circuit to refresh artefacts.
    if existing is not None and existing.get("sha256") == new_sha and not force:
        part.unlink(missing_ok=True)
        _touch_checked(manifest, existing)
        manifest.save()
        log.info("UNCHANGED %s (sha256=%s)", document_id, new_sha[:12])
        return DownloadResult(
            source_url=url, outcome=Outcome.UNCHANGED, document_id=document_id,
            category=category, sha256=new_sha, pdf_path=existing.get("pdf_relpath"),
            message=f"unchanged ({nbytes} bytes)", bytes=nbytes,
        )

    # --- decide new / unchanged(forced) / updated -----------------------------
    if existing is None:
        outcome = Outcome.NEW
        store.place_new_pdf(part, category, document_id)
    elif existing.get("sha256") == new_sha:
        # --force with identical content: re-place file and refresh metadata,
        # but this is not a new version.
        outcome = Outcome.UNCHANGED
        store.place_new_pdf(part, category, document_id)
    else:
        outcome = Outcome.UPDATED
        # Preserve the previous legal text before replacing it.
        prev_pdf = store.pdf_path(category, document_id)
        if prev_pdf.exists():
            archive = store.archive_current_pdf(category, document_id, existing["sha256"])
            log.info("Archived previous version -> %s", store.relpath(archive))
            _mark_previous_archived(existing, store.relpath(archive), now)
        store.place_new_pdf(part, category, document_id)

    # --- optional companion bitstreams (English only) -------------------------
    companion_status = {}
    if all_bitstreams:
        companion_status = _download_companions(
            session, store, item, category, document_id, selected=english
        )

    # --- update manifest entry + metadata.json --------------------------------
    entry = _build_manifest_entry(
        existing, store, item, category, document_id,
        primary_url=primary_url, sha256=new_sha, nbytes=nbytes,
        outcome=outcome, now=now, english=english, fetched=fetched,
    )
    manifest.upsert(entry)

    metadata = _build_metadata(
        item, category, document_id, entry,
        primary_url=primary_url, sha256=new_sha, nbytes=nbytes,
        fetched=fetched, companion_status=companion_status, now=now, english=english,
    )
    meta_path = store.write_metadata(category, document_id, metadata)
    entry["metadata_relpath"] = store.relpath(meta_path)
    manifest.upsert(entry)
    manifest.save()

    log.info("%s %s (sha256=%s)", outcome.value.upper(), document_id, new_sha[:12])
    return DownloadResult(
        source_url=url, outcome=outcome, document_id=document_id, category=category,
        sha256=new_sha, pdf_path=entry["pdf_relpath"],
        message=f"{outcome.value} ({nbytes} bytes)", bytes=nbytes,
    )


# --- helpers -------------------------------------------------------------------


def _document_id_for(item: ParsedItem, english: BitstreamRef) -> str:
    """The corpus identity of this document.

    Acts are identified by their DSpace handle. Rules and Regulations have no
    handle of their own, so they reuse the id discovery assigned them — keyed
    on the file URL — which keeps an inventory entry and its downloaded
    document one and the same document.
    """
    if item.subordinate is not None:
        return make_subordinate_document_id(
            item.subordinate.document_type,
            item.subordinate.description or item.title,
            english.url,
            fallback_name=english.filename,
        )
    return make_document_id(item.title, item.handle_id)


def _remove_if_empty(directory: Path) -> None:
    """Delete *directory* if nothing was written into it. Never recursive."""
    try:
        next(directory.iterdir())
    except StopIteration:
        try:
            directory.rmdir()
        except OSError:  # pragma: no cover - a racing writer keeps it
            pass
    except OSError:
        pass


def _touch_checked(manifest: Manifest, entry: dict) -> None:
    entry["last_checked_at"] = utcnow_iso()
    manifest.upsert(entry)


def _mark_previous_archived(entry: dict, archive_relpath: str, now: str) -> None:
    versions = entry.setdefault("versions", [])
    if versions:
        prev = versions[-1]
        prev["archived"] = True
        prev["archived_relpath"] = archive_relpath
        prev["archived_at"] = now


def _build_manifest_entry(existing, store, item, category, document_id, *,
                          primary_url, sha256, nbytes, outcome, now, english, fetched):
    pdf_relpath = store.relpath(store.pdf_path(category, document_id))
    document_type = config.CATEGORY_TO_DOCUMENT_TYPE.get(category)
    # Where the bytes actually came from, recorded only when it differs from the
    # India Code URL (it always does for Rules/Regulations, never for Acts).
    final_url = fetched.final_url if fetched.final_url != primary_url else None

    if existing is None:
        entry = {
            "document_id": document_id,
            "provider": "India Code",
            "source_url": item.source_url,
            "primary_pdf_url": primary_url,
            "handle": item.handle,
            "category": category,
            "document_type": document_type,
            "title": item.title,
            # How the PDF must be fetched: "bitstream" (Acts) or
            # "viewfileuploaded" (Rules/Regulations).
            "pdf_url_kind": indiacode.pdf_url_kind(primary_url),
            # Only English documents are ever stored (see ingestion.language).
            "language": language.CORPUS_LANGUAGE,
            "language_source": english.language_source,
            "sha256": sha256,
            "bytes": nbytes,
            "pdf_relpath": pdf_relpath,
            "metadata_relpath": None,  # filled in after metadata is written
            "first_downloaded_at": now,
            "last_updated_at": now,
            "last_checked_at": now,
            "version_count": 1,
            "versions": [
                {
                    "version": 1,
                    "sha256": sha256,
                    "bytes": nbytes,
                    "downloaded_at": now,
                    "pdf_relpath": pdf_relpath,
                    "archived": False,
                }
            ],
        }
        if final_url:
            entry["final_pdf_url"] = final_url
        if fetched.fallback_url:
            entry["fetched_via_fallback"] = True
        if item.subordinate is not None:
            entry["parent_handle"] = item.subordinate.parent_handle
            entry["parent_title"] = item.subordinate.parent_title
            entry["parent_url"] = item.subordinate.parent_url
        return entry

    entry = existing
    entry["last_checked_at"] = now
    entry["title"] = entry.get("title") or item.title
    entry["primary_pdf_url"] = primary_url
    entry["source_url"] = item.source_url
    entry["pdf_url_kind"] = indiacode.pdf_url_kind(primary_url)
    if final_url:
        entry["final_pdf_url"] = final_url
    if fetched.fallback_url:
        entry["fetched_via_fallback"] = True
    entry["language"] = language.CORPUS_LANGUAGE
    entry["language_source"] = english.language_source
    if outcome is Outcome.UPDATED:
        entry["sha256"] = sha256
        entry["bytes"] = nbytes
        entry["last_updated_at"] = now
        next_version = int(entry.get("version_count", len(entry.get("versions", [])))) + 1
        entry["version_count"] = next_version
        entry.setdefault("versions", []).append(
            {
                "version": next_version,
                "sha256": sha256,
                "bytes": nbytes,
                "downloaded_at": now,
                "pdf_relpath": pdf_relpath,
                "archived": False,
            }
        )
    return entry


def _build_metadata(item, category, document_id, entry, *, primary_url, sha256,
                    nbytes, fetched, companion_status, now, english):
    handle_url = f"{config.BASE_URL}/handle/{item.handle}"
    notes: list[str] = []

    # Merge the reliably-parsed fields; never invent absent keys.
    meta = {
        "schema_version": config.METADATA_SCHEMA_VERSION,
        "document_id": document_id,
        "provider": "India Code",
        "source_url": item.source_url,
        "india_code_handle_url": handle_url,
        "handle": item.handle,
        # The India Code URL of the PDF itself, exactly as India Code publishes
        # it (a ViewFileUploaded query string is never re-encoded).
        "primary_pdf_url": primary_url,
        "pdf_url_kind": indiacode.pdf_url_kind(primary_url),
        "category": category,
        "document_type": config.CATEGORY_TO_DOCUMENT_TYPE.get(category),
        "jurisdiction": "India",
        "title": item.title,
    }
    if item.original_url and item.original_url != primary_url:
        meta["requested_url"] = item.original_url

    # Rules/Regulations hang off a parent act rather than a DSpace item.
    if item.subordinate is not None:
        meta["parent_handle"] = item.subordinate.parent_handle
        meta["parent_title"] = item.subordinate.parent_title
        meta["parent_url"] = item.subordinate.parent_url
        if item.subordinate.india_code_path:
            meta["india_code_path"] = item.subordinate.india_code_path
        notes.append(
            "Subordinate legislation: India Code publishes this file on its "
            f"parent act's page ({item.subordinate.parent_url}), not as a "
            "DSpace item of its own."
        )

    # Curated metadata fields (present only if India Code provided them).
    for key in (
        "short_title", "long_title", "hindi_title", "act_number", "act_year",
        "enactment_date", "enforcement_date", "ministry", "ministry_relation",
        "india_code_act_id",
    ):
        if key in item.metadata and item.metadata[key] not in (None, ""):
            meta[key] = item.metadata[key]

    # Language: this corpus is English-only, so a stored document is English by
    # construction — recorded together with the rule that established it.
    meta["language"] = language.CORPUS_LANGUAGE
    meta["language_source"] = english.language_source
    meta["language_evidence"] = english.language_evidence
    notes.append(
        f"English-only corpus: {english.filename} was verified as English via "
        f"{english.language_source} ({english.language_evidence}). "
        "Non-English bitstreams are listed but never downloaded."
    )

    meta["download"] = {
        "downloaded_at": now,
        "sha256": sha256,
        "bytes": nbytes,
        "content_type": fetched.content_type,
        "http_status": fetched.http_status,
        # ``primary_pdf_url`` above stays exactly as India Code publishes it;
        # this is where the bytes were actually served from after redirects.
        "final_url": fetched.final_url,
        "redirects": list(fetched.redirects),
        "downloader": f"ingestion.download v{_version()}",
    }
    if fetched.fallback_url:
        meta["download"]["fallback_url"] = fetched.fallback_url
        notes.append(
            "India Code's /ViewFileUploaded lookup could not resolve this file "
            f"(it does not emit a usable redirect for the name {english.filename!r}), "
            "so the bytes were fetched from the address that lookup redirects to "
            "(see ingestion.indiacode.showfile_url). The citable India Code URL "
            "above is unchanged."
        )
    if fetched.redirects:
        notes.append(
            "India Code served this file after a redirect; the redirect target "
            "is repaired before it is followed when India Code leaves an "
            "unescaped '&' in the file name (see ingestion.utils.repair_location)."
        )

    # Every file the item offers is listed for traceability, each with the
    # language decision and why it was or was not taken. Non-English entries are
    # a *record* that they were excluded — their bytes are never fetched.
    meta["available_bitstreams"] = [
        {
            "url": b.url,
            "filename": b.filename,
            "sequence": b.sequence,
            "role": "selected_english" if b.url == english.url else "other",
            "language": b.language,
            "language_source": b.language_source,
            "language_evidence": b.language_evidence,
            "downloaded": b.url == english.url or companion_status.get(b.url, False),
            "excluded_reason": _exclusion_reason(b, english),
        }
        for b in item.bitstreams
    ]

    meta["version_history"] = entry.get("versions", [])
    meta["raw"] = {
        "dublin_core": item.dublin_core,
        "metadata_table": item.metadata_table,
    }
    meta["notes"] = notes
    meta["ingested_by"] = f"ingestion.download v{_version()}"
    meta["ingested_at"] = now
    return meta


def _exclusion_reason(ref: BitstreamRef, english: BitstreamRef) -> str | None:
    """Why a listed bitstream was not stored (``None`` if it was)."""
    if ref.url == english.url:
        return None
    if ref.language is None:
        return "language could not be determined from India Code metadata"
    if ref.language != language.CORPUS_LANGUAGE:
        return f"non-English ({ref.language}); this corpus is English-only"
    return "additional English file; not the selected primary text"


def _download_companions(
    session, store, item: ParsedItem, category, document_id, *, selected: BitstreamRef
) -> dict:
    """Best-effort download of *additional English* bitstreams.

    Returns ``{url: downloaded_bool}``. Hindi and unverifiable files are skipped
    outright — ``--all-bitstreams`` widens the selection within English, it
    never relaxes the language gate. Failures are logged but never abort the
    primary document.
    """
    results: dict[str, bool] = {}
    for ref in item.bitstreams:
        if ref.url == selected.url:
            continue
        if ref.language != language.CORPUS_LANGUAGE:
            log.info(
                "Skipping companion bitstream %s (language=%s); the corpus is "
                "English-only.", ref.url, ref.language or "undetermined",
            )
            continue
        suffix = f"en-seq{ref.sequence}" if ref.sequence else "en-extra"
        dest = store.document_dir(category, document_id) / f"{document_id}.{suffix}.pdf"
        part = dest.with_suffix(".pdf.part")
        try:
            companion = download_stream(session, ref.url, part)
            validate_pdf(part, content_type=companion.content_type, source_url=ref.url)
            part.replace(dest)
            ref.downloaded = True
            results[ref.url] = True
            log.info("Downloaded companion bitstream -> %s", store.relpath(dest))
        except Exception as exc:  # noqa: BLE001 - best effort
            part.unlink(missing_ok=True)
            results[ref.url] = False
            log.warning("Could not download companion bitstream %s: %s", ref.url, exc)
    return results


def _version() -> str:
    from . import __version__
    return __version__
