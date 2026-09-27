"""
corpus_layout.py
=================
Single source of truth for where saved article files live under an
output_dir -- written by icelandic_text_extractor.py's scrape_one(), read
by corpus_metadata.py's backfill. Stdlib only, so the migration CLI at the
bottom runs in any Python, same as corpus_metadata.py's.

Layout (variant first, then language):

    <output_dir>/raw/<lang>/<file>.txt
    <output_dir>/lemmatized/<lang>/<file>__lemma.txt

Before 2026-09-27 the nesting was the other way round
(<output_dir>/<lang>/raw/...). iter_raw_files() reads BOTH layouts, so a
half-migrated corpus is never silently under-counted; migrate_legacy_
layout() moves old-layout files into the new one. reconcile_manifest() and
clear_downloaded_articles() in the main script walk every .txt regardless
of nesting, so they never needed to know about either layout.
"""

import os

RAW = "raw"
LEMMATIZED = "lemmatized"
VARIANTS = (RAW, LEMMATIZED)


def variant_dir(output_dir, variant, lang_code):
    return os.path.join(output_dir, variant, lang_code)


def iter_raw_files(output_dir):
    """Yields (lang_code, path) for every raw .txt under output_dir, in
    either the current (raw/<lang>/) or legacy (<lang>/raw/) layout.
    Matched on the exact two-level path relative to output_dir, so nothing
    deeper or shallower is mistaken for article text."""
    for root, _dirs, files in os.walk(output_dir):
        parts = os.path.relpath(root, output_dir).split(os.sep)
        if len(parts) != 2:
            continue
        if parts[0] == RAW:
            lang_code = parts[1]
        elif parts[1] == RAW and parts[0] not in VARIANTS:
            lang_code = parts[0]
        else:
            continue
        for name in files:
            if name.endswith(".txt"):
                yield lang_code, os.path.join(root, name)


def _legacy_variant_dirs(output_dir):
    """Yields (lang_code, variant, path) for every legacy <lang>/<variant>/
    directory directly under output_dir."""
    if not os.path.isdir(output_dir):
        return
    for lang_code in sorted(os.listdir(output_dir)):
        lang_path = os.path.join(output_dir, lang_code)
        if lang_code in VARIANTS or not os.path.isdir(lang_path):
            continue
        for variant in VARIANTS:
            src_dir = os.path.join(lang_path, variant)
            if os.path.isdir(src_dir):
                yield lang_code, variant, src_dir


def has_legacy_layout(output_dir):
    return any(True for _ in _legacy_variant_dirs(output_dir))


def _remove_if_empty(path):
    try:
        os.rmdir(path)  # only ever succeeds on an empty directory
    except OSError:
        pass


def migrate_legacy_layout(output_dir, dry_run=False):
    """Moves <output_dir>/<lang>/{raw,lemmatized}/* to
    <output_dir>/{raw,lemmatized}/<lang>/*. Never overwrites: a file whose
    destination already exists stays where it is and is reported as a
    conflict. Legacy directories are removed only once empty, and only
    the ones this function actually moved files out of -- unrelated empty
    folders at the top level are left alone.

    Returns (moved_count, conflict_paths). dry_run=True reports what WOULD
    move without touching anything. Filenames are unchanged, so
    scraped_urls.txt and corpus_metadata.jsonl (both keyed by URL, not
    path) stay valid with no rewrite."""
    moved, conflicts = 0, []
    touched_lang_dirs = set()
    for lang_code, variant, src_dir in list(_legacy_variant_dirs(output_dir)):
        dst_dir = variant_dir(output_dir, variant, lang_code)
        for name in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dst_dir, name)
            if os.path.exists(dst):
                conflicts.append(src)
                continue
            if not dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                os.replace(src, dst)
            moved += 1
        if not dry_run:
            _remove_if_empty(src_dir)
        touched_lang_dirs.add(os.path.dirname(src_dir))
    if not dry_run:
        for lang_path in touched_lang_dirs:
            _remove_if_empty(lang_path)
    return moved, conflicts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Corpus folder layout maintenance (stdlib only)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    migrate_p = sub.add_parser("migrate", help="Move <lang>/raw|lemmatized/ files into raw|lemmatized/<lang>/")
    migrate_p.add_argument("--output-dir", required=True)
    migrate_p.add_argument("--dry-run", action="store_true", help="Report what would move without moving anything")
    args = parser.parse_args()

    if args.cmd == "migrate":
        moved, conflicts = migrate_legacy_layout(args.output_dir, dry_run=args.dry_run)
        verb = "Would move" if args.dry_run else "Moved"
        print(f"{verb} {moved} file(s).")
        if conflicts:
            print(f"{len(conflicts)} file(s) left in place -- destination already exists:")
            for c in conflicts:
                print(f"   {c}")
