"""
cleanup_orphaned_videos.py
===========================
One-time (or occasional) sweep for per-video folders left behind by a
FAILED attempt -- a stamp/slug folder that exists under TRANS_DIR,
MUT_DIR, and/or CAPTIONS_DIR but never got a mutations_*.csv written to
it, because the run died (e.g. download_audio()'s HTTP 403s, 2026-08)
somewhere between _video_slug() creating the folders and analyze()
actually producing mutation rows.

welsh_pipeline.py's per-video except blocks now call
corpus_ops.cleanup_incomplete_video_dirs() automatically going forward
(see the PATCH comment there), so this script is only needed to clear
out debris that accumulated BEFORE that fix existed. Safe to run any
time afterward too -- it will simply find nothing to do.

Identity: a video attempt is "orphaned" if its stamp/slug folder exists
under TRANS_DIR or CAPTIONS_DIR but MUT_DIR/<stamp>/<slug>/mutations_*.csv
does NOT exist. This deliberately mirrors cleanup_incomplete_video_dirs()'s
own test (mutations file present or not) rather than re-deriving a
different rule here.

NOT the same question as "0 mutations found for this video" -- a video
that completed normally but genuinely triggered no mutations still gets
a mutations_*.csv written (empty of rows, but present -- see the
`if data:` guard in welsh_pipeline.py, which only skips the _append call
entirely, and the "0 segments survived filtering" warning is a SEPARATE,
already-logged case). This script only ever removes folders where that
file was never created in the first place.

Dry-run by default, same convention as rerun_rules.py: prints exactly
what it would delete and why. Pass --commit to actually delete.

Usage:
    python cleanup_orphaned_videos.py                 # dry run
    python cleanup_orphaned_videos.py --commit         # actually delete
    python cleanup_orphaned_videos.py --commit --yes   # skip confirmation
"""
import argparse
import sys

from corpus_io import TRANS_DIR, MUT_DIR, CAPTIONS_DIR


def _stamp_slug_pairs(base_dir):
    """Yields (stamp, slug, folder_path) for every real stamp/slug folder
    under base_dir. Two levels deep only (stamp/slug), matching
    _video_slug()'s layout -- deliberately not a recursive glob, so a
    same-named stray folder some other tool created doesn't get swept in."""
    if not base_dir.exists():
        return
    for stamp_dir in sorted(base_dir.iterdir()):
        if not stamp_dir.is_dir():
            continue
        for slug_dir in sorted(stamp_dir.iterdir()):
            if slug_dir.is_dir():
                yield stamp_dir.name, slug_dir.name, slug_dir


def find_orphans():
    """Returns a dict of {(stamp, slug): [existing folder paths]} for every
    attempt that has a transcription and/or captions folder but no
    mutations CSV. Folders that don't exist for a given (stamp, slug) are
    simply omitted from that entry's list."""
    candidates = {}
    for base_dir in (TRANS_DIR, CAPTIONS_DIR):
        for stamp, slug, folder in _stamp_slug_pairs(base_dir):
            candidates.setdefault((stamp, slug), []).append(folder)

    orphans = {}
    for (stamp, slug), folders in candidates.items():
        mut_dir = MUT_DIR / stamp / slug
        has_mutations_csv = any(mut_dir.glob("mutations_*.csv")) if mut_dir.exists() else False
        if has_mutations_csv:
            continue
        all_folders = list(folders)
        if mut_dir.exists():
            all_folders.append(mut_dir)
        orphans[(stamp, slug)] = all_folders
    return orphans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commit", action="store_true",
                     help="actually delete the orphaned folders (default: dry run, just lists them)")
    ap.add_argument("--yes", action="store_true",
                     help="skip the confirmation prompt when --commit is given")
    args = ap.parse_args()

    orphans = find_orphans()
    if not orphans:
        print("No orphaned video folders found -- nothing to do.")
        return

    total_folders = sum(len(v) for v in orphans.values())
    print(f"Found {len(orphans)} incomplete attempt(s) across {total_folders} "
          f"folder(s) with no mutations CSV:\n")
    for (stamp, slug), folders in sorted(orphans.items()):
        print(f"  {stamp}/{slug}")
        for f in folders:
            print(f"    - {f}")

    if not args.commit:
        print(f"\nDry run only -- nothing deleted. Re-run with --commit to remove "
              f"these {total_folders} folder(s).")
        return

    if not args.yes:
        reply = input(f"\nDelete these {total_folders} folder(s)? This cannot be undone. "
                      f"[y/N]: ").strip().lower()
        if reply != "y":
            print("Cancelled -- nothing deleted.")
            return

    import shutil
    removed = 0
    for (stamp, slug), folders in orphans.items():
        for f in folders:
            if f.exists():
                shutil.rmtree(f, ignore_errors=True)
                removed += 1
    print(f"\nDeleted {removed} folder(s).")


if __name__ == "__main__":
    sys.exit(main())
