# News Scraper

A general-purpose article scraper that works against most news sites out of the box, with an optional per-site tuning layer for sites you scrape a lot (evidence-backed — see **Site overrides** below). Two modes: auto-discover articles from a listing page, or scrape specific article URLs directly. Saved articles are automatically sorted into per-language subfolders by a local, offline 3-judge language-ID panel.

## Install

```bash
pip install -r requirements.txt
playwright install chromium
```

`requirements.txt` already includes Playwright, trafilatura, lingua, fasttext-wheel (Windows-friendly; use `fasttext` on Linux/macOS if you prefer), huggingface_hub, stanza, google-genai, and tqdm.

**Gemini API key** (only if you use `--detect-boilerplate` or `--detect-language-llm`):

```bash
export GEMINI_API_KEY=...   # free key: https://aistudio.google.com/apikey
# or rotate across multiple free-tier keys:
export GEMINI_API_KEYS="key1,key2,key3"
```

On Windows CMD use `set GEMINI_API_KEY=...` instead of `export`.

**Language-ID models:** the first time GlotLID or OpenLID-v3 runs, each downloads a ~1–2GB model from HuggingFace (one-time; needs internet). The scraper still runs if those packages are missing — the panel falls back to whichever judges are available (see **Language detection**).

## Usage

**No arguments — interactive menu:**

```bash
python icelandic_text_extractor.py
```

Walks you through: output directory → mode (auto-discover, direct URLs, scrape by language code, review/clear/delete) → URLs (or language codes, for that mode) → lemmatize y/n (and language code, if yes) → LLM boilerplate detection y/n → Gemini language-ID tiebreak y/n → runs the scrape with a live progress bar, then asks if you want to go again. Type `q` at any step to go back one step (or to quit, at the very first step).

**With arguments — same as before, for scripting/automation** (skips the menu entirely):

**Mode `auto`:** point it at a listing/section page (homepage, `/world`, `/sport`, etc.) and it discovers and scrapes article links on its own.

```bash
python icelandic_text_extractor.py auto https://example.com/news https://example.com/sport
```

**Mode `url`:** scrape specific article URLs directly, no discovery step.

```bash
python icelandic_text_extractor.py url https://example.com/news/some-headline-slug
```

**Mode 3 — scrape by language code(s):** interactive-menu-only (no CLI flag form yet). Type one or more ISO codes instead of pasting URLs; resolves to the listing page(s) of every `language_lock`'d domain in `LANGUAGE_OVERRIDES` that serves that code, skipping detection entirely for those domains since they're pre-verified to always resolve to a known language — see **Site overrides** below for exactly which codes are currently reachable this way.

**Mode `review`:** interactively walk through `boilerplate_candidates.json` (logged by `--detect-boilerplate`), see a suggested regex for each new fragment (auto-generalized once there are 2+ examples), and approve/skip each one — approved patterns are written straight into `boilerplate_patterns.py`.

```bash
python icelandic_text_extractor.py review --output-dir ./news_corpus
```

**Mode `clear`:** permanently deletes every saved article `.txt` file (raw + lemmatized, all language subfolders) under the given output dir and resets `scraped_urls.txt`. Confirms first unless `--yes` is given.

```bash
python icelandic_text_extractor.py clear --output-dir ./news_corpus
```

**Mode `delete`:** permanently deletes `boilerplate_candidates.json` for the given output dir (the LLM review-queue log — not `boilerplate_patterns.py` itself). Confirms first unless `--yes` is given.

```bash
python icelandic_text_extractor.py delete --output-dir ./news_corpus
```

## Options (CLI mode)

| Flag | Default | Applies to | Description |
| --- | --- | --- | --- |
| `--output-dir DIR` | `./news_corpus` | all modes | Where `.txt` files and the manifest are saved |
| `--lemmatize LANG` | off | `auto`, `url` | Stanza language code (e.g. `en`, `cy`, `is`) — lemmatizes text before saving |
| `--delay SECONDS` | `1.5` | `auto`, `url` | Pause between requests (+ random jitter), for politeness / rate-limit avoidance |
| `--detect-boilerplate` | off | `auto`, `url` | Use Gemini's free tier to flag possible leftover boilerplate for review. Only suspicious-looking articles are checked, sent in batches — see **Boilerplate detection** below. Candidates are logged, never auto-applied. Requires `GEMINI_API_KEY`/`GEMINI_API_KEYS` |
| `--detect-language-llm` | off | `auto`, `url` | Let Gemini break ties when the local 3-judge language panel genuinely disagrees on an article. Only called for the disputed subset, not every article — see **Language detection** below |
| `--yes` | off | `clear`, `delete` | Skip the confirmation prompt |

The interactive menu currently always uses the default delay (1.5s); pass `--delay` via the CLI form if you need to change it.

## Output

Each saved article is written as `domain__slug__hash.txt`, split first into **`raw/`** and **`lemmatized/`**, then into a **per-language subfolder** — e.g. `news_corpus/raw/is/ruv.is__some-slug__a1b2c3d4.txt`:

```
news_corpus/
  raw/        cy/  en/  is/  disputed/  ...
  lemmatized/ cy/  en/  is/  disputed/  ...
  scraped_urls.txt  corpus_metadata.jsonl  language_disputes.json  boilerplate_candidates.json
```

**Corpora saved before 2026-09-27** use the old order (`<lang>/raw/`). Scraping, dedup, and backfill all still read old-layout files, and each run prints a reminder while any remain. To move them over once (never overwrites; filenames unchanged, so the manifest and metadata stay valid):

```bash
python corpus_layout.py migrate --output-dir ./news_corpus --dry-run
python corpus_layout.py migrate --output-dir ./news_corpus
```

The raw extracted text is **always** saved to `raw/<lang>/`. If `--lemmatize` is given, a lemmatized copy is **additionally** saved to `lemmatized/<lang>/`, with `__lemma` appended to the filename before the extension (same domain/slug/hash prefix as its raw counterpart, so the two are easy to spot as a matched pair) — lemmatizing never replaces or discards the raw text. This also means re-running the same URL later with a different `--lemmatize` setting doesn't lose anything: both a raw-only scrape and a lemmatized scrape of the same article can coexist, since dedup (below) triggers off a single extraction either way rather than off which variant was saved.

Language is detected automatically from each article's own extracted text (via a local 3-judge panel — see **Language detection** below), not guessed from the source site, so a mixed-language listing page sorts itself correctly on its own. Text lands in one of three places:

- a real language folder (`raw/is/`, `raw/en/`, etc.) when the panel reaches consensus,
- `raw/unknown/` when no judge had anything usable to say at all,
- `raw/disputed/` when judges genuinely disagreed (or only one could vote) even after an optional Gemini tiebreak — see `language_disputes.json`.

A `scraped_urls.txt` manifest is kept at the **top level** of the output directory (shared across all language subfolders, since dedup is corpus-wide), one line per article:

```
https://example.com/news/some-slug    <content-hash>
```

This powers two dedup checks on every run:

- **URL dedup** — same URL won't be re-scraped.
- **Content dedup** — the same story republished under a *different* URL/slug is detected by hashing the extracted text, and skipped.

Old manifests from before content-hash dedup existed (URL-only, no second column) still load fine. The manifest is also reconciled against what's actually on disk at the start of every run — if you delete a saved article file by hand, its manifest entry is pruned automatically rather than silently blocking a re-scrape forever.

## How article discovery works (auto mode)

Links found on a listing page are kept as candidate articles if they:

- stay on the same domain,
- aren't an obvious non-article path (`/tag/`, `/video/`, `/author/`, etc.),
- aren't a static asset (image, PDF, video, etc.), and
- **either** match a site-specific `listing_path_prefixes` override (see below) **or**, generically, have a hyphen-rich slug (`biden-signs-new-bill`) or a long numeric ID in the path.

The generic heuristic covers most news CMSs, but it's not a guarantee — some sites use opaque IDs it can't recognize at all (BBC's `/news/articles/<id>` has neither hyphens nor a long digit run; without the override below, `auto` mode would silently discover zero BBC articles from any listing page). If discovery misses articles or grabs junk on a new site, either add an override (see below) or fall back to **Mode `url`** with direct article URLs.

## How text extraction works

For a domain with a **site override** (below), a confirmed CSS selector is tried first. Otherwise, `trafilatura` pulls the main article body out of the page HTML automatically, discarding nav bars, ads, and related-story widgets with no manual per-site rules required. If that returns nothing usable, the script falls back to scanning known article-container elements (`article`, `main`, `.article-body`, etc.) directly, and finally to raw `<p>` tag concatenation as a last resort.

At every stage, extracted paragraphs are run through a boilerplate filter before being saved — this catches things that render as ordinary `<p>` tags *inside* the article body but aren't actually article prose: footer contact blocks (phone numbers, addresses), copyright lines, cookie-consent placeholders shown in place of a blocked embed, and "read more" teaser duplicates. This list grows as new patterns turn up — see `boilerplate_patterns.py` (`BOILERPLATE_PATTERNS`, `END_OF_ARTICLE_MARKERS`, `READ_MORE_SUFFIXES`).

### Site overrides

`SITE_OVERRIDES` in the main script holds evidence-backed *extraction* rules for specific domains — built using the included `inspect_selectors.py` tool (see below), never guessed. Each entry can supply a proven `article_selector`, `exclude_selectors` for known junk inside it, and/or `listing_path_prefixes` to fix article discovery on sites the generic heuristic can't handle. An override that matches nothing on a given page (a template change, or a page type it wasn't built against — e.g. a live-blog vs. a standard article) falls through cleanly to the generic path rather than breaking the scrape.

`LANGUAGE_OVERRIDES` in `language_detection.py` is a **separate** table, keyed by the same domains but deliberately not merged into `SITE_OVERRIDES` — extraction and language ID are two independent concerns that happen to both key off domain. It holds `expected_language` (a soft corroborating prior for the judge panel), `language_lock` (a much stronger claim — when true, the panel is skipped entirely for that domain and the mapped code is trusted directly), and `listing_urls` (what mode 3 resolves a language code to).

Currently covered by `SITE_OVERRIDES` (each entry has comments in the script for what was actually tested): **ruv.is**, **bbc.com** (incl. Cymru Fyw), **apnews.com**, **theguardian.com** (ordinary articles only — `/live/*` pages use a different template and aren't covered), **tagesschau.de**, **dw.com**, **nrk.no**, **nos.nl**, **ansa.it**, **rtve.es**, **rtp.pt**, **svt.se**, **lefigaro.fr**, **france24.com**, **newyddion.s4c.cymru** (Newyddion S4C), **ycymro.cymru** (Y Cymro).

Every one of those domains is also `language_lock`'d in `LANGUAGE_OVERRIDES`, reachable via mode 3 with these codes: `is` (ruv.is), `en` (bbc.com, apnews.com, theguardian.com), `cy` (bbc.com/Cymru Fyw, newyddion.s4c.cymru, ycymro.cymru), `de` (tagesschau.de, dw.com), `nb` (nrk.no), `nl` (nos.nl), `it` (ansa.it), `es` (rtve.es), `pt` (rtp.pt), `sv` (svt.se), `fr` (lefigaro.fr, france24.com).

This list only limits **mode 3** and **tuned extraction** — modes 1 and 2 accept any site's URLs and fall back to trafilatura + the generic discovery heuristic. See `LANGUAGE_OVERRIDES_NOTES` at the bottom of `language_detection.py` for Welsh sources that were checked and deliberately *not* added (the 360.cymru network), and for the trade-off of language-locking a source.

**Sitemaps as listing URLs:** any listing URL whose path ends in `.xml` is read as a leaf XML sitemap — every `<loc>` goes through the same article filter as links on a normal listing page. This is usually the only way to reach a site's back catalogue rather than just its front page (Newyddion S4C's sitemap covers a rolling month, ~500 items, vs ~70 links on its homepage). Sitemap *index* files aren't followed; point at the leaf sitemap.

**Adding a new site:** run `inspect_selectors.py` against a real article URL on that site — it checks every plausible container selector (not just the biggest) for nested junk, auto-recommends a selector + exclusions, and prints a ready-to-paste `SITE_OVERRIDES` entry with its reasoning. Review the reasoning before pasting it in — it's a draft based on one sample article, not a guarantee for the whole domain.

```bash
python inspect_selectors.py https://example.com/some/real/article
```

Worth re-running on a couple of different article types per site (a live sports blog and an ordinary bylined story extract very differently on RÚV, for instance) before fully trusting a new override. If you also want the domain reachable via mode 3, add a matching `LANGUAGE_OVERRIDES` entry by hand afterward — `inspect_selectors.py` only handles the extraction side, not the listing URL or the lock decision.

## Boilerplate detection (optional, LLM-assisted)

`--detect-boilerplate` adds an LLM pass (Gemini, free tier) over already-saved, already-filtered article text, looking for leftover site chrome that `boilerplate_patterns.py` didn't catch — the same kind of thing you'd find by eye and add to that file by hand. It never changes what gets saved; it only logs candidates to `boilerplate_candidates.json` for review.

To keep this within the free tier's tight per-minute request cap:

- **Suspicion filter** — most normal-length, well-formed articles skip the LLM pass entirely. Short articles (under ~80 words), articles with a short/list-shaped first or last paragraph, or articles that are mostly short paragraphs throughout are the ones actually checked.
- **Batching** — suspicious articles queue up and are sent as one combined request per batch of 15, instead of one request per article.

Run `python icelandic_text_extractor.py review --output-dir ./news_corpus` afterward to interactively approve or skip each logged candidate — approved patterns are written straight into `boilerplate_patterns.py`.

## Language detection

Each article's raw extracted text (before any `--lemmatize` step) is run through a panel of three **local, offline** judges — no API calls, no rate limit — to decide its output folder:

- **GlotLID** — a fastText model covering 2000+ languages, including ones well outside a typical closed-list detector's range (e.g. Kalaallisut/Greenlandic).
- **OpenLID-v3** — a second fastText model, reported to have better precision/false-positive rate than GlotLID; meant to pair with it.
- **lingua** — the detector this project used to rely on alone. Solid within its closed candidate list (currently Icelandic, English, Welsh, Norwegian Bokmål/Nynorsk, Swedish, Danish, German, French, Spanish, Dutch, Irish, Italian, Portuguese) — it was chosen over `langdetect` after `langdetect` consistently misidentified real Icelandic article text as Norwegian or Swedish in testing.

(CLD3/gcld3 was previously a fourth judge and has since been removed from the panel — see `requirements.txt`.)

Each judge's vote is weighted by (its own base reliability) × (its own confidence on this specific text). A language "wins" only if it holds at least 55% of the total weighted vote **and** at least 2 judges actually voted for it — a confident-but-lone judge doesn't count as consensus. If no judge has anything to say, the article goes to `unknown/`. If the panel genuinely disagrees, the article goes to `disputed/` (see `language_disputes.json` for the full per-judge breakdown), unless `--detect-language-llm` is on, in which case Gemini gets one tiebreak vote for that article only — not every article, which is what keeps this optional feature within the free tier's rate limit.

Separately, a domain can be `language_lock`'d in `LANGUAGE_OVERRIDES` (see **Site overrides** above) — locked domains skip this whole panel and are filed directly under their mapped code, no text inspection at all. That's a much stronger trust step than the corroborating `expected_language` prior every other domain gets, so it's only set on domains confirmed, not assumed, to never publish outside their mapped language(s).

If you start scraping a new language and articles keep landing in `unknown/`, check whether it's in `_LINGUA_LANGUAGES` in the script (for the `lingua` judge) — GlotLID and OpenLID-v3 already cover a much broader set on their own. Spot-check a real sample afterward rather than trusting an addition blindly (see the Icelandic/Norwegian mix-up above for why).

## Corpus metadata (`corpus_metadata.jsonl`)

Every saved article also gets one row appended to `corpus_metadata.jsonl` (top level of `--output-dir`, same spot as `scraped_urls.txt`) — a flat, append-only table meant to be loaded straight into pandas/R for exploratory analysis, one JSON object per line:

| Field | Meaning |
| --- | --- |
| `url`, `domain`, `scraped_at` | Provenance |
| `discovered_via` | The listing page or sitemap the article was found on (auto mode); `null` for direct-URL scrapes and backfilled rows. If several listing pages link the same article in one run, the **first** listing URL given wins — so pass specific section/topic pages before general ones (homepage, sitemap) to get a useful topic signal |
| `language_code`, `language_locked`, `language_disputed` | Where the article was filed and how it got there |
| `language_winner_share`, `language_votes` | The language panel's full per-judge breakdown (code + confidence per judge) and how confidently it agreed — recorded for **every** article, not just disputed ones, so a low `language_winner_share` on a non-disputed article is a usable signal (e.g. possible code-switching) rather than discarded data |
| `extraction_method` | `"override"` / `"trafilatura"` / `"fallback"` — which extraction path actually produced the saved text |
| `was_suspicious_for_boilerplate_check` | Whether `is_suspicious()` flagged this article (regardless of whether `--detect-boilerplate` was on) |
| `word_count`, `sentence_count`, `paragraph_count`, `char_count` | Basic size stats |
| `avg_sentence_length`, `avg_word_length`, `type_token_ratio` | Cheap, standard corpus-linguistics proxies for lexical complexity/diversity — **not** a verified register/formality label; useful as one exploratory axis, not something to cite as "this text is formal" without further validation |
| `url_topic_hint` | First meaningful URL path segment (e.g. `sport`), a free rough section/topic proxy — `null` when the URL has no real section segment |
| `lemmatized_lang` | The `--lemmatize` code used, if any |
| `backfilled` | `true` for rows added by `backfill-metadata` rather than a live scrape (see below) — several fields are `null` on these rows since they aren't recoverable from a saved `.txt` alone |

**Backfilling an existing corpus:** if you already scraped articles before this feature existed, `python corpus_metadata.py backfill --output-dir ./news_corpus` (stdlib only, works in any Python, no scraper dependencies needed; `icelandic_text_extractor.py backfill-metadata` and the interactive menu's "backfill" option do the same thing but need Playwright etc. installed since that script imports them at the top) walks every saved `raw/*.txt`, recovers each one's real URL from `scraped_urls.txt`, and appends whatever subset of the fields above is computable from disk alone (`extraction_method` and the language-panel breakdown aren't — those only exist at scrape time). Safe to re-run; already-logged URLs are skipped.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests live in `tests/` and cover the pure logic modules only (`language_detection.py`'s ensemble resolution, `corpus_metadata.py`'s stats/URL-hint functions, `boilerplate.py`'s suspicion filter) — they don't need Playwright, a browser, fasttext models, or a Gemini key, since none of those are imported at module level in the code under test. `conftest.py` at the repo root makes the flat, package-less modules importable from `tests/`.

## Notes

- Pages are rendered with headless Chromium (via Playwright) before extraction, so JS-heavy sites work too. The script waits for a real paragraph *count* (not just one `<p>`) plus a short network-idle grace period before reading the page, to avoid snapshotting a bylined article template mid-hydration and capturing only its header/teaser block.
- Failed URLs get one automatic retry pass at the end of each run.
- `--lemmatize` loads its model up front for the requested language — a bad language code fails immediately with a clear error instead of failing silently on every article. Pipelines are cached per language code (not a single shared slot), so doing multiple runs back-to-back in the interactive menu with different `--lemmatize` languages loads a fresh model for each one instead of silently reusing whichever language loaded first.
- Progress is shown with a live `tqdm` bar (✅/💥 counts in the postfix); per-article status lines print above it without corrupting the bar.
- `GEMINI_API_KEYS` (comma- or newline-separated) rotates across multiple free-tier keys when one hits its limit. Worth reading Google's terms of service on this before relying on it — quota is enforced per Google Cloud *project*, not per key, so this only helps if the keys are on genuinely separate projects, and using multiple free-tier keys/projects specifically to route around a rate limit is generally understood to be against the free tier's terms.

## Companion tool: `inspect_selectors.py`

Standalone, not part of the main pipeline. Loads one real article URL, checks every plausible article-container selector (not just the biggest) for suspicious nested content (share widgets, related-content/ad widgets, accessibility controls, etc.), and auto-recommends a selector plus any needed exclusions — printing a ready-to-paste `SITE_OVERRIDES` entry with its reasoning. Used to build evidence-backed overrides — see **Site overrides** above.

```bash
python inspect_selectors.py <article_url>
```