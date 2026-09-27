"""
inspect_selectors.py
=====================
One-off helper -- NOT part of the main pipeline. Loads a single real
article URL and prints out every plausible "this might be the article
body container" element, ranked by how much paragraph text it holds,
checks EACH candidate (not just the biggest) for junk sitting inside
it, then prints a draft SITE_OVERRIDES entry -- the same selector/
exclude_selectors decision this project has been making by hand,
automated so there's less to transcribe and re-explain per site.

This is still a DRAFT, not a commit-blindly answer: the auto-pick logic
below is the same mechanical rule that's been applied by hand on every
site so far (prefer a clean tighter container over a dirty bigger one;
otherwise exclude the junk from the biggest one) -- it can't tell you
whether the ONE sample article it looked at is representative of the
whole domain (a different article template, a live-blog, a paywalled
piece), the same caveat that applied to every hand-picked entry before
it. Read the printed reasoning, don't just paste the dict.

Point this at one article URL per site you want a SITE_OVERRIDES entry
for.

Usage:
    python inspect_selectors.py https://www.ruv.is/frett/some-article
    python inspect_selectors.py https://www.bbc.com/news/some-article
    python inspect_selectors.py https://www.bbc.co.uk/cymrufyw/some-article
"""
import asyncio
import re
import sys
from urllib.parse import urlparse

from playwright.async_api import async_playwright

# Common container candidates across news CMS templates -- same spirit
# as icelandic_text_extractor.py's ARTICLE_CONTAINER_SELECTORS, just a
# longer list here since this is a one-off diagnostic, not something
# that runs on every page of a production scrape.
CANDIDATE_CONTAINERS = [
    "article", "main article", "[itemprop='articleBody']",
    "[data-component='text-block']",  # BBC's block-based body markup
    ".article-body", ".article-content", ".article__body",
    "#article-body", "main", "[role='main']",
    "[class*='ArticleBody']", "[class*='article-body']",
    "[class*='RichText']", "[class*='story-body']",
]

# Elements that often sit INSIDE the main container but aren't real
# body text -- checked within EVERY candidate container now (not just
# the biggest one), so a tighter candidate can be recognized as clean
# even when a bigger one isn't.
SUSPECT_INNER_SELECTORS = [
    "footer", "nav", "[class*='social']", "[class*='share']",
    "[class*='related']", "[class*='promo']", "[class*='newsletter']",
    "[class*='advert']", "[class*='font-size']", "[class*='accessib']",
    "[class*='byline']", "[class*='tag']",
]

# Tags that are safe to exclude wholesale wherever they appear inside
# the article container -- real body prose doesn't legitimately live
# inside a <footer> or <nav>, so excluding by TAG here is more stable
# than pinning to a styled-components hash class that could go stale on
# the site's next rebuild (see independent.co.uk's SITE_OVERRIDES entry
# for a real example of exactly that choice).
SEMANTIC_TAGS = {"footer", "nav", "aside"}

# A clean alternative candidate has to hold onto at least this fraction
# of the biggest candidate's char count to be preferred over "biggest
# candidate + exclude its junk" -- otherwise a tiny fragment container
# (e.g. matching just a pull-quote) could out-rank the real article body
# purely for being junk-free. 0.7 mirrors the AP News judgment call by
# hand (RichText at 7067 chars was accepted over main's 7883 -- a 90%
# ratio) with headroom below that, not an exact science.
CLEAN_ALTERNATIVE_MIN_RATIO = 0.7


async def dedupe_nested_matches(elements):
    """Given the ElementHandles from ONE selector's query_selector_all(),
    drops any element that's a DESCENDANT of another element already in
    the same list -- collapses a deeply nested widget (share-bar div
    containing a share-tools div containing a <ul> containing <li>s
    containing <a>s, every level independently matching
    [class*='share']) down to just the outermost wrapper. Confirmed
    necessary on a real site: lefigaro.fr's share widget alone produced
    8 separate matches for '[class*='share']' before this, one per
    nesting level, all describing the same single widget.

    O(n^2) element-containment checks -- fine for the handful of matches
    a single selector realistically produces on one page; this is a
    one-off diagnostic tool, not something run at scrape volume."""
    keep = []
    for i, el in enumerate(elements):
        nested_in_another = False
        for j, other in enumerate(elements):
            if i == j:
                continue
            try:
                contained = await other.evaluate(
                    "(outer, inner) => outer.contains(inner) && outer !== inner", el)
            except Exception:
                contained = False
            if contained:
                nested_in_another = True
                break
        if not nested_in_another:
            keep.append(el)
    return keep


async def collect_suspects(container_el):
    """Runs every SUSPECT_INNER_SELECTORS check against ONE container
    element, returns {selector: [{"text":..., "class":..., "tag":...}]}
    for every selector that matched at least once inside it. Empty dict
    means this specific container is clean.

    Every per-element property read is wrapped individually: attribute
    selectors like [class*='share'] can match SVG elements (inline share
    icons are a common case), and Playwright's inner_text() is an
    HTML-layout operation that raises "Node is not an HTMLElement" on
    those -- confirmed on a real site (lefigaro.fr) crashing the whole
    run over a single non-HTML node. text_content() works on any node
    type as a fallback; if even that fails (detached node, etc.) the
    match is still recorded with an empty string rather than losing the
    whole selector group over one bad element."""
    found = {}
    for selector in SUSPECT_INNER_SELECTORS:
        try:
            inner = await container_el.query_selector_all(selector)
        except Exception:
            continue
        inner = await dedupe_nested_matches(inner)
        matches = []
        for el in inner:
            try:
                t = (await el.inner_text()).strip()
            except Exception:
                try:
                    t = (await el.text_content() or "").strip()
                except Exception:
                    t = ""
            t = t.replace("\n", " ")
            try:
                cls = await el.get_attribute("class") or ""
            except Exception:
                cls = ""
            try:
                tag = await el.evaluate("e => e.tagName.toLowerCase()")
            except Exception:
                tag = "?"
            matches.append({"text": t, "class": cls, "tag": tag})
        if matches:
            found[selector] = matches
    return found


# Below this, a common token is more likely a short utility/layout class
# ("on", "row", "col") than something specific enough to safely narrow
# an exclude selector to.
MIN_TOKEN_CHARS = 4

_HASH_PREFIX_RE = re.compile(r"^(sc|css|emotion|jsx|styled)[-_][a-z0-9]{5,}$", re.IGNORECASE)


def _looks_hash_like(token):
    """Heuristic filter for CSS-in-JS / bundler-generated class tokens
    (styled-components 'sc-jXbNGH', Emotion 'css-1a2b3c', a CSS-module
    '<name>_<hash>', etc). These often end up as the LONGEST common
    token in a class list -- exactly what common_class_token() otherwise
    treats as "most specific" -- but they're regenerated on every
    rebuild, so an exclude selector pinned to one is likely to silently
    stop matching after the site's next redeploy: long and unique isn't
    the same thing as stable. Not a real hash detector, just two cheap,
    evidence-motivated signals: a known generator prefix, or a long
    token with implausibly few vowels for a real human-authored class
    name (words have vowels; hashes mostly don't)."""
    if _HASH_PREFIX_RE.match(token):
        return True
    if len(token) >= 6 and token.replace("-", "").isalnum():
        if sum(1 for c in token.lower() if c in "aeiou") <= 1:
            return True
    return False


def common_class_token(matches):
    """Given the matches for one suspect-selector group, finds a single
    class token shared by EVERY matched element's class list (the same
    thing that made independent.co.uk's many differently-styled Taboola
    widgets excludable with one selector: they all carried
    'trc_related_container' despite nothing else in their class lists
    matching). Among tokens common to every match, prefers the longest
    one that also clears MIN_TOKEN_CHARS and doesn't look build-hash-
    generated (see _looks_hash_like()) -- "longest" alone isn't a
    reliable proxy for "most specific", since a hash is often the
    longest token in a class list precisely because it's random, not
    because it's meaningful.

    Returns (token, caveat): token is the chosen class string, or None
    if nothing common was found, or if there's only one match (a single
    element's own class list isn't "common" to anything -- falls
    through to the broad selector instead, flagged for manual review
    rather than guessed at). caveat is a warning string when the
    fallback had to reach for a token that looked short or hash-like
    (nothing clean survived the filter), None when the pick looks like
    a genuine, stable, human-authored class name."""
    if len(matches) < 2:
        return None, None
    token_sets = [set(m["class"].split()) for m in matches if m["class"]]
    if not token_sets:
        return None, None
    common = set.intersection(*token_sets)
    if not common:
        return None, None

    clean = [t for t in common if len(t) >= MIN_TOKEN_CHARS and not _looks_hash_like(t)]
    if clean:
        return max(clean, key=len), None

    # Nothing clean survived -- every common token is either too short
    # to be specific, or looks build-generated. Still return the longest
    # raw token (some narrowing beats none), but flag it so the printed
    # recommendation makes the tradeoff visible rather than silently
    # picking a class that may already be stale by the next scrape.
    token = max(common, key=len)
    return token, (
        f"{token!r} looks build-generated or too short to be a stable "
        f"class name -- it may stop matching after the site's next "
        f"redeploy. Worth re-checking with inspect_selectors.py if this "
        f"domain's extraction ever regresses."
    )


def build_exclude_selectors(suspects):
    """suspects: the dict returned by collect_suspects() for whichever
    container ended up recommended. Returns (excludes, warnings) --
    excludes is the deduplicated list of selector strings to actually
    put in SITE_OVERRIDES; warnings flags any group that couldn't be
    narrowed and fell back to the broad SUSPECT_INNER_SELECTORS entry
    as-is, which is worth a human double-check before trusting it (a
    broad "[class*='related']" could exclude more than intended)."""
    excludes = []
    warnings = []
    for selector, matches in suspects.items():
        tags = {m["tag"] for m in matches}
        if tags <= SEMANTIC_TAGS and len(tags) == 1:
            candidate = tags.pop()
        else:
            token, caveat = common_class_token(matches)
            if token:
                candidate = f"[class*='{token}']"
                if caveat:
                    warnings.append(caveat)
            else:
                candidate = selector
                warnings.append(
                    f"couldn't narrow {selector!r} ({len(matches)} match(es)) to a "
                    f"specific class token -- using it as-is, which may exclude more "
                    f"than intended. Check the printed matches by hand."
                )
        if candidate not in excludes:
            excludes.append(candidate)
    return excludes, warnings


def pick_recommendation(candidates):
    """candidates: list of (selector, n_paras, total_len, suspects_dict,
    preview), already sorted by total_len descending. Returns
    (chosen_selector, exclude_selectors, warnings, reasoning_lines).

    Mirrors the same call made by hand on every site so far: a clean
    container close in size to the biggest one wins outright (AP News:
    RichText over main); otherwise, the biggest container is kept and
    its junk gets excluded (Independent: article + excluding the
    Taboola widgets, since no clean alternative existed there at all)."""
    reasoning = []
    top_selector, top_paras, top_len, top_suspects, _ = candidates[0]

    if not top_suspects:
        reasoning.append(
            f"{top_selector!r} is both the fullest candidate ({top_len} chars) "
            f"AND clean -- no suspect elements found inside it. Using it directly."
        )
        return top_selector, [], [], reasoning

    threshold = top_len * CLEAN_ALTERNATIVE_MIN_RATIO
    clean_alts = [c for c in candidates[1:] if not c[3] and c[2] >= threshold]
    if clean_alts:
        chosen = max(clean_alts, key=lambda c: c[2])
        reasoning.append(
            f"{top_selector!r} ({top_len} chars) has suspect elements inside it "
            f"(see below). {chosen[0]!r} ({chosen[2]} chars, "
            f"{chosen[2] / top_len:.0%} of the biggest candidate) is clean AND close "
            f"enough in size -- using the clean, tighter candidate instead of "
            f"excluding junk from the bigger one."
        )
        return chosen[0], [], [], reasoning

    excludes, warnings = build_exclude_selectors(top_suspects)
    reasoning.append(
        f"{top_selector!r} ({top_len} chars) is the fullest candidate, but has "
        f"suspect elements inside it, and no clean alternative came within "
        f"{CLEAN_ALTERNATIVE_MIN_RATIO:.0%} of its size. Keeping {top_selector!r} "
        f"and excluding what was found inside it instead."
    )
    return top_selector, excludes, warnings, reasoning


async def inspect(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(30000)
        await page.goto(url, wait_until="domcontentloaded")
        try:
            await page.wait_for_function(
                "document.querySelectorAll('p').length >= 3", timeout=8000)
        except Exception:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass

        print(f"\n{'='*70}\nInspecting: {url}\n{'='*70}\n")

        print("--- CANDIDATE ARTICLE CONTAINERS (ranked by paragraph text length) ---\n")
        candidates = []
        for selector in CANDIDATE_CONTAINERS:
            try:
                el = await page.query_selector(selector)
            except Exception:
                continue
            if not el:
                continue
            paras = await el.query_selector_all("p")
            texts = []
            for pa in paras:
                t = (await pa.inner_text()).strip()
                if len(t) >= 20:
                    texts.append(t)
            total_len = sum(len(t) for t in texts)
            suspects = await collect_suspects(el)
            candidates.append((selector, len(texts), total_len, suspects, texts[:2]))

        candidates.sort(key=lambda c: c[2], reverse=True)

        for selector, n_paras, total_len, suspects, preview in candidates:
            flag = f" -- {len(suspects)} suspect group(s) inside" if suspects else " -- clean"
            print(f"  {selector!r:40s} -> {n_paras} paragraphs, {total_len} chars{flag}")
            for p_text in preview:
                snippet = p_text[:90] + ("..." if len(p_text) > 90 else "")
                print(f"      | {snippet}")
            print()

        if not candidates:
            # Whole-page paragraph count is a quick way to tell "wrong
            # selector" apart from "this page has no real article text
            # at all" (a bot-check/consent wall/paywall shell) -- the
            # latter needs a different fix entirely (headers, wait
            # conditions, or the site may not be scrapeable this way),
            # not a new CANDIDATE_CONTAINERS entry.
            all_paras = await page.query_selector_all("p")
            all_text_len = 0
            for pa in all_paras:
                all_text_len += len((await pa.inner_text()).strip())
            print("  (none of the candidate selectors matched anything on this page)")
            print(f"  Whole-page <p> count: {len(all_paras)}, total chars: {all_text_len}")
            if all_text_len < 500:
                print("  That's very little text anywhere on the page -- this looks "
                      "less like a missing selector and more like the real article "
                      "never loaded (bot/consent check, paywall shell, JS that needs "
                      "a longer wait). Worth checking what page.content() actually "
                      "contains before adding a new selector to CANDIDATE_CONTAINERS.")
            else:
                print("  There IS real text on the page, just not inside any known "
                      "candidate selector -- this site may need a custom selector "
                      "added to CANDIDATE_CONTAINERS above.")
            await browser.close()
            return

        for selector, n_paras, total_len, suspects, _ in candidates:
            if not suspects:
                continue
            print(f"--- SUSPECT INNER ELEMENTS inside {selector!r} ---\n")
            for sus_selector, matches in suspects.items():
                for m in matches:
                    snippet = m["text"][:90] + ("..." if len(m["text"]) > 90 else "")
                    print(f"  matched {sus_selector!r} (tag={m['tag']!r}, class={m['class']!r})")
                    print(f"      | {snippet}\n")

        chosen_selector, excludes, warnings, reasoning = pick_recommendation(candidates)

        print("--- RECOMMENDATION ---\n")
        for line in reasoning:
            print(f"  {line}\n")
        for w in warnings:
            print(f"  ⚠️  {w}\n")

        domain = urlparse(url).netloc.replace("www.", "").lower()
        excludes_repr = "[" + ", ".join(repr(x) for x in excludes) + "]"
        print("  Draft SITE_OVERRIDES entry -- REVIEW before pasting in, especially "
              "any ⚠️ warnings above, and remember this reflects ONE sample article, "
              "not necessarily every template this domain uses:\n")
        print(f'    "{domain}": {{')
        print(f'        "article_selector": {chosen_selector!r},')
        print(f'        "exclude_selectors": {excludes_repr},')
        print('        "listing_path_prefixes": None,  # not checked by this tool -- '
              'verify against a listing page separately')
        print('    },')

        await browser.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_selectors.py <article_url>")
        sys.exit(1)
    asyncio.run(inspect(sys.argv[1]))