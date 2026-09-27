"""
Boilerplate / junk-text tables for the news scraper.
=====================================================
Pure data, no functions -- mirrors the mutation_tables.py split on the
Welsh project (data lives here, matching/filtering logic stays in the
main script, which imports these). Kept separate specifically because
this list is expected to keep growing as new sites get scraped; adding a
new pattern should never require touching extraction logic.

Three tables, three different jobs:

  BOILERPLATE_PATTERNS   -- fixed junk text to drop wherever it appears
                             (cookie banners, consent placeholders, promo
                             blocks, editorial-policy footers, phone/
                             address patterns, etc). Each entry is a
                             regex tried against a single paragraph.

  END_OF_ARTICLE_MARKERS -- stronger than a boilerplate pattern: once one
                             of these matches a paragraph, EVERYTHING
                             from that paragraph to the end of the
                             extracted text is dropped, not just the
                             matching paragraph itself. Needed for
                             footers that are followed by dynamic
                             related-content widgets (different headlines
                             every time, so there's no fixed string for
                             BOILERPLATE_PATTERNS to catch) -- see
                             SVT.se's "Så arbetar vi" footer.

  READ_MORE_SUFFIXES     -- "read more" / expand-widget trailing words.
                             Some CMS teaser widgets keep both the
                             truncated teaser (ending in one of these)
                             and the full text in the DOM at once; used
                             by dedupe_consecutive_paragraphs() in the
                             main script to collapse the pair into just
                             the full version.

Every entry below is evidence-backed from a real scraped sample, not
guessed -- see the comment above each one for what was actually observed
and which file it came from.
"""

# --------------------------------------------------------------------------
# BOILERPLATE_PATTERNS -- drop the matching paragraph, wherever it appears
# --------------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    # --- Icelandic (RUV) ---
    r"s[ií]mi\s*[:.]?\s*\d",          # "Sími: 515-3000"
    r"\bnetfang\s*[:.]?",              # "Netfang:"
    r"\bkt\.?\s*\d{6}-?\d{4}\b",       # Icelandic kennitala
    r"\b\d{3}[-\s]\d{4}\b",            # bare phone number pattern
    r"^\S+\s+\d+,\s*\d{3}\s+\S+$",     # "Efstaleiti 1, 103 Reykjavík" -- street, postcode, place
    r"^©", r"\ball rights reserved\b",
    r"^\s*(privacy policy|terms of (use|service))\s*$",
    # Cookie-consent placeholder shown in place of a blocked third-party
    # embed (RUV, confirmed from a real scraped article: appeared 3x
    # verbatim in one piece, once per blocked embed -- since it's UI
    # chrome rendered as <p> tags INSIDE the article body container, it
    # survives container-scoping and needs a text-content check instead).
    r"innfellt efni frá annarri vefsíðu",
    r"vafrakök\w*",
    r"^viltu samt sjá\??$",   # standalone consent-box follow-up, confirmed
                              # appearing as its own <p> alongside the above

    # --- Dutch (NOS.nl) ---
    # Promo block that sits at the end of almost every article (confirmed
    # across many real scrapes). Appears as one or two <p> tags inside/
    # after the body, so container-scoping alone does not remove it.
    r"maak nos jouw voorkeursbron in google",
    r"met een nieuwe functie van google bepaal je voortaan zelf",
    r"^meer bekijken\??$",
    # "Opens in new window" accessibility label attached to inline links --
    # extracted as its own paragraph, which splits the sentence it was
    # embedded in right down the middle (confirmed: "...criminaliteit /
    # (opent in nieuw venster) / onder Palestijnse Israëliërs..." was one
    # continuous sentence in the source, broken into three paragraphs by
    # this leaking through). Not real content in any case.
    r"^\(opent in nieuw venster\)$",

    # --- Danish (DR.dk) ---
    # Cookie-consent banner. Confirmed on a real scrape where this was the
    # ENTIRE extracted "article" (no actual story text at all) -- not just
    # noise near real content, a full extraction failure on its own. This
    # doesn't fix WHY DR is returning the banner/homepage river instead of
    # the article (that needs a SITE_OVERRIDES entry, still pending
    # inspect_selectors.py output) -- it just stops the banner text from
    # being saved as if it were prose.
    r"^dr passer p[åa] dine data$",
    r"dr indsamler oplysninger om dine bes[øo]g ved hj[æa]lp af cookies",
    r"dr bruger egne cookies og cookies fra tredjepart",

    # --- Swedish (SVT.se) ---
    # Editorial-policy footer ("This is how we work"), confirmed verbatim
    # and identical across every single SVT.se sample scraped so far -- a
    # fixed site-wide footer, not article content. Belt-and-suspenders
    # pattern on the body sentence itself; the header line ("Så arbetar
    # vi") is also handled as an END_OF_ARTICLE marker below, since on
    # some articles a dynamic "related stories" widget follows right
    # after this footer and needs to be dropped too, not just this one
    # paragraph.
    r"svt:s nyheter ska st[åa] f[öo]r saklighet och opartiskhet",

    # --- Welsh (Y Cymro, ycymro.cymru) ---
    # Publisher disclaimer, confirmed verbatim as the last <p> inside
    # .entry-content on 2 of 2 samples (a /newyddion/ and a /barn/ piece),
    # right after "Cyhoeddir gan Cyfryngau Cymru Cyf." (handled as an
    # END_OF_ARTICLE marker below). Belt-and-braces here because the raw
    # <p> fallback path drops paragraphs under 40 chars BEFORE end-marker
    # truncation runs, so that short marker line never gets a chance to
    # fire on that path -- this longer sentence does survive to be caught.
    r"^nid yw['’]r farn a amlygir yn erthyglau unigol y cyhoeddiad hwn",

    # --- video.js player chrome (any site embedding it) ---
    # Accessibility text from the video.js modal dialog, rendered as <p>
    # tags. Confirmed on a newyddion.s4c.cymru video-only item
    # (/article/36898), where it was the ONLY <p> text on the page: 3
    # paragraphs, 112 chars -- the first two read directly, the third
    # inferred as video.js's stock "End of dialog window." (exactly the
    # remaining 21 chars). Without these, a video-only page falling
    # through to the raw-<p> fallback could be saved as a 'Welsh article'
    # consisting of English player UI text.
    r"^this is a modal window\.?$",
    r"^beginning of dialog window\. escape will cancel and close the window\.?$",
    r"^end of dialog window\.?$",

    # --- English (BBC) ---
    # Newsletter sign-up CTA, confirmed as the final paragraph across 3
    # real scraped articles (a wildfire-insurance piece, a J&J settlement
    # piece, a US-munitions InDepth piece) -- and NOT generalizable by
    # _generalize_group()'s prefix/suffix diffing (checked directly: it
    # correctly returns None on these 3, since they share no meaningful
    # common prefix or suffix -- "Sign up for our Tech Decoded
    # newsletter...", "Get our flagship newsletter...", and "BBC InDepth
    # is the home... Sign up for the newsletter here" don't even start or
    # end the same way). This is a real gap in that tool: it generalizes
    # "same wrapper, different fill" (the Bob Howard byline case), not
    # "same semantic shape, entirely different wording" -- worth a second
    # look if this shape keeps recurring on other CMSs. Hand-written
    # here instead as an unordered "contains all three" check (word order
    # differs across the 3 examples -- sometimes "sign up" precedes
    # "newsletter", sometimes it follows), which is looser than the
    # prefix/suffix style used elsewhere in this file and carries a real
    # (believed low) false-positive risk on any single paragraph that
    # happens to mention a newsletter, signing up, AND the word "here" --
    # e.g. an article that's actually ABOUT a newsletter lawsuit. Revisit
    # if that ever shows up in the boilerplate_candidates.json review.
    r"(?=.*\bnewsletter\b)(?=.*\bsign up\b)(?=.*\bhere\b)",
    # Regional cross-promo footer, confirmed on a real scraped BBC Isle
    # of Man article ("Read more stories from the Isle of Man on the
    # BBC, watch BBC North West Tonight on BBC iPlayer and follow BBC
    # Isle of Man on Facebook and X."). Single example so far -- kept
    # literal rather than generalized to a "Read more stories from X...
    # BBC iPlayer... Facebook and X" template, same single-example
    # discipline as everywhere else in this file. Generalize once a
    # second regional BBC franchise's footer is seen.
    r"^read more stories from the isle of man on the bbc, watch bbc north west tonight on bbc iplayer and follow bbc isle of man on facebook and x\.?$",
    # Mid-article podcast-promo insert, confirmed on a real scraped BBC
    # article about the D4vd murder trial -- unlike the other BBC
    # patterns above, this one was NOT at the end of the article, it was
    # spliced into the middle of the body text ("Listen to the BBC's Fame
    # Under Fire podcast which brings you the latest from inside the Los
    # Angeles courtroom at the D4vd preliminary hearing."). Single
    # example, kept literal for the same reason as the footer above --
    # generalize to "Listen to the BBC's X podcast which..." once a
    # second example turns up.
    r"^listen to the bbc's fame under fire podcast which brings you the latest from inside the los angeles courtroom at the d4vd preliminary hearing\.?$",
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, promotional blurb, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/clye651yzdjo
    "^Watch\\ the\\ full\\ interview,\\ BBC\\ Panorama\\ 'Andy\\ Burnham:\\ the\\ Laura\\ Kuenssberg\\ Interview'\\ on\\ iPlayer\\ from\\ 06:00\\ on\\ Monday\\ and\\ BBC\\ One\\ on\\ Monday\\ at\\ 20:00\\ BST\\.$",
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, reporter credit, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/c33y3151vydo
    '^Additional\\ reporting\\ by\\ Bob\\ Howard\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, reader engagement prompt, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/cm2rgkp6z7no
    '^Do\\ you\\ have\\ a\\ story\\ suggestion\\ for\\ Essex\\?\\ Contact\\ us\\ below\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, social-follow prompt, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/cm2rgkp6z7no
    '^Follow\\ Essex\\ news\\ on\\ BBC\\ Sounds,\\ Facebook,\\ Instagram\\ and\\ X\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, content warning, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/c78gjyx4q2yo
    '^Contains\\ upsetting\\ scenes\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, omission placeholder, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/c872nj1n4xyo
    '^\\[\\.\\.\\.\\ article\\ middle\\ omitted\\ \\.\\.\\.\\]$',
    # Auto-suggested from boilerplate_candidates.json review (2026-07-30) -- bbc.com, site chrome / video player UI text, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/cwymnvkv2zlo
    '^Automatically\\ selects\\ the\\ best\\ quality\\ available$',
    # Auto-suggested from boilerplate_candidates.json review (2026-07-31) -- france24.com, error message, 1 example(s).
    # e.g. https://www.france24.com/fr/vid%C3%A9o/20260730-histoire-fontaines-wallace-embl%C3%A9matiques-d%C3%A9cor-parisien-paris-france-histoire-culture-prime
    '^Page\\ non\\ trouvée$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- bbc.com, social-share prompt, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/c74gwvygkjdo
    '^Follow\\ BBC\\ News\\ India\\ on\\ Instagram,\\ YouTube,\\ Twitter\\ and\\ Facebook\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- apnews.com, site access block / error page text, 1 example(s).
    # e.g. https://apnews.com/article/savannah-nancy-guthrie-missing-mom-ransom-note-7a53c668597233ef4dbc82204118067f
    '^The\\ owner\\ of\\ this\\ website\\ \\(apnews\\.com\\)\\ has\\ banned\\ you\\ temporarily\\ from\\ accessing\\ this\\ website\\.\\\nPlease\\ see\\ https://developers\\.cloudflare\\.com/support/troubleshooting/http\\-status\\-codes/cloudflare\\-1xxx\\-errors/error\\-1015/\\ for\\ more\\ details\\.\\\n\\ \\ \\ \\ \\ \\ \\ \\ Thank\\ you\\ for\\ your\\ feedback!$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- apnews.com, promotional blurb for site feature, 1 example(s).
    # e.g. https://apnews.com/article/news-quiz-july-31-fauci-trump-fifa-cracker-barrel-bts-berlin
    '^It\\ was\\ a\\ busy\\ week\\ in\\ the\\ news\\ cycle\\.\\ Test\\ your\\ knowledge\\ of\\ the\\ biggest\\ stories\\ and\\ see\\ what\\ you\\ might\\ have\\ missed\\ in\\ The\\ Associated\\ Press’\\ weekly\\ news\\ quiz\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, header navigation and newsletter subscription widget, 1 example(s).
    # e.g. https://www.theguardian.com/world/2026/jul/30/sign-up-for-this-is-india-our-free-weekly-newsletter-on-life-in-the-subcontinent
    "^The\\ Guardian\\\nThe\\ Guardian\\\nINT\\ Focused\\\nINT\\ Focused\\\nSign\\ up\\ for\\ This\\ is\\ India:\\ our\\ free\\ weekly\\ newsletter\\ on\\ life\\ in\\ the\\ subcontinent\\\nNiha\\ Masih’s\\ weekly\\ newsletter\\ about\\ the\\ stories,\\ ideas\\ and\\ news\\ makers\\ of\\ modern\\ India\\\nTell\\ your\\ friendsShare\\ \\\nYou'll\\ receive\\ this\\ newsletter\\ weekly$",
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, newsletter subscription prompt, 1 example(s).
    # e.g. https://www.theguardian.com/commentisfree/2026/aug/01/biohacker-bryan-johnson-girlfriend-menstrual-blood
    '^Sign\\ up\\ to\\ The\\ Week\\ in\\ Patriarchy\\\n\\\nGet\xa0Arwa\\ Mahdawi’s\xa0weekly\\ recap\\ of\\ the\\ most\\ important\\ stories\\ on\\ feminism\\ and\\ sexism\\ and\\ those\\ fighting\\ for\\ equality$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, layout structure label marker, 1 example(s).
    # e.g. https://www.theguardian.com/commentisfree/2026/aug/01/biohacker-bryan-johnson-girlfriend-menstrual-blood
    '^after\\ newsletter\\ promotion$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, image viewer UI element and photo credit, 1 example(s).
    # e.g. https://www.theguardian.com/commentisfree/picture/2026/aug/01/christopher-harry-buying-suv-cartoon
    '^View\\ image\\ in\\ fullscreen\\ Illustration:\\ Christopher\\ Harry/The\\ Guardian$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, topic tag list navigation, 1 example(s).
    # e.g. https://www.theguardian.com/commentisfree/picture/2026/aug/01/christopher-harry-buying-suv-cartoon
    '^Explore\\ more\\ on\\ these\\ topicsRoad\\ transportSaturday\\ Opinion\\ cartoonGreenhouse\\ gas\\ emissionsRoad\\ safetyMotoringAutomotive\\ emissionsAutomotive\\ industry$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, navigation link label, 1 example(s).
    # e.g. https://www.theguardian.com/culture/2026/aug/01/from-ish-to-jared-leto-hollywoods-dark-secret-the-week-in-rave-reviews
    '^Read\\ the\\ full\\ review$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, related-article link fragment, 1 example(s).
    # e.g. https://www.theguardian.com/culture/2026/aug/01/from-ish-to-jared-leto-hollywoods-dark-secret-the-week-in-rave-reviews
    '^Further\\ reading\\ ‘I\\ thought\\ it$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, submission callout prompt, 1 example(s).
    # e.g. https://www.theguardian.com/lifeandstyle/2026/aug/01/blind-date-richard-katrina
    '^Fancy\\ a\\ blind\\ date\\?\\ Email\\ blind\\.date@theguardian\\.com$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, form assistance link and terms of service footer, 1 example(s).
    # e.g. https://www.theguardian.com/tv-and-radio/2026/jul/22/tell-us-your-favourite-happy-tv-show-ending
    '^If\\ you’re\\ having\\ trouble\\ using\\ the\\ form\\ click\\ here\\.\\ Read\\ terms\\ of\\ service\\ here\\ and\\ privacy\\ policy\\ here\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, newsletter privacy notice footer, 1 example(s).
    # e.g. https://www.theguardian.com/global/2022/sep/20/sign-up-for-the-first-edition-newsletter-our-free-news-email
    '^Privacy\\ Notice:\\ Newsletters\\ may\\ contain\\ information\\ about\\ charities,\\ online\\ ads,\\ and\\ content\\ funded\\ by\\ outside\\ parties\\.\\ If\\ you\\ do\\ not\\ have\\ an\\ account,\\ we\\ will\\ create\\ a\\ guest\\ account\\ for\\ you\\ on\\ theguardian\\.com\\ to\\ send\\ you\\ this\\ newsletter\\.\\ You\\ can\\ complete\\ full\\ registration\\ at\\ any\\ time\\.\\ For\\ more\\ information\\ about\\ how\\ we\\ use\\ your\\ data\\ see\\ our\\ Privacy\\ Policy\\.$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-01) -- theguardian.com, newsletter subscription promotional block, 1 example(s).
    # e.g. https://www.theguardian.com/info/2022/nov/14/football-daily-email-sign-up
    "^The\\ Guardian\\\nThe\\ Guardian\\\nSign\\ up\\ for\\ the\\ Football\\ Daily\\ newsletter:\\ our\\ free\\ football\\ email\\\nKick\\ off\\ your\\ afternoon\\ with\\ the\\ Guardian’s\\ take\\ on\\ the\\ world\\ of\\ football\\\nTell\\ your\\ friends\\\nYou'll\\ receive\\ this\\ newsletter\\ every\\ weekday\\\nThe\\ Guardian\\\nThe\\ Guardian\\\nKick\\ off\\ your\\ afternoon\\ with\\ the\\ Guardian’s\\ take\\ on\\ the\\ world\\ of\\ football\\\nTell\\ your\\ friends\\\nYou'll\\ receive\\ this\\ newsletter\\ every\\ weekday$",
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- france24.com, editorial credit, 1 example(s).
    # e.g. https://www.france24.com/fr/asie-pacifique/20260729-la-population-de-tigres-continue-d-augmenter-au-n%C3%A9pal-avec-429-individus
    '^Avec\\ AFP$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- france24.com, editorial credit, 1 example(s).
    # e.g. https://www.france24.com/fr/culture/20260726-le-mont-olympe-demeure-des-dieux-grecs-inscrit-au-patrimoine-mondial-de-l-unesco
    '^AFP$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- france24.com, related-article link, 1 example(s).
    # e.g. https://www.france24.com/fr/culture/20260726-le-mont-olympe-demeure-des-dieux-grecs-inscrit-au-patrimoine-mondial-de-l-unesco
    "^À\\ lire\\ aussiLes\\ plages\\ du\\ Débarquement\\ en\\ Normandie\\ inscrites\\ au\\ patrimoine\\ mondial\\ de\\ l'Unesco$",
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- france24.com, related-article link, 1 example(s).
    # e.g. https://www.france24.com/fr/am%C3%A9riques/20260729-stations-service-pharmacies-ports-cuba-ouvre-de-nombreux-secteurs-au-priv%C3%A9
    '^À\\ lire\\ aussiReportage\\ à\\ Cuba\\ :\\ un\\ pays\\ sous\\ pression,\\ paralysé\\ par\\ les\\ pénuries$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- bbc.com, video player UI element, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/cx2v91xn1z9o
    '^Play\\ next\\ item\\ automatically\\\n\\\nCurrently\\ unavailable$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- bbc.com, video player UI element, 1 example(s).
    # e.g. https://www.bbc.com/news/articles/cz05yx5dd7yo
    '^Play\\ next\\ item\\ automatically\\\n\\\nCurrently\\ unavailable\\\n\\\nPlay\\ next\\ item\\ automatically\\\n\\\nCurrently\\ unavailable$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- apnews.com, navigational topic hub link, 1 example(s).
    # e.g. https://apnews.com/article/south-africa-oprah-winfrey-school-closure-8869f220df77c92d875911740557bdf5
    '^AP\\ Africa\\ news:\\ https://apnews\\.com/hub/africa$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- apnews.com, navigational topic hub link, 1 example(s).
    # e.g. https://apnews.com/article/fever-fire-score-c658fcf4e1e10556740ae65cd494958d
    '^AP\\ WNBA:\\ https://apnews\\.com/hub/wnba\\-basketball$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- apnews.com, navigational topic hub link, 1 example(s).
    # e.g. https://apnews.com/article/chiefs-eric-bieniemy-shooting-5f9c6ade939436c884197902e3a4f557
    '^AP\\ NFL:\\ https://apnews\\.com/hub/nfl$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- theguardian.com, contributor credit, 1 example(s).
    # e.g. https://www.theguardian.com/us-news/2026/aug/01/d4vd-hearing-celeste-rivas-hernandez
    '^The\\ Associated\\ Press\\ contributed\\ reporting$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- theguardian.com, contributor credit, 1 example(s).
    # e.g. https://www.theguardian.com/us-news/2026/aug/01/troy-jackson-maine-senate
    '^Andrew\\ Witherspoon\\ contributed\\ reporting$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- theguardian.com, related-article link, 1 example(s).
    # e.g. https://www.theguardian.com/culture/2026/aug/01/from-ish-to-jared-leto-hollywoods-dark-secret-the-week-in-rave-reviews
    '^Further\\ reading\\ ‘It\\ was\\ 100%\\ the\\ moment\\ I\\ lost\\ innocence’:\\ how\\ a\\ childhood\\ stop\\-and\\-search\\ led\\ to\\ the\\ British\\ debut\\ movie\\ of\\ the\\ year$',
    # Auto-suggested from boilerplate_candidates.json review (2026-08-06) -- theguardian.com, related-article link, 1 example(s).
    # e.g. https://www.theguardian.com/culture/2026/aug/01/from-ish-to-jared-leto-hollywoods-dark-secret-the-week-in-rave-reviews
    '^Further\\ reading\\ Tom\\ Holland\\ says\\ some\\ of\\ his\\ films\\ are\\ ‘shit’\\ and\\ people\\ shouldn’t\\ see\\ them$',
]

# --------------------------------------------------------------------------
# END_OF_ARTICLE_MARKERS -- drop this paragraph AND everything after it
# --------------------------------------------------------------------------
# Confirmed on SVT.se: the "Så arbetar vi" editorial footer is sometimes
# followed by a "related stories" widget -- a topic tag plus bullet lines
# that glue a "Just nu" breaking-news badge directly onto the next
# headline with no space ("Just nuTiotusentals flyr...") -- clearly UI
# chrome, but each headline is different text every time, so there's no
# fixed string for BOILERPLATE_PATTERNS to catch. Truncating at the
# marker instead of pattern-matching the noise handles this regardless of
# what the related-content widget happens to say.
END_OF_ARTICLE_MARKERS = [
    r"^så arbetar vi$",  # SVT.se
    # Y Cymro: "Published by Cyfryngau Cymru Ltd." -- site-wide publisher
    # footer inside .entry-content, always followed by the disclaimer
    # sentence (see BOILERPLATE_PATTERNS). Confirmed on 2 of 2 samples.
    r"^cyhoeddir gan cyfryngau cymru cyf\.?$",
]

# --------------------------------------------------------------------------
# READ_MORE_SUFFIXES -- "read more" widget trailing words, by language
# --------------------------------------------------------------------------
# Some CMS teaser widgets (confirmed on DR.dk, using "vis mere") keep BOTH
# the truncated teaser (ending in this word) and the full text in the DOM
# at once -- CSS toggles which one shows, but both get scraped, producing
# the same paragraph twice in a row, once truncated once full.
READ_MORE_SUFFIXES = [
    "vis mere",   # Danish (DR.dk, confirmed)
    "vis mer",    # Norwegian
    "visa mer",   # Swedish
    "lees meer",  # Dutch
]