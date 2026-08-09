package deepresearchbench

import "regexp"

// citationMarkerPatterns match bracketed numeric citation forms that
// upstream's LLM-based article cleaner (clean_article.py) strips from the
// candidate article before RACE scoring. This is a deliberate, conservative
// simplification: only unambiguous bracketed forms are stripped.
//
// Freestanding bare trailing numbers (upstream's first citation form, e.g.
// "... 7 levels 15") are NOT stripped here — that pattern is indistinguishable
// from ordinary prose (list markers, dates, plain numbers) without semantic
// understanding, which is exactly why upstream itself resorts to an LLM for
// it. Under-stripping is preferable to corrupting prose.
//
// Markdown link-style citations ("[title](url)") are also NOT touched here:
// upstream's RACE scoring prompt never strips markdown links from the scored
// article text. Only FACT's post-extraction fact-text cleanup does that (see
// removeURLs in fact.go), which operates on extracted citation strings, not
// on the article passed to RACE.
var citationMarkerPatterns = []*regexp.Regexp{
	// OpenAI-style tool-citation form, e.g. "[12†source]" or "[5†L23]".
	regexp.MustCompile(`\[\d+†[^\]]*\]`),
	// Runs of one or more bracketed numeric citations, e.g. "[12]" or
	// "[1][2][3]" or "[1, 2]".
	regexp.MustCompile(`(?:\[\d+(?:,\s*\d+)*\]\s*)+`),
}

// stripCitationMarkers removes in-text bracketed citation-marker noise from
// a candidate report before RACE scoring, deterministically and without an
// LLM call.
func stripCitationMarkers(article string) string {
	cleaned := article
	for _, pattern := range citationMarkerPatterns {
		cleaned = pattern.ReplaceAllString(cleaned, "")
	}
	return cleaned
}
