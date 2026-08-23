"""Comparators for 'did the model produce the same action': exact and fuzzy variants."""
import difflib
import json
from urllib.parse import urlparse


def _canonicalize(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value).strip()


def actions_match(oracle_tool_calls, oracle_text, predicted):
    """predicted: {"tool_calls": [...], "text": ...} from ModelClient.complete, or {"error": ...}."""
    if predicted is None or "error" in predicted:
        return False

    if oracle_tool_calls:
        pred_calls = predicted.get("tool_calls") or []
        if len(pred_calls) != len(oracle_tool_calls):
            return False
        for a, b in zip(oracle_tool_calls, pred_calls):
            if a["name"] != b["name"]:
                return False
            if _canonicalize(a["arguments"]) != _canonicalize(b["arguments"]):
                return False
        return True

    # no tool calls in the oracle turn -> compare final text exactly (normalized whitespace)
    return " ".join(oracle_text.split()) == " ".join((predicted.get("text") or "").split())


def _domain(url):
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return url.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _last_segment(path):
    return path.rstrip("/").rsplit("/", 1)[-1]


def _urls_match(a, b, threshold, segment_threshold=0.7):
    """Same domain alone isn't enough (en.wikipedia.org/wiki/X vs /wiki/Y are different pages),
    and comparing full paths lets a long shared prefix ("/wiki/") mask a totally different
    article name -- so same-domain URLs are compared on their last path segment (+ query) with
    a stricter threshold, since that segment is usually the actual page/article identifier.
    Falls back to whole-URL similarity at the looser threshold when domains differ outright."""
    if _domain(a) != _domain(b):
        return _similarity(a, b) >= threshold
    pa, pb = urlparse(a), urlparse(b)
    seg_a, seg_b = _last_segment(pa.path) + "?" + pa.query, _last_segment(pb.path) + "?" + pb.query
    return _similarity(seg_a, seg_b) >= segment_threshold


def fuzzy_actions_match(oracle_tool_calls, oracle_text, predicted, text_sim_threshold=0.35):
    """Same tool(s), same-ish arguments: URLs matched by domain (or string similarity as a
    fallback), search queries / shell commands by string similarity. Other argument keys
    (maxChars, count, ...) aren't required to match -- those are incidental parameters, not
    the decision being tested. Falls back to text similarity when the oracle turn has no tool calls.
    """
    if predicted is None or "error" in predicted:
        return False

    if oracle_tool_calls:
        pred_calls = predicted.get("tool_calls") or []
        if len(pred_calls) != len(oracle_tool_calls):
            return False
        for a, b in zip(oracle_tool_calls, pred_calls):
            if a["name"] != b["name"]:
                return False
            aa, bb = a["arguments"], b["arguments"]
            if not isinstance(aa, dict) or not isinstance(bb, dict):
                if _canonicalize(aa) != _canonicalize(bb):
                    return False
                continue
            if "url" in aa and "url" in bb:
                if not _urls_match(aa["url"], bb["url"], text_sim_threshold):
                    return False
            elif "query" in aa and "query" in bb:
                if _similarity(aa["query"], bb["query"]) < text_sim_threshold:
                    return False
            elif "command" in aa and "command" in bb:
                if _similarity(aa["command"], bb["command"]) < text_sim_threshold:
                    return False
        return True

    return _similarity(oracle_text, predicted.get("text") or "") >= text_sim_threshold


def cluster_predictions(predictions, match_fn):
    """Group predictions that are mutually equivalent under match_fn. Greedy single-pass
    clustering: each prediction joins the first existing cluster whose representative it
    matches, else starts a new one. Predictions with an "error" (failed calls) are dropped.
    Returns a list of clusters, each a list of indices into `predictions`.
    """
    clusters = []
    for i, p in enumerate(predictions):
        if p is None or "error" in p:
            continue
        for cluster in clusters:
            rep = predictions[cluster[0]]
            if match_fn(rep.get("tool_calls") or [], rep.get("text") or "", p):
                cluster.append(i)
                break
        else:
            clusters.append([i])
    return clusters


def establish_reference(predictions, match_fn):
    """Pick a self-consistent reference action from repeated samples of the same (unmasked)
    context: cluster the samples, and if a majority of them agree with each other, that
    consensus action is a more meaningful ground truth to minimize against than a single
    historical sample (which may itself have been an arbitrary, non-majority choice).
    Returns (reference_prediction_or_None, is_stable, agreement_fraction).
    """
    clusters = cluster_predictions(predictions, match_fn)
    if not clusters:
        return None, False, 0.0
    largest = max(clusters, key=len)
    fraction = len(largest) / len(predictions)
    return predictions[largest[0]], fraction > 0.5, fraction


def render_action(tool_calls, text):
    """Human-readable rendering of an action: one line per tool call, or the reply text."""
    if not tool_calls:
        return text.strip() or "(empty reply)"
    lines = []
    for tc in tool_calls:
        args = tc["arguments"]
        primary = args.get("command") if isinstance(args, dict) else None
        if primary is None and isinstance(args, dict):
            primary = args.get("query")
        if primary is not None:
            lines.append(f"{tc['name']}: {primary}")
        else:
            lines.append(f"{tc['name']}({_canonicalize(args)})")
    return "\n".join(lines)
