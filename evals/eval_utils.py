import re
from typing import Any, Mapping, MutableMapping, Optional, Sequence
import verifiers as vf

# Shared Regex Patterns
_BOXED_PATTERN = re.compile(r"\\boxed\s*(?:\{([^}]*)\}|([^\\s]+))", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_CONF_HEADER_PATTERN = re.compile(r"(?i)confidence:\s*(.+)")
_CONF_VALUE_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?|\.\d+")
_ANSWER_PATTERN = re.compile(r"(?i)final\s*answer:\s*(.*)")

def _strip_boxed(value: str) -> str:
    """Extract the contents of the last LaTeX ``\boxed{...}`` wrapper if present."""
    matches = _BOXED_PATTERN.findall(value)
    if not matches:
        return value
    last_braced, last_bare = matches[-1]
    return last_braced or last_bare or value

def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    stripped = _strip_boxed(value.strip())
    return "".join(ch.lower() for ch in stripped if ch.isalnum() or ch.isspace())

def _strip_code_fence(text: str) -> str:
    if text.startswith("```") and text.rstrip().endswith("```"):
        lines = text.strip().splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return text

def normalize_answer(answer: str | None) -> str | None:
    """
    Standard normalization: lowercase, strip, remove spaces, handle simple latex.
    """
    if answer is None:
        return None
    answer = answer.strip()
    # Remove enclosing \text{}
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
    if m is not None:
        answer = m.group("text").strip()

    # Strip spaces and lowercase
    return answer.replace(" ", "").lower()

class SelfPredictionParser(vf.Parser):
    """
    Parser for 'SelfPrediction' format:
    <think>...</think>
    FINAL ANSWER: ...
    CONFIDENCE: 0.8
    """
    def _extract_thinking(self, text: str) -> tuple[str, str]:
        match = _THINK_PATTERN.search(text)
        if not match:
            return "", text
        rationale = match.group(1).strip()
        after = text[match.end() :].strip()
        return rationale, after

    def _extract_answer(self, text: str) -> tuple[str | None, bool]:
        match = _ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).strip(), True

        # Fallback logic if explicit tag is missing
        stripped = text.strip()
        if not stripped:
            return None, False

        # Try to find last line or something meaningful if needed,
        # but for Strict/Calibration tasks we might prefer explicit failure
        # or use a softer fallback. Using the one from self_prediction.py:

        # Remove think block if present but not caught above (e.g. only think block)
        # (This logic was in self_prediction.py _fallback_answer)
        lower = stripped.lower()
        start = lower.find("<think>")
        end = lower.find("</think>")
        if start != -1:
            if end != -1 and end > start:
                stripped = stripped[:start] + stripped[end + len("</think>") :]
            else:
                stripped = stripped[:start]
            stripped = stripped.strip()

        if not stripped:
            return None, False

        # Return last non-empty line
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        return (lines[-1] if lines else None), False

    def _extract_confidence(self, text: str) -> tuple[float | None, bool]:
        header = _CONF_HEADER_PATTERN.search(text)
        if not header:
            return 0.5, False
        tail = header.group(1)
        value_match = _CONF_VALUE_PATTERN.search(tail)
        if not value_match:
            return 0.5, False
        try:
            return float(value_match.group(0)), True
        except ValueError:
            return None, False

    def parse(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = _strip_code_fence(text.strip())
        rationale, payload = self._extract_thinking(cleaned)
        answer, answer_from_header = self._extract_answer(payload)
        confidence, confidence_from_header = self._extract_confidence(payload)

        report: dict[str, Any] = {"format_ok": answer_from_header and confidence_from_header}
        if rationale:
            report["rationale"] = rationale
        if answer:
            report["answer"] = answer
        if confidence is not None:
            report["confidence"] = confidence
        return report or None

class BasicSanityParser(vf.Parser):
    """
    Robust parser for sanity checks.
    Expects the answer to be in the text.
    If 'FINAL ANSWER:' is present, uses it.
    Otherwise, tries to extract the last number or return the whole stripped text.
    """
    def parse(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        cleaned = _strip_code_fence(text.strip())

        # Try finding FINAL ANSWER first
        match = _ANSWER_PATTERN.search(cleaned)
        if match:
             return {"answer": match.group(1).strip(), "parsed_by": "regex_tag"}

        # Fallback: Just take the last non-empty line
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            return {"answer": lines[-1], "parsed_by": "last_line"}

        return {"answer": cleaned, "parsed_by": "raw"}

def grade_exact(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def grade_numeric(pred: str, gold: str, tol: float = 1e-6) -> float:
    try:
        p_val = float(str(pred).replace(",", ""))
        g_val = float(str(gold).replace(",", ""))
        return 1.0 if abs(p_val - g_val) <= tol else 0.0
    except (ValueError, TypeError):
        return 0.0
