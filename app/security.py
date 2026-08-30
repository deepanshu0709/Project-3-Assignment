import re
from typing import Tuple
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize Presidio Analyzer and Anonymizer engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# 14 Heuristic regex patterns for prompt injection detection
INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"(?i)system\s+prompt\s+override",
    r"(?i)you\s+are\s+now\s+in\s+developer\s+mode",
    r"(?i)disregard\s+all\s+safety\s+guidelines",
    r"(?i)jailbreak",
    r"(?i)dan\s+mode",
    r"(?i)as\s+an\s+unfiltered\s+ai",
    r"(?i)repeat\s+the\s+system\s+prompt",
    r"(?i)reveal\s+your\s+hidden\s+instructions",
    r"(?i)bypass\s+all\s+policy\s+checks",
    r"(?i)output\s+the\s+full\s+source\s+code",
    r"(?i)act\s+as\s+root",
    r"(?i)drop\s+database",
    r"(?i)<\s*script\s*>"
]

COMPILED_INJECTION_REGEX = [re.compile(p) for p in INJECTION_PATTERNS]


def scan_for_prompt_injection(text: str) -> bool:
    """
    Checks input text against compiled heuristic regex patterns.
    Returns True if an injection attempt is detected (0-token cost).
    """
    return any(pattern.search(text) for pattern in COMPILED_INJECTION_REGEX)


def anonymize_pii(text: str) -> Tuple[str, bool]:
    """
    Detects and masks PII entities (PHONE_NUMBER, EMAIL_ADDRESS,
    CREDIT_CARD, LOCATION, PERSON) using Presidio.
    Returns:
        (anonymized_text, pii_detected_bool)
    """
    results = analyzer.analyze(
        text=text,
        entities=[
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "LOCATION",
            "PERSON"
        ],
        language="en"
    )

    if not results:
        return text, False

    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text, True