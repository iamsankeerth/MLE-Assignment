import re

# Regex patterns for standard PII
SSN_PATTERN = re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b')
# Matches typical credit card numbers with or without spaces/hyphens
CC_PATTERN = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
EMAIL_PATTERN = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')

# Words/phrases typical of prompt injections
INJECTION_KEYWORDS = [
    "ignore previous", "ignore all instructions", "system instruction override",
    "you are now", "developer mode", "jailbreak", "do not follow", "forget your guidelines",
    "override security", "act as a", "pretend to be", "bypass filter"
]

def detect_pii(text: str) -> bool:
    """Returns True if any PII pattern is found in the text."""
    if not text:
        return False
    if SSN_PATTERN.search(text):
        return True
    if CC_PATTERN.search(text):
        # Additional Luhn algorithm check to avoid false positives on random number runs
        for match in CC_PATTERN.finditer(text):
            digits = [int(c) for c in match.group(0) if c.isdigit()]
            if 13 <= len(digits) <= 16:
                # Simple Luhn validation
                total = 0
                reverse_digits = digits[::-1]
                for i, d in enumerate(reverse_digits):
                    if i % 2 == 1:
                        d_double = d * 2
                        total += d_double - 9 if d_double > 9 else d_double
                    else:
                        total += d
                if total % 10 == 0:
                    return True
    if EMAIL_PATTERN.search(text):
        return True
    if PHONE_PATTERN.search(text):
        return True
    return False

def redact_pii(text: str) -> str:
    """Replaces PII in the text with generic placeholders to prevent echoing."""
    if not text:
        return text
    
    # Redact Email
    text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    
    # Redact SSN
    text = SSN_PATTERN.sub("[SSN_REDACTED]", text)
    
    # Redact Credit Card with Luhn double-check to avoid destroying non-card numbers
    matches = list(CC_PATTERN.finditer(text))
    # Process from right to left to not corrupt indices of subsequent matches
    for match in reversed(matches):
        match_str = match.group(0)
        digits = [int(c) for c in match_str if c.isdigit()]
        if 13 <= len(digits) <= 16:
            # Luhn validation
            total = 0
            reverse_digits = digits[::-1]
            for i, d in enumerate(reverse_digits):
                if i % 2 == 1:
                    d_double = d * 2
                    total += d_double - 9 if d_double > 9 else d_double
                else:
                    total += d
            if total % 10 == 0:
                # Replace with redacted form containing last 4 digits
                last_four = "".join(match_str.strip()[-4:])
                placeholder = f"[CARD_REDACTED_XXXX]"
                start, end = match.span()
                text = text[:start] + placeholder + text[end:]
                
    # Redact Phone
    text = PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
    
    return text

def detect_prompt_injection(text: str) -> bool:
    """Simple heuristic to detect prompt injection attempts."""
    if not text:
        return False
    normalized = text.lower()
    for kw in INJECTION_KEYWORDS:
        if kw in normalized:
            return True
    # Look for patterns attempting system command or roleplay overrides
    if "system" in normalized and ("instruction" in normalized or "override" in normalized or "ignore" in normalized):
        return True
    if "assistant" in normalized and "act as" in normalized:
        return True
    return False
