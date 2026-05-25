import re
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import Literal

# Regex patterns for standard PII
SSN_PATTERN = re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b')
CC_PATTERN = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
EMAIL_PATTERN = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')

# Words/phrases typical of prompt injections
INJECTION_KEYWORDS = [
    "ignore previous", "ignore all instructions", "system instruction override",
    "you are now", "developer mode", "jailbreak", "do not follow", "forget your guidelines",
    "override security", "act as a", "pretend to be", "bypass filter"
]

def luhn_checksum_ok(number_str: str) -> bool:
    """Validates credit card numbers using the Luhn algorithm."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if not (13 <= len(digits) <= 16):
        return False
    total = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d_double = d * 2
            total += d_double - 9 if d_double > 9 else d_double
        else:
            total += d
    return total % 10 == 0

class PolicyViolation(BaseModel):
    policy_name: str = Field(..., description="The name of the violated policy.")
    message: str = Field(..., description="User-friendly explanation of the violation.")
    remediation: str = Field(..., description="Action to remediate the violation.")

class InspectionResult(BaseModel):
    is_safe: bool = Field(..., description="True if no prompt injection exists and no action violates policy.")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Overall risk level.")
    pii_detected: bool = Field(..., description="True if PII was detected.")
    redacted_subject: str = Field(..., description="Sanitized ticket subject.")
    redacted_conversation: List[Dict[str, str]] = Field(..., description="Sanitized conversation log.")
    suggested_action: Literal["proceed", "escalate", "verify_identity"] = Field(..., description="Immediate operational guidance.")
    justification: str = Field(..., description="Concise justification of the safety disposition.")
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list, description="Sanitized or adjusted tool calls.")

class SafetyInspector:
    """
    A deep module designed as a high-leverage safety seam.
    It encapsulates input inspection, PII scanning, prompt injections, and post-action gates.
    """
    
    @staticmethod
    def detect_pii(text: str) -> bool:
        """Returns True if any PII matches exist in the text."""
        if not text:
            return False
        if SSN_PATTERN.search(text):
            return True
        if EMAIL_PATTERN.search(text):
            return True
        if PHONE_PATTERN.search(text):
            return True
        if CC_PATTERN.search(text):
            for match in CC_PATTERN.finditer(text):
                if luhn_checksum_ok(match.group(0)):
                    return True
        return False

    @staticmethod
    def redact_pii(text: str) -> str:
        """Replaces detected PII in the text with standard placeholders."""
        if not text:
            return text
        
        text = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        text = SSN_PATTERN.sub("[SSN_REDACTED]", text)
        text = PHONE_PATTERN.sub("[PHONE_REDACTED]", text)
        
        # Redact Credit Cards with Luhn double-check
        matches = list(CC_PATTERN.finditer(text))
        for match in reversed(matches):
            match_str = match.group(0)
            if luhn_checksum_ok(match_str):
                placeholder = "[CARD_REDACTED_XXXX]"
                start, end = match.span()
                text = text[:start] + placeholder + text[end:]
                
        return text

    @staticmethod
    def detect_prompt_injection(text: str) -> bool:
        """Heuristically detects common prompt injection strings."""
        if not text:
            return False
        normalized = text.lower()
        for kw in INJECTION_KEYWORDS:
            if kw in normalized:
                return True
        if "system" in normalized and ("instruction" in normalized or "override" in normalized or "ignore" in normalized):
            return True
        if "assistant" in normalized and "act as" in normalized:
            return True
        return False

    def inspect(
        self,
        conversation: List[Dict[str, str]],
        subject: str,
        company: str,
        proposed_actions: Optional[List[Dict[str, Any]]] = None
    ) -> InspectionResult:
        """
        The single, high-leverage safety interface entry point.
        Analyzes conversation history and subject, redacts PII globally,
        filters prompt injections, and validates proposed actions.
        """
        subject_str = subject if isinstance(subject, str) else ""
        company_str = company if isinstance(company, str) else "None"
        
        # 1. Compile full text to analyze input threat vector
        full_user_text = ""
        raw_emails = []
        raw_phones = []
        
        for m in conversation:
            content = m.get("content", "")
            if m.get("role") == "user":
                full_user_text += content + "\n"
                # Keep track of any raw emails/phones for identity verification challenge target
                emails = EMAIL_PATTERN.findall(content)
                if emails:
                    raw_emails.extend(emails)
                phones = PHONE_PATTERN.findall(content)
                if phones:
                    raw_phones.extend(phones)
                    
        full_text_input = f"Subject: {subject_str}\nBody: {full_user_text}"
        
        # 2. Check PII
        pii_detected = self.detect_pii(full_text_input)
        
        # 3. Check Prompt Injection (Fail-fast trigger)
        if self.detect_prompt_injection(full_text_input):
            return InspectionResult(
                is_safe=False,
                risk_level="critical",
                pii_detected=pii_detected,
                redacted_subject="[REDACTED DUE TO SECURITY VIOLATION]",
                redacted_conversation=[{"role": "user", "content": "[REDACTED DUE TO SECURITY VIOLATION]"}],
                suggested_action="escalate",
                justification="Adversarial prompt injection attempt detected and safely neutralized.",
                actions_taken=[{
                    "action": "escalate_to_human",
                    "parameters": {
                        "priority": "urgent",
                        "department": "security",
                        "summary": "Adversarial prompt injection attempt detected on ticket ingestion."
                    }
                }]
            )

        # 4. Redact conversation turns and subject globally
        redacted_subject = self.redact_pii(subject_str)
        redacted_conversation = []
        for m in conversation:
            redacted_conversation.append({
                "role": m.get("role", "user"),
                "content": self.redact_pii(m.get("content", ""))
            })

        # 5. Assess standard risk levels
        # Default risk matches Visa policy or presence of PII/sensitive phrases
        overall_risk: Literal["low", "medium", "high", "critical"] = "low"
        if company_str.lower() == "visa":
            overall_risk = "medium"  # Visa operations are medium risk by default
            if pii_detected:
                overall_risk = "critical"  # Visa PII is promoted to critical risk
        elif pii_detected:
            overall_risk = "high"  # General PII leak is high risk

        # 6. Parse identity verification context
        # We traverse the conversation history looking for confirmation of IDV
        identity_verified = False
        full_chat_history = " ".join(m.get("content", "").lower() for m in conversation)
        idv_keywords = ["identity verified", "user verified", "otp verified", "verification code matched", "identity has been verified"]
        for kw in idv_keywords:
            if kw in full_chat_history:
                identity_verified = True
                break

        # 7. Post-processing action validation
        violations: List[PolicyViolation] = []
        adjusted_actions: List[Dict[str, Any]] = []
        suggested_action: Literal["proceed", "escalate", "verify_identity"] = "proceed"
        justifications = []

        if proposed_actions:
            for act in proposed_actions:
                name = act.get("action")
                params = act.get("parameters", {})
                
                # Check A: Refund limit check ($500)
                if name == "issue_refund":
                    amount = params.get("amount", 0)
                    if amount > 500:
                        violations.append(PolicyViolation(
                            policy_name="refund_limit_exceeded",
                            message=f"Refund request of ${amount} exceeds the $500 agent authorization limit.",
                            remediation="escalate"
                        ))
                        # Override action to human escalation
                        adjusted_actions.append({
                            "action": "escalate_to_human",
                            "parameters": {
                                "priority": "high",
                                "department": "billing",
                                "summary": f"Refund request of ${amount} exceeds the $500 limit."
                            }
                        })
                        suggested_action = "escalate"
                        overall_risk = "high"
                        justifications.append("Refund amount limit violation corrected to escalation.")
                        continue
                        
                # Check B: Destructive action identity verification check
                destructive_actions = {"issue_refund", "modify_subscription", "lock_account"}
                if name in destructive_actions and not identity_verified:
                    # Identity is not established in context! Remediate with verification challenge
                    # Choose standard verification target (extract from PII scans or default)
                    target = "user@example.com"
                    method = "email_otp"
                    if raw_emails:
                        target = raw_emails[0]
                    elif raw_phones:
                        target = raw_phones[0]
                        method = "sms_otp"
                        
                    violations.append(PolicyViolation(
                        policy_name="unverified_identity",
                        message=f"Action '{name}' requires verified user identity.",
                        remediation="verify_identity"
                    ))
                    
                    adjusted_actions.append({
                        "action": "verify_identity",
                        "parameters": {
                            "method": method,
                            "target": target
                        }
                    })
                    suggested_action = "verify_identity"
                    justifications.append("Identity verification prerequisite enforced.")
                    continue
                    
                # Action is safe to proceed as is
                adjusted_actions.append(act)
        
        # Compile final justification
        if not justifications:
            if proposed_actions:
                final_justification = "Successfully validated proposed actions against policies."
            else:
                final_justification = "Successfully inspected input conversation log."
        else:
            final_justification = "; ".join(justifications)

        is_safe = len(violations) == 0

        return InspectionResult(
            is_safe=is_safe,
            risk_level=overall_risk,
            pii_detected=pii_detected,
            redacted_subject=redacted_subject,
            redacted_conversation=redacted_conversation,
            suggested_action=suggested_action,
            violations=violations,
            justification=final_justification,
            actions_taken=adjusted_actions
        )
