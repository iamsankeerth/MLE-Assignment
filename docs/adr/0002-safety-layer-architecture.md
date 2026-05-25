# Safety Layer Architecture

We will use a separate, lightweight safety layer to pre-process Support Tickets before passing them to the main LLM.
This layer will use Regex to detect PII (like SSNs and Credit Cards) and heuristics or a small, fast model to detect prompt injections and adversarial attacks.

**Status:** accepted
**Consequences:** This allows the system to fail fast and cheaply on malicious or sensitive tickets, saving LLM context and cost, while ensuring deterministic safety checks.
