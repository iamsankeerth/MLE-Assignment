# Support Triage Agent

This directory contains the source code for the Support Triage Agent.

## Setup Instructions

1. **Install Dependencies**:
   Ensure you have Python 3.10+ installed. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, `pip install scikit-learn openai litellm python-dotenv pandas`)*

2. **Configure API Keys**:
   Copy `.env.example` from the repository root to `.env`:
   ```bash
   cp ../.env.example ../.env
   ```
   Open the `.env` file and populate it with your API key:
   - For NVIDIA NIM: `NVIDIA_API_KEY=your_key_here`
   - For OpenAI: `OPENAI_API_KEY=your_key_here`
   - For Anthropic: `ANTHROPIC_API_KEY=your_key_here`
   - For Google Gemini: `GOOGLE_API_KEY=your_key_here`

## How to Run the Agent

Execute the primary entry point to process all tickets in `support_tickets/support_tickets.csv` and write outputs to `support_tickets/output.csv`:

```bash
python main.py
```

This command runs the current submission path by default:
- tickets are processed with `SUPPORT_AGENT_MAX_WORKERS=4` unless overridden
- deterministic safety/routing rules are applied before any LLM call
- the LLM is only used as a bounded fallback for tickets not covered by deterministic handling
- each run prints start/end timers and appends timing metadata to `support_tickets/run_history.csv`

To force strict sequential execution for repeatability checks:

```bash
SUPPORT_AGENT_MAX_WORKERS=1 python main.py
```

To change throughput for local debugging:

```bash
SUPPORT_AGENT_MAX_WORKERS=8 python main.py
```

The latest validated local run processed 89 tickets in 76.93 seconds with 60 replied rows, 29 escalated rows, 6 `verify_identity` calls, 24 `escalate_to_human` calls, 1 `lock_account` call, and 0 fallback/error rows.

## How to Validate Compliance

Run the project format validator to ensure structural and constraint compliance:

```bash
python validate_output.py
```

The validator checks output structure only. The latest generated `support_tickets/output.csv` passes with 89 rows and all 14 required columns.

For repeatability checks, run the offline test suite:

```bash
python test_safety.py
python test_retriever.py
python test_pipeline.py
python test_main.py
```
