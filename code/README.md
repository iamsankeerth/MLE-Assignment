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
   - For OpenAI: `OPENAI_API_KEY=your_key_here`
   - For Anthropic: `ANTHROPIC_API_KEY=your_key_here`
   - For Google Gemini: `GOOGLE_API_KEY=your_key_here`

## How to Run the Agent

Execute the primary entry point to process all tickets in `support_tickets/support_tickets.csv` and write outputs to `support_tickets/output.csv`:

```bash
python main.py
```

## How to Validate Compliance

Run the project format validator to ensure structural and constraint compliance:

```bash
python validate_output.py
```
