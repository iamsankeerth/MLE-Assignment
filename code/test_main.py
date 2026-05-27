import unittest
import pandas as pd

from main import process_ticket_rows


class RecordingOrchestrator:
    def __init__(self):
        self.seen_subjects = []

    def process_ticket(self, issue_json, subject, company):
        self.seen_subjects.append(subject)
        return {
            "issue": issue_json,
            "subject": subject,
            "company": company,
            "response": f"handled {subject}",
            "product_area": "general",
            "status": "replied",
            "request_type": "product_issue",
            "justification": "deterministic test",
            "confidence_score": 0.8,
            "source_documents": "",
            "risk_level": "low",
            "pii_detected": "false",
            "language": "en",
            "actions_taken": "[]",
        }


class TestMainDeterminism(unittest.TestCase):
    def test_process_ticket_rows_sequential_mode_preserves_order(self):
        df_input = pd.DataFrame([
            {"Issue": "i0", "Subject": "first", "Company": "DevPlatform"},
            {"Issue": "i1", "Subject": "second", "Company": "Claude"},
            {"Issue": "i2", "Subject": "third", "Company": "Visa"},
        ])
        orchestrator = RecordingOrchestrator()

        results = process_ticket_rows(df_input, orchestrator, max_workers=1, progress_every=2)

        self.assertEqual(orchestrator.seen_subjects, ["first", "second", "third"])
        self.assertEqual([row["subject"] for row in results], ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
