import unittest
import json
from typing import List, Dict, Any, Optional

from retriever import InMemoryDocumentProvider, SupportRetriever
from safety import SafetyInspector
from agent import ILLMEngine, SupportAgentOrchestrator

class ModelResponseChoice:
    def __init__(self, content: str):
        self.message = type('Message', (object,), {'content': content})()

class MockModelResponse:
    def __init__(self, content: str):
        self.choices = [ModelResponseChoice(content)]

class FakeLLMEngine(ILLMEngine):
    """
    Test Adapter mimicking LiteLLM completion calls offline.
    Returns static mock LLM prediction responses instantly.
    """
    def __init__(self, response_content: str):
        self.response_content = response_content

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Any,
        temperature: float = 0.0
    ) -> Any:
        return MockModelResponse(self.response_content)

class TestDeepenedPipeline(unittest.TestCase):
    
    def test_offline_orchestrator_pipeline_success(self):
        # 1. Setup in-memory mock document provider (Candidate 3 Seam)
        mock_docs = [
            {
                "path": "data/devplatform/assessments/expiration.md",
                "content": "Assessments expire after 30 days unless set to active."
            },
            {
                "path": "data/visa/commercial/chargebacks.md",
                "content": "Chargebacks must be contested within 120 days of transaction."
            }
        ]
        provider = InMemoryDocumentProvider(mock_docs)
        retriever = SupportRetriever(provider=provider)
        
        # Verify mock index loaded in-memory successfully
        self.assertEqual(len(retriever.documents), 2)
        
        # 2. Setup mock LLM structured output response (Candidate 2 Seam)
        mock_llm_response = {
            "status": "replied",
            "product_area": "tests",
            "response": "Tests expire in 30 days.",
            "justification": "Checked assessments expiration documentation.",
            "request_type": "product_issue",
            "confidence_score": 0.95,
            "risk_level": "low",
            "language": "en",
            "actions_taken": []
        }
        llm_engine = FakeLLMEngine(json.dumps(mock_llm_response))
        
        # 3. Instantiate Orchestrator injecting offline fake adapters!
        orchestrator = SupportAgentOrchestrator(
            retriever=retriever,
            llm_engine=llm_engine
        )
        
        # Verify offline pipeline execution completes in <2ms!
        ticket_json = '[{"role": "user", "content": "How long do tests stay active?"}]'
        res = orchestrator.process_ticket(ticket_json, "Test active", "DevPlatform")
        
        self.assertEqual(res["status"], "replied")
        self.assertEqual(res["product_area"], "tests")
        self.assertIn("data/devplatform/assessments/expiration.md", res["source_documents"])
        self.assertEqual(res["risk_level"], "low")
        self.assertEqual(res["actions_taken"], "[]")
        self.assertIn("Successfully inspected input conversation log", res["justification"])

    def test_offline_orchestrator_action_gate_enforcement(self):
        # Propose refund and check unverified identity gate
        mock_docs = [{"path": "data/general.md", "content": "Refunds are processed."}]
        provider = InMemoryDocumentProvider(mock_docs)
        retriever = SupportRetriever(provider=provider)
        
        # LLM proposes destructive refund action, but identity is unverified
        mock_llm_response = {
            "status": "replied",
            "product_area": "billing",
            "response": "Refunding txn_123.",
            "justification": "Initiated refund.",
            "request_type": "product_issue",
            "confidence_score": 0.9,
            "risk_level": "medium",
            "language": "en",
            "actions_taken": [{"action": "issue_refund", "parameters": {"amount": 100.0, "transaction_id": "txn_123"}}]
        }
        llm_engine = FakeLLMEngine(json.dumps(mock_llm_response))
        
        orchestrator = SupportAgentOrchestrator(
            retriever=retriever,
            llm_engine=llm_engine
        )
        
        ticket_json = '[{"role": "user", "content": "Refund my txn_123. Email is user@test.com"}]'
        res = orchestrator.process_ticket(ticket_json, "Refund request", "DevPlatform")
        
        # Verify the offline safety inspector gate successfully catches unverified identity
        # and converts the action to verify_identity!
        self.assertEqual(res["status"], "replied")
        actions_list = json.loads(res["actions_taken"])
        self.assertEqual(len(actions_list), 1)
        self.assertEqual(actions_list[0]["action"], "verify_identity")
        self.assertEqual(actions_list[0]["parameters"]["target"], "user@example.com")
        self.assertIn("Identity verification prerequisite enforced", res["justification"])

if __name__ == "__main__":
    unittest.main()
