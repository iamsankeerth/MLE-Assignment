import os
import json
import traceback
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from typing_extensions import Literal
import litellm
from litellm import completion

from safety import detect_pii, redact_pii, detect_prompt_injection
from retriever import SupportRetriever

# Load environment variables if dot-env is available
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Also search parent directory explicitly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_env = os.path.abspath(os.path.join(current_dir, "..", ".env"))
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
except ImportError:
    pass

class ToolCall(BaseModel):
    action: str = Field(description="The name of the tool to execute (e.g. 'issue_refund', 'reset_password')")
    parameters: Dict[str, Any] = Field(description="Arguments for the tool matching its schema exactly")

class TicketPrediction(BaseModel):
    status: Literal["replied", "escalated"] = Field(
        description="Routing status. Use 'escalated' for high-risk, PII-heavy, out-of-scope, or suspect tickets, otherwise 'replied'."
    )
    product_area: str = Field(
        description="The primary product/domain category (e.g., 'screen', 'tests', 'billing', 'account-management', 'integrations', 'visa-core', 'general', etc.)"
    )
    response: str = Field(
        description="User-facing response grounded ONLY in retrieved documents. Do not echo PII. Cite sources for factual claims."
    )
    justification: str = Field(
        description="Explanation of safety assessment, risk level, routing, and tool calling decisions."
    )
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"] = Field(
        description="The class of support request."
    )
    confidence_score: float = Field(
        description="Calibrated confidence score between 0.0 and 1.0."
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="The risk level based on financial exposure, safety, data sensitivity, and threat presence."
    )
    language: str = Field(
        description="ISO 639-1 language code (e.g., 'en', 'es', 'fr')."
    )
    actions_taken: List[ToolCall] = Field(
        description="List of tool calls to perform. Must be empty [] if no action is warranted or if prerequisite identity check is needed first."
    )

class SupportAgentOrchestrator:
    def __init__(self):
        self.retriever = SupportRetriever()
        
        # Load API specs for tools
        current_dir = os.path.dirname(os.path.abspath(__file__))
        api_spec_path = os.path.abspath(os.path.join(current_dir, "..", "data", "api_specs", "internal_tools.json"))
        self.tools_schema_str = ""
        if os.path.exists(api_spec_path):
            try:
                with open(api_spec_path, 'r', encoding='utf-8') as f:
                    self.tools_schema_str = f.read()
            except Exception as e:
                print(f"[WARNING] Failed to load internal tools spec: {str(e)}")
        else:
            print("[WARNING] internal_tools.json not found.")

    def _determine_model(self) -> str:
        """Determines which model to use based on env variables."""
        # Check environment variables
        if os.environ.get("OPENAI_API_KEY"):
            return "openai/gpt-4o-mini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic/claude-3-5-haiku-20241022"
        elif os.environ.get("GOOGLE_API_KEY"):
            return "gemini/gemini-2.5-flash"
        elif os.environ.get("GROQ_API_KEY"):
            return "groq/llama-3.3-70b-specdec"
        else:
            # If no key is set, check if we can fall back to standard testing env key or raise error
            raise ValueError(
                "Error: No LLM API keys found in the environment!\n"
                "Please configure OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY in your environment/system."
            )

    def process_ticket(self, issue_json: str, subject: str, company: str) -> Dict[str, Any]:
        """
        Executes the full agent pipeline for a single support ticket.
        Returns a dictionary populated with all 14 required columns.
        """
        # Step 1: Pre-process PII Detection & Safety checks
        pii_detected = "false"
        subject_str = subject if isinstance(subject, str) else ""
        
        try:
            messages = json.loads(issue_json)
        except Exception:
            messages = [{"role": "user", "content": str(issue_json)}]
            
        # Standardize message content
        ticket_body = ""
        for m in messages:
            if m.get("role") == "user":
                ticket_body += m.get("content", "") + "\n"
                
        full_text_input = f"Subject: {subject_str}\nBody: {ticket_body}"
        
        # Check PII
        if detect_pii(full_text_input):
            pii_detected = "true"
            
        # Check prompt injection
        if detect_prompt_injection(full_text_input):
            # Fast fail-safe escalation for adversarial prompt injection
            return {
                "issue": issue_json,
                "subject": subject,
                "company": company,
                "status": "escalated",
                "product_area": "general",
                "response": "I apologize, but I cannot fulfill this request due to system security guidelines.",
                "justification": "Adversarial prompt injection attempt detected and safely neutralized.",
                "request_type": "invalid",
                "confidence_score": 1.0,
                "source_documents": "",
                "risk_level": "critical",
                "pii_detected": pii_detected,
                "language": "en",
                "actions_taken": "[]"
            }

        # Step 2: Redact PII in conversation input to LLM to prevent echo
        redacted_messages = []
        for m in messages:
            redacted_messages.append({
                "role": m.get("role", "user"),
                "content": redact_pii(m.get("content", ""))
            })
            
        redacted_subject = redact_pii(subject_str)
        redacted_company = company if isinstance(company, str) else "None"
        
        # Step 3: Document Retrieval via TF-IDF index
        # We query the retriever using subject + last user message
        retrieval_query = f"{redacted_subject} {redacted_messages[-1]['content']}"
        retrieved_docs = self.retriever.retrieve(retrieval_query, company=redacted_company, top_k=3)
        
        source_docs_paths = [doc["path"] for doc in retrieved_docs]
        source_documents_col = "|".join(source_docs_paths)
        
        docs_context = ""
        for idx, doc in enumerate(retrieved_docs):
            docs_context += f"--- DOCUMENT {idx+1} ({doc['path']}) ---\n{doc['content']}\n\n"

        # Step 4: Call LLM with Pydantic Structured Output
        try:
            model = self._determine_model()
            
            # Formulate robust system prompt
            system_prompt = f"""You are a senior customer support triage specialist representing DevPlatform, Claude, and Visa.
Your objective is to classify and safely answer the user support ticket using ONLY the retrieved support documentation.

[SUPPORT DOCUMENTATION RETRIEVED]
{docs_context}

[INTERNAL TOOLS SPECIFICATION]
{self.tools_schema_str}

[RULES & GUIDELINES]
1. GROUNDEDNESS: Your reply MUST be completely grounded in the retrieved documentation above. Do not claim policies or facts not supported by the retrieved files. Cite specific paths (e.g. data/...) when making factual claims. If no documentation is retrieved or it is insufficient, set status to "escalated" or state out-of-scope in a polite response.
2. PII SAFETY: Do not echo back any PII (credit cards, SSNs, phone numbers, emails).
3. DESTRUCTIVE ACTIONS & ID VERIFICATION: Before calling any destructive or sensitive tools (e.g. modify_subscription, issue_refund), you MUST ensure identity is verified.
   - If the user's identity is NOT established as verified in the chat history, you MUST call 'verify_identity' first! Do not call 'issue_refund' or 'modify_subscription' directly.
   - If identity is compromised or theft is suspected, call 'lock_account' and set status to escalated.
4. AUTHORIZATION LIMITS:
   - issue_refund is only valid for transactions <= 90 days and <= $500. Otherwise, you MUST call 'escalate_to_human'.
5. CALIBRATED CONFIDENCE: Calibrate your confidence score (0.0 to 1.0) honestly. If you have to guess, or if information is sparse, lower the score.
6. OUTPUT STRUCTURE: Return a JSON object matching the requested schema strictly. Do not include markdown codeblocks around it; return the raw structured schema format.
"""

            # Combine system prompt with redacted user messages
            llm_messages = [{"role": "system", "content": system_prompt}]
            # We append the conversation messages
            for rm in redacted_messages:
                llm_messages.append({
                    "role": rm["role"],
                    "content": rm["content"]
                })
                
            # Perform completion with structured outputs via LiteLLM with rate limit retries
            import time
            max_retries = 5
            backoff_factor = 2
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = completion(
                        model=model,
                        messages=llm_messages,
                        response_format=TicketPrediction,
                        temperature=0.0
                    )
                    break
                except Exception as e:
                    # Check if it is a rate limit or 429 error
                    err_msg = str(e).lower()
                    is_rate_limit = "429" in err_msg or "rate" in err_msg or "quota" in err_msg or "exhausted" in err_msg
                    if is_rate_limit and attempt < max_retries - 1:
                        sleep_time = (backoff_factor ** attempt) * 3
                        print(f"[RATE_LIMIT] 429 detected, retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                    else:
                        raise e
            
            # Parse structured output
            content = response.choices[0].message.content
            parsed_data = TicketPrediction.model_validate_json(content)
            
            # Serialize the tool calls back to JSON string for the CSV column
            actions_list = []
            for action_call in parsed_data.actions_taken:
                actions_list.append({
                    "action": action_call.action,
                    "parameters": action_call.parameters
                })
            actions_taken_str = json.dumps(actions_list)
            
            # Grounding check: verify that any tool calls match schemas & AUTHORIZATION limits
            # (e.g. if amount > 500 in issue_refund, change action to escalate_to_human)
            modified_actions = actions_list.copy()
            modified_status = parsed_data.status
            
            for act in actions_list:
                name = act.get("action")
                params = act.get("parameters", {})
                if name == "issue_refund":
                    amount = params.get("amount", 0)
                    if amount > 500:
                        # Refund limit exceeded. Switch tool call to escalation.
                        modified_actions = [{
                            "action": "escalate_to_human",
                            "parameters": {
                                "priority": "high",
                                "department": "billing",
                                "summary": f"Refund request for ${amount} exceeds the $500 agent authorization limit."
                            }
                        }]
                        modified_status = "escalated"
                        break
            
            actions_taken_str = json.dumps(modified_actions)

            return {
                "issue": issue_json,
                "subject": subject,
                "company": company,
                "status": modified_status,
                "product_area": parsed_data.product_area,
                "response": parsed_data.response,
                "justification": parsed_data.justification,
                "request_type": parsed_data.request_type,
                "confidence_score": float(parsed_data.confidence_score),
                "source_documents": source_documents_col,
                "risk_level": parsed_data.risk_level,
                "pii_detected": pii_detected,
                "language": parsed_data.language,
                "actions_taken": actions_taken_str
            }
            
        except Exception as e:
            # Fallback in case of LLM failure / Parse failure
            print(f"[WARNING] Single ticket processing error: {str(e)}")
            traceback.print_exc()
            return {
                "issue": issue_json,
                "subject": subject,
                "company": company,
                "status": "escalated",
                "product_area": "general",
                "response": "I apologize, but I am currently unable to complete your request. An agent will follow up shortly.",
                "justification": f"Exception encountered during processing: {str(e)}",
                "request_type": "product_issue",
                "confidence_score": 0.5,
                "source_documents": source_documents_col,
                "risk_level": "medium",
                "pii_detected": pii_detected,
                "language": "en",
                "actions_taken": "[]"
            }
