import os
import csv
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent import SupportAgentOrchestrator

# Standardized headers for the MLE challenge
EXPECTED_HEADERS = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification", "confidence_score",
    "source_documents", "risk_level", "pii_detected", "language",
    "actions_taken"
]

def _failure_row(row, justification: str):
    return {
        "issue": row["Issue"],
        "subject": row["Subject"],
        "company": row["Company"],
        "status": "escalated",
        "product_area": "general",
        "response": "I apologize, but I am unable to complete your request safely. A support agent will follow up shortly.",
        "justification": justification,
        "request_type": "product_issue",
        "confidence_score": 0.5,
        "source_documents": "",
        "risk_level": "medium",
        "pii_detected": "false",
        "language": "en",
        "actions_taken": "[]"
    }

def process_ticket_rows(df_input, orchestrator, max_workers: int = 1, progress_every: int = 10):
    num_tickets = len(df_input)
    results = [None] * num_tickets
    completed_count = 0

    if max_workers <= 1:
        print("[INFO] Processing tickets sequentially in deterministic submission mode...")
        for idx, row in df_input.iterrows():
            try:
                results[idx] = orchestrator.process_ticket(
                    row["Issue"],
                    row["Subject"],
                    row["Company"]
                )
            except Exception as e:
                print(f"[ERROR] Exception processing row {idx}: {str(e)}")
                results[idx] = _failure_row(row, "Technical processing failure in sequential submission mode.")
            completed_count += 1
            if completed_count % progress_every == 0 or completed_count == num_tickets:
                print(f"[PROGRESS] Completed {completed_count}/{num_tickets} tickets")
        return results

    print(f"[INFO] Processing tickets concurrently using {max_workers} threads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                orchestrator.process_ticket,
                row["Issue"],
                row["Subject"],
                row["Company"]
            ): idx
            for idx, row in df_input.iterrows()
        }

        for future in as_completed(futures):
            idx = futures[future]
            completed_count += 1
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[ERROR] Exception processing row {idx}: {str(e)}")
                results[idx] = _failure_row(df_input.iloc[idx], "Technical processing failure in parallel debug mode.")
            if completed_count % progress_every == 0 or completed_count == num_tickets:
                print(f"[PROGRESS] Completed {completed_count}/{num_tickets} tickets")
    return results

def main():
    print("[START] Support Triage Agent Execution")
    start_time = time.time()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "..", "support_tickets", "support_tickets.csv")
    output_path = os.path.join(current_dir, "..", "support_tickets", "output.csv")
    
    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return
        
    # Read input tickets using pandas
    try:
        df_input = pd.read_csv(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to read support_tickets.csv: {str(e)}")
        return
        
    num_tickets = len(df_input)
    print(f"[INFO] Loaded {num_tickets} support tickets.")
    
    # Initialize our orchestrator
    try:
        orchestrator = SupportAgentOrchestrator()
    except Exception as e:
        print(f"[ERROR] Failed to initialize orchestrator: {str(e)}")
        return

    max_workers = int(os.environ.get("SUPPORT_AGENT_MAX_WORKERS", "1"))
    results = process_ticket_rows(df_input, orchestrator, max_workers=max_workers)

    # Ensure all slots are filled to prevent index errors
    for i in range(num_tickets):
        if results[i] is None:
            row = df_input.iloc[i]
            results[i] = _failure_row(row, "Fallback due to incomplete slot processing.")

    # Create outputs dataframe
    df_output = pd.DataFrame(results)
    
    # Reorder columns to match standard expected headers exactly
    df_output = df_output[EXPECTED_HEADERS]
    
    # Write to CSV
    try:
        df_output.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"[OK] Predictions written successfully to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write outputs CSV: {str(e)}")
        
    total_time = time.time() - start_time
    print(f"[FINISHED] Processed all tickets in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
