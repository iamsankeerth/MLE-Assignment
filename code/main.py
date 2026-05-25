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

    # Process tickets concurrently in a thread pool
    results = [None] * num_tickets
    max_workers = 10  # Balance speed vs rate limits
    
    print(f"[INFO] Processing tickets concurrently using {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks with their original row index
        futures = {
            executor.submit(
                orchestrator.process_ticket,
                row["Issue"],
                row["Subject"],
                row["Company"]
            ): idx
            for idx, row in df_input.iterrows()
        }
        
        completed_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            completed_count += 1
            try:
                ticket_res = future.result()
                results[idx] = ticket_res
                # Print periodic progress
                if completed_count % 10 == 0 or completed_count == num_tickets:
                    elapsed = time.time() - start_time
                    print(f"[PROGRESS] Completed {completed_count}/{num_tickets} tickets ({elapsed:.1f}s elapsed)")
            except Exception as e:
                print(f"[ERROR] Exception processing row {idx}: {str(e)}")
                # Fail-safe backup row content to avoid empty predictions and program crashes
                row = df_input.iloc[idx]
                results[idx] = {
                    "issue": row["Issue"],
                    "subject": row["Subject"],
                    "company": row["Company"],
                    "status": "escalated",
                    "product_area": "general",
                    "response": "I apologize, but we are experiencing technical difficulties. An agent will follow up shortly.",
                    "justification": f"Technical processing failure: {str(e)}",
                    "request_type": "product_issue",
                    "confidence_score": 0.5,
                    "source_documents": "",
                    "risk_level": "medium",
                    "pii_detected": "false",
                    "language": "en",
                    "actions_taken": "[]"
                }

    # Ensure all slots are filled to prevent index errors
    for i in range(num_tickets):
        if results[i] is None:
            row = df_input.iloc[i]
            results[i] = {
                "issue": row["Issue"],
                "subject": row["Subject"],
                "company": row["Company"],
                "status": "escalated",
                "product_area": "general",
                "response": "I apologize, but we are experiencing technical difficulties. An agent will follow up shortly.",
                "justification": "Fallback due to incomplete slot processing.",
                "request_type": "product_issue",
                "confidence_score": 0.5,
                "source_documents": "",
                "risk_level": "medium",
                "pii_detected": "false",
                "language": "en",
                "actions_taken": "[]"
            }

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
