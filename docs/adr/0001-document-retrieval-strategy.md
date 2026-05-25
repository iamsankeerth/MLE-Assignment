# Document Retrieval Strategy

We will use a Classical ML + LLM Hybrid approach (Plan 4). 
For retrieval, we will use TF-IDF or BM25, along with a lightweight classifier for routing tickets. The LLM will be used strictly for response generation. 

**Status:** accepted
**Consequences:** This ensures fast and deterministic retrieval while satisfying the 3-minute time limit, at the cost of limited deep reasoning during the retrieval phase.
