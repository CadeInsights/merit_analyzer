1. Add OPENAI_API_KEY to .env
2. Parse errors into AssertionState objects (assertion.py)
3. Call cluster_failures() from clustering.py with parsed errors
4. see tests/unit/test_clustering.py for examples

Execution overview:
- get 'failure_reason' and generate dense embeddings with openai small
- cluster them using HDBSCAN
- for each cluster generate name and description using gpt-5
- noise (outliers) put into separate cluster with hardcoded name and description