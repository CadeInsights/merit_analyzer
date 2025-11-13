# Merit

**Description:**
Error analysis for AI projects.

**Expected behavior:**
- user runs the tool in CLI
- the tool takes a .csv file with test results and error messages as input. It tells the user what columns and types are necessary
- backend parses each row inside this file into a data object: test input (any), expected output (any), actual output (any), pass (bool), error_message (str | None), additional_context (str | None). If no error_message is provided - it generates it with LLM
- backend clusters all messages into groups. For each group it generates a name (str) and pattern (str).
- backend predicts what code contributes to each group of errors the most, and provide ideas on fixing them.
- the tool returns a Markdown file with the following structure: error group name > problematic behavior > problematic code > relevant test results

## Environment Variables

Merit Analyzer requires environment variables to configure the LLM provider. You can set these in a `.env` file in your project root or export them in your shell.

### Required Variables

**Core Configuration:**
- `MODEL_VENDOR`: The model family to use. Options: `openai` or `anthropic`
- `INFERENCE_VENDOR`: The inference provider to use. Options depend on `MODEL_VENDOR`:
  - For `openai`: `openai`
  - For `anthropic`: `anthropic`, `aws`, or `gcp`

### Provider-Specific Variables

**OpenAI:**
```bash
MODEL_VENDOR=openai
INFERENCE_VENDOR=openai
OPENAI_API_KEY=your_openai_api_key_here
```

**Anthropic (Direct API):**
```bash
MODEL_VENDOR=anthropic
INFERENCE_VENDOR=anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**AWS Bedrock:**
```bash
MODEL_VENDOR=anthropic
INFERENCE_VENDOR=aws
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1  # Optional, defaults to us-east-1
```

**Google Cloud Vertex AI:**
```bash
MODEL_VENDOR=anthropic
INFERENCE_VENDOR=gcp
GOOGLE_CLOUD_PROJECT=your_project_id  # or use ANTHROPIC_VERTEX_PROJECT_ID
CLOUD_ML_REGION=us-east5  # Optional, defaults to us-east5
```

## CLI

Run the full pipeline from any project directory:

```bash
uv run merit-analyzer analyze path/to/tests.csv
```

### Command Options

**Basic Usage:**
```bash
uv run merit-analyzer analyze <csv_path> [OPTIONS]
```

**Arguments:**
- `csv_path` (required): Path to CSV file containing test results with columns: `case_input`, `reference_value`, `output_for_assertions`, `passed`, `error_message`

**Options:**
- `--report <path>`: Where to write the Markdown report (default: `merit_report.md`)
- `--model-vendor <vendor>`: Override `MODEL_VENDOR` environment variable (e.g., `openai`, `anthropic`)
- `--inference-vendor <vendor>`: Override `INFERENCE_VENDOR` environment variable (e.g., `openai`, `anthropic`, `aws`, `gcp`)

### Examples

**Using OpenAI (with environment variables):**
```bash
# Set in .env file or export:
# MODEL_VENDOR=openai
# INFERENCE_VENDOR=openai
# OPENAI_API_KEY=sk-...

uv run merit-analyzer analyze tests.csv --report analysis.md
```

**Using Anthropic Direct API:**
```bash
# Set in .env file or export:
# MODEL_VENDOR=anthropic
# INFERENCE_VENDOR=anthropic
# ANTHROPIC_API_KEY=sk-ant-...

uv run merit-analyzer analyze tests.csv --report analysis.md
```

**Using AWS Bedrock:**
```bash
# Set in .env file or export:
# MODEL_VENDOR=anthropic
# INFERENCE_VENDOR=aws
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=us-east-1

uv run merit-analyzer analyze tests.csv --report analysis.md
```

**Overriding vendors via CLI flags:**
```bash
uv run merit-analyzer analyze tests.csv \
  --model-vendor anthropic \
  --inference-vendor aws \
  --report bedrock_analysis.md
```

Note: When using CLI flags to override vendors, you still need the appropriate API keys set in your environment variables.

## Structure

**Key data abstractions:**
- src/merit_analyzer/types

**Key stateless processors:** 
- src/merit_analyzer/processors

**Key stateful engines:**
- src/merit_analyzer/engines

**Core layer:**
- src/merit_analyzer/core

**Interfaces:**
- src/merit_analyzer/interface

**Tests:**
- tests/unit
- tests/integrations

## Design patterns

1. Keep all LLM-related callables async. 

## TODO
1. Test end-to-end with Bedrock

## License

This software is licensed under the Merit Commercial Evaluation License (v1.0).

You may use this software for evaluation and testing purposes. Continued use beyond the evaluation period requires a commercial license from Merit.

See [LICENSE](LICENSE) for full terms.

For commercial licensing or support, visit: https://appmerit.com
