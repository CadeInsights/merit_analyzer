# Merit

**Description:**
Error analysis for AI projects.

**Expected behavior:**
- user runs the tool in CLI
- the tool takes a .csv file with test results and error messages as input. It tells the user what columns and types are necessary
- backend parses each row inside this file into a data object: test input (any), expected output (any), actual output (any), pass (bool), error_message (str | None), additional_context (str | None). If no error_message is provided - it generates it with LLM
- backend clusters all messages into groups. For each group it generates a name (str) and pattern (str).
- backend predicts what code contributes to each group of errors the most, and provide ideas on fixing them.
- the tool returns an HTML file with the following structure: error group name > problematic behavior > problematic code > relevant test results. The HTML report can be opened directly in a browser and displays a clickable file:// URL in the terminal output.

## Environment Variables

Merit Analyzer requires environment variables to configure the LLM provider. The tool automatically loads variables from a `.env` file in the current working directory, so you can simply create a `.env` file and it will be detected. Alternatively, you can export variables in your shell.

**Note:** The `.env` file is automatically loaded when you run the CLI tool, regardless of whether you use `uvx`, `uv run`, or direct execution. No manual sourcing or loading is required.

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
- `--report <path>`: Where to write the HTML report (default: `merit_report.html`)
- `--model-vendor <vendor>`: Override `MODEL_VENDOR` environment variable (e.g., `openai`, `anthropic`)
- `--inference-vendor <vendor>`: Override `INFERENCE_VENDOR` environment variable (e.g., `openai`, `anthropic`, `aws`, `gcp`)

### Examples

**Using OpenAI (with environment variables):**
```bash
# Set in .env file or export:
# MODEL_VENDOR=openai
# INFERENCE_VENDOR=openai
# OPENAI_API_KEY=sk-...

uv run merit-analyzer analyze tests.csv --report analysis.html
```

**Using Anthropic Direct API:**
```bash
# Set in .env file or export:
# MODEL_VENDOR=anthropic
# INFERENCE_VENDOR=anthropic
# ANTHROPIC_API_KEY=sk-ant-...

uv run merit-analyzer analyze tests.csv --report analysis.html
```

**Using AWS Bedrock:**
```bash
# Set in .env file or export:
# MODEL_VENDOR=anthropic
# INFERENCE_VENDOR=aws
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION=us-east-1

uv run merit-analyzer analyze tests.csv --report analysis.html
```

**Overriding vendors via CLI flags:**
```bash
uv run merit-analyzer analyze tests.csv \
  --model-vendor anthropic \
  --inference-vendor aws \
  --report bedrock_analysis.html
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Merit is open source! We provide this SDK and CLI freely under the MIT License. 

For premium cloud features and enterprise support, visit: https://appmerit.com
