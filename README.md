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

## CLI

Run the full pipeline from any project directory:

```bash
uv run merit-analyzer analyze path/to/tests.csv --report reports/analysis.md --model-vendor openai --inference-vendor openai
```

Flags are optional; omit the vendor overrides to rely on existing environment variables.

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
