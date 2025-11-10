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

## Structure

**Key data abstractions:**
- src/merit_analyzer/types

**Key stateless processors:** 
- src/merit_analyzer/processors

**Key stateful engines:**
- src/merit_analyzer/engines

**Core layer:**
- src/merit_analyzer/core

**Tests:**
- tests/unit

## Design patterns

1. Keep all LLM-related callables async. 

## TODO

1. Build the csv-to-data_objects parser
2. Build the error-to-code mapping
3. Build the output processor