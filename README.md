# Merit Analyzer

**AI system test failure analysis and recommendation engine**

Merit Analyzer is a universal Python SDK that analyzes **any AI system's** test results to provide specific, actionable recommendations for fixing failures. It automatically discovers your system's architecture, identifies failure patterns, and generates targeted fixes across code, prompts, and agent architecture - whether you're building chatbots, RAG systems, code generators, or any other AI application.

## 🚀 Key Features

- **Universal AI System Support**: Works with chatbots, RAG systems, code generators, agents, and more
- **Intelligent Schema Discovery**: Automatically understands your system's data patterns and validation rules
- **Hierarchical Pattern Detection**: Clusters failures using input similarity, output patterns, and delta analysis
- **AI-Powered Root Cause Analysis**: Uses LLM to discover why failures occur and generate failure context
- **System-Agnostic Recommendations**: Provides fixes tailored to your specific AI system type
- **Multiple Output Formats**: JSON, Markdown, HTML reports
- **Framework Agnostic**: Works with any AI framework (LangChain, LlamaIndex, Anthropic, OpenAI, etc.)

## 🌐 Universal AI System Support

Merit Analyzer works with **any type of AI system**:

### **Chatbots & Conversational AI**
- Analyzes conversation flows, response quality, and user intent handling
- Detects missing greetings, inappropriate responses, or context loss
- Recommends prompt improvements and conversation logic fixes

### **RAG (Retrieval-Augmented Generation) Systems**
- Identifies missing citations, low confidence responses, or retrieval failures
- Analyzes document processing and knowledge base integration
- Suggests improvements to retrieval strategies and response generation

### **Code Generators & AI Development Tools**
- Detects code quality issues, missing error handling, or incorrect implementations
- Analyzes code structure, imports, and best practices
- Recommends code improvements and development workflow fixes

### **Multi-Modal AI Systems**
- Handles text, image, audio, and file inputs/outputs
- Analyzes cross-modal consistency and processing pipelines
- Identifies issues in data transformation and output formatting

### **Custom AI Applications**
- Automatically discovers your system's unique patterns and requirements
- Adapts analysis to your specific data schemas and validation rules
- Provides recommendations tailored to your architecture and use case

## 📋 What It Does

1. **Discovers System Schema** (Standard API - Fast): Automatically understands your AI system's data patterns and validation rules
2. **Analyzes Test Results**: Takes your test failures and groups them into meaningful patterns using hierarchical clustering
3. **Maps Architecture** (Standard API - Fast): Infers system components, data flow, and decision points from project structure
4. **Analyzes Patterns** (Claude Agent SDK - Intelligent): For each failure pattern:
   - Uses **Grep** to search for relevant functions, classes, and keywords in your codebase
   - Uses **Glob** to find files matching patterns (e.g., agents, prompts, configs)
   - Uses **Read** to examine actual code files and trace failure cascades (READ-ONLY)
   - Identifies root causes from actual code analysis
   - Generates specific, actionable recommendations
5. **Prioritizes Actions**: Deduplicates and ranks recommendations by impact and effort

## 🎯 Target Users

- **AI/ML Engineers** building any type of AI system (chatbots, RAG, code generators, etc.)
- **Development Teams** using any AI framework (LangChain, LlamaIndex, Anthropic, OpenAI, etc.)
- **QA Engineers** running test suites on AI applications
- **DevOps Teams** debugging AI system failures in production
- **Product Teams** improving AI system reliability and user experience

## 📦 Installation

### Prerequisites

Merit Analyzer uses **Claude Agent SDK** which requires:

1. **Python 3.10+**
2. **Node.js** (for Claude Code CLI)
3. **Claude Code CLI**: `npm install -g @anthropic-ai/claude-code`

### Install Merit Analyzer

```bash
pip install merit-analyzer
```

### Verify Installation

The Claude Code CLI is required for the SDK to navigate your codebase. If you see errors about CLI not found, ensure you've installed it:

```bash
npm install -g @anthropic-ai/claude-code
claude-code --version  # Should display version
```

## ⚡ Performance & Architecture

Merit Analyzer uses a **hybrid architecture** for optimal speed and intelligence:

### **Standard Anthropic API** (Fast & Scalable)
- ✅ **Schema Discovery**: ~45s for 60 tests
- ✅ **Architecture Inference**: ~12s 
- ⚡ **Parallel Execution**: Multiple patterns analyzed concurrently

### **Claude Agent SDK** (Intelligent Code Analysis)
- 🔍 **Lazy Loading**: SDK only loads when analyzing patterns (no startup overhead)
- 🔍 **Intelligent Search**: Uses Grep to find relevant code based on test keywords
- 🔍 **Smart Navigation**: Uses Glob to locate files matching patterns
- 📖 **Code Reading**: Uses Read to analyze actual code files (READ-ONLY)
- 🎯 **Root Cause Tracing**: Traces failure cascades through code paths

**Why This Architecture?**
- The Claude Agent SDK hangs on import if loaded eagerly
- We use lazy loading to keep startup instant
- Standard API handles fast inference tasks (schema, architecture)
- Agent SDK handles complex code navigation (pattern analysis)

### API Provider Options

Merit Analyzer supports two ways to access Claude models:

#### Option 1: Anthropic Direct (Default)

Use your Anthropic API key directly:

```python
analyzer = MeritAnalyzer(
    project_path="./my-ai-app",
    api_key="sk-ant-..."  # Your Anthropic API key
)
```

#### Option 2: Amazon Bedrock

Use Claude models via Amazon Bedrock:

```python
analyzer = MeritAnalyzer(
    project_path="./my-ai-app",
    api_key="",  # Not needed - uses AWS credentials
    provider="bedrock"
)
```

**Bedrock Prerequisites:**
1. **Enable model access** in Amazon Bedrock console
2. **Configure AWS credentials** via AWS CLI:
   ```bash
   aws configure
   ```
   Or set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_REGION=us-east-1  # Or your preferred region
   ```

## 🚀 Quick Start

### Universal Analysis (Recommended)

The universal analyzer works with **ANY AI system** - chatbots, RAG systems, code generators, agents, etc. It automatically discovers your system type and adapts its analysis accordingly.

```python
from merit_analyzer import MeritAnalyzer, TestResult

# Create test results for ANY AI system
test_results = [
    # Chatbot example
    TestResult(
        test_id="chat_001",
        input={"message": "Hello, how are you?"},
        expected_output={"response": "Hello! I'm doing well.", "tone": "friendly"},
        actual_output={"response": "Hi there!", "tone": "friendly"},
        status="failed",
        category="greeting"
    ),
    # Code generator example  
    TestResult(
        test_id="code_001",
        input={"prompt": "Create a sort function", "language": "python"},
        expected_output={"code": "def sort_list(lst): return sorted(lst)", "runs": True},
        actual_output={"code": "def sort_list(lst): lst.sort()", "runs": True},
        status="passed",
        category="functions"
    ),
    # RAG system example
    TestResult(
        test_id="rag_001", 
        input={"question": "What is the refund policy?", "context": "policy_doc.pdf"},
        expected_output={"answer": "30-day money back guarantee", "has_citation": True},
        actual_output={"answer": "We offer refunds", "has_citation": False},
        status="failed",
        category="policies"
    )
]

# Initialize analyzer (Anthropic direct)
analyzer = MeritAnalyzer(
    project_path="./my-ai-app",
    api_key="your-anthropic-api-key"
)

# OR: Initialize with Amazon Bedrock
analyzer = MeritAnalyzer(
    project_path="./my-ai-app",
    api_key="",  # Not needed - uses AWS credentials
    provider="bedrock",
    config={"aws_region": "us-west-2"}  # Optional: defaults to us-east-1
)

# Run analysis (works with ANY AI system)
report = analyzer.analyze(test_results)

# Save report
analyzer.save_report(report, "universal_analysis_report.json")
```

### 1. Command Line Usage

```bash
# Basic analysis
merit-analyze --test-results test_results.json --api-key sk-ant-...

# With custom project path
merit-analyze --project-path ./my-ai-app --test-results results.json --output analysis.json

# Export recommendations separately
merit-analyze --test-results results.json --export-recommendations recs.md
```

### 2. Load Test Results from File

```python
from merit_analyzer import MeritAnalyzer

# Load from JSON file
analyzer = MeritAnalyzer(
    project_path="./my-ai-app",
    api_key="sk-ant-..."
)

# The analyzer can parse various formats
report = analyzer.analyze("test_results.json")
```

## 📊 Understanding the Output

### Analysis Report

The analysis report contains:

- **Summary Statistics**: Test counts, pass rates, patterns found
- **Failure Patterns**: Grouped failures with root cause analysis
- **Recommendations**: Specific, actionable fixes prioritized by impact
- **Action Plan**: Step-by-step implementation guide

### Example Output

```
📊 ANALYSIS SUMMARY
====================================
Total tests: 15
Passed: 8
Failed: 7
Pass rate: 53.3%
Patterns found: 3
Recommendations: 12

🔍 FAILURE PATTERNS
------------------------------------
pricing_vague_responses: 4 failures (26.7%)
  Root cause: Prompt template lacks specific pricing examples

timeout_issues: 2 failures (13.3%)
  Root cause: No timeout handling for complex requests

validation_errors: 1 failures (6.7%)
  Root cause: Missing input validation

💡 TOP RECOMMENDATIONS
------------------------------------
1. Add specific pricing examples to prompt template
   Type: Prompt
   Effort: 30 minutes
   Impact: Fixes 4 tests in pricing_vague_responses

2. Implement timeout handling for complex requests
   Type: Code
   Effort: 1 hour
   Impact: Fixes 2 tests in timeout_issues
```

## 🔧 Configuration

### Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export MERIT_PROJECT_PATH="./my-ai-app"
export MERIT_PROVIDER="anthropic"
```

### Configuration File

Create `merit_config.yaml`:

```yaml
project_path: "./my-ai-app"
api_key: "sk-ant-..."
provider: "anthropic"
model: "claude-sonnet-4-5-20250929"  # Latest Claude Sonnet 4.5
max_tokens: 4096
min_cluster_size: 2
max_patterns: 10
verbose: true
```

Use with:

```bash
merit-analyze --config merit_config.yaml --test-results results.json
```

## 📁 Project Structure

```
merit-analyzer/
├── merit_analyzer/
│   ├── core/           # Core analysis engine
│   ├── discovery/      # Project scanning and framework detection
│   ├── analysis/       # Root cause analysis and Claude integration
│   ├── recommendations/ # Recommendation generation and prioritization
│   ├── models/         # Data models
│   └── cli.py         # Command-line interface
├── examples/           # Usage examples
├── tests/             # Test suite
└── docs/              # Documentation
```

## 🧪 Supported Test Formats

Merit Analyzer supports multiple test result formats:

- **JSON**: Standard test result format
- **CSV**: Comma-separated values
- **pytest JSON**: pytest --json-report output
- **JUnit XML**: JUnit test results

### JSON Format

```json
[
  {
    "test_id": "test_001",
    "test_name": "pricing_inquiry",
    "input": "How much does the pro plan cost?",
    "expected_output": "$49/month",
    "actual_output": "We have various pricing tiers",
    "status": "failed",
    "failure_reason": "Response too vague",
    "category": "pricing",
    "tags": ["pricing", "pro_plan"],
    "execution_time_ms": 1250,
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

## 🏗️ Architecture

Merit Analyzer follows a layered architecture that processes test failures through multiple stages of analysis, from pattern detection to actionable recommendations.

```
┌─────────────────────────────────────────────────────────────┐
│                     Merit Analyzer SDK                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Input Layer                                       │  │
│  │    - Test result parser                              │  │
│  │    - Schema validation (Pydantic)                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Discovery Layer (Standard API - Fast ~1min)      │  │
│  │    - Schema Discovery: Analyze test data patterns    │  │
│  │    - Project Scanner: Analyze file structure         │  │
│  │    - Framework Detector: Identify AI frameworks      │  │
│  │    - Architecture Inference: Infer system design     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. Pattern Detection (Clustering - instant)          │  │
│  │    - Hierarchical clustering (TF-IDF + DBSCAN)       │  │
│  │    - Input similarity → Output patterns → Deltas     │  │
│  │    - Pattern merging for scalability                 │  │
│  │    - AI-powered pattern naming                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. Pattern Analysis (Claude Agent SDK - Intelligent) │  │
│  │    🔍 STEP 1: Find Relevant Code                     │  │
│  │       - Grep: Search for keywords from test errors   │  │
│  │       - Glob: Find files matching patterns           │  │
│  │    📖 STEP 2: Analyze Code (READ-ONLY)               │  │
│  │       - Read: Examine actual code files              │  │
│  │       - Trace failure cascades through code paths    │  │
│  │       - Identify root causes from actual code        │  │
│  │    💡 STEP 3: Generate Recommendations               │  │
│  │       - Code fixes with specific locations           │  │
│  │       - Prompt improvements with exact text          │  │
│  │       - Config/design changes                        │  │
│  │    ⚡ Parallel execution: Multiple patterns at once  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 5. Recommendation Engine                             │  │
│  │    - Deduplication: Remove similar recommendations   │  │
│  │    - Prioritization: Rank by impact & effort         │  │
│  │    - Dependency analysis: Identify blockers          │  │
│  │    - Multi-format output (JSON/Markdown/CLI)         │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 6. Output Layer                                      │  │
│  │    - Actionable analysis report                      │  │
│  │    - Prioritized recommendations                     │  │
│  │    - Root causes with code references                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↑                                    ↑
         │                                    │
    User's API Key                    User's Codebase
   (Anthropic Direct or           (Claude Agent SDK - lazy loaded)
    Amazon Bedrock)                (Grep/Glob/Read tools - READ-ONLY)
```

### Technical Stack

#### Core Dependencies
- **Python**: 3.10+
- **Claude Agent SDK**: `claude-agent-sdk` (provides Read, Grep, Glob tools for codebase navigation)
- **Claude Code CLI**: `npm install -g @anthropic-ai/claude-code` (required by SDK)
- **Required packages**:
  - `claude-agent-sdk>=0.1.0` (for agentic codebase analysis with file system access)
  - `scikit-learn` (for pattern clustering)
  - `numpy` (for similarity calculations)
  - `pydantic` (for data validation)
  - `rich` (for CLI output formatting)
  - `GitPython` (optional, for git integration)

#### Project Structure
```
merit-analyzer/
├── merit_analyzer/
│   ├── __init__.py
│   ├── core/
│   │   ├── analyzer.py           # Main Analyzer class
│   │   ├── test_parser.py        # Parse test results
│   │   ├── pattern_detector.py   # Cluster failures
│   │   └── config.py              # Configuration management
│   ├── discovery/
│   │   ├── project_scanner.py    # Quick project analysis
│   │   ├── framework_detector.py # Detect LangChain, etc.
│   │   └── code_mapper.py        # Map patterns to code
│   ├── analysis/
│   │   ├── claude_agent.py       # Claude Code integration
│   │   ├── root_cause.py         # Root cause analysis
│   │   └── comparative.py        # Compare pass/fail tests
│   ├── recommendations/
│   │   ├── generator.py          # Generate recommendations
│   │   ├── prioritizer.py        # Prioritize fixes
│   │   └── formatter.py          # Format output
│   └── models/
│       ├── test_result.py        # Pydantic models
│       ├── pattern.py
│       ├── recommendation.py
│       └── report.py
├── tests/
├── examples/
├── docs/
├── setup.py
└── README.md
```

## 📚 API Reference

### TestResult Schema

The `TestResult` model defines the structure for test data input to Merit Analyzer.

#### Required Fields
- `test_id` (str): Unique identifier for the test
- `input` (str | dict): Input to the AI system (flexible: string or structured dict)
- `actual_output` (str | dict): Actual response from system (flexible: string or structured dict)
- `status` (str): Test result status - **must be one of**: `"passed"`, `"failed"`, `"error"`, `"skipped"`

#### Optional Fields
- `test_name` (str, optional): Human-readable test name
- `expected_output` (str | dict, optional): Expected response (flexible: string or structured dict)
- `failure_reason` (str, optional): Why the test failed ⚠️ **Highly recommended for better pattern detection**
- `category` (str, optional): Test category (e.g., 'pricing', 'support')
- `tags` (list[str], optional): Tags for grouping (defaults to empty list)
- `execution_time_ms` (int, optional): Test execution time in milliseconds
- `timestamp` (str, optional): ISO timestamp
- `trace` (dict, optional): Execution trace/logs
- `metadata` (dict, optional): Additional metadata

> **💡 Tip**: While `failure_reason` is optional, providing it significantly improves pattern detection quality. The system will auto-generate basic failure context when missing, but explicit failure reasons lead to more accurate clustering.

#### Example Usage

```python
# Minimal test result (only required fields) - String format
test_result = TestResult(
    test_id="test_001",
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    status="passed"
)

# Structured data format (dict) - for complex AI systems
test_result = TestResult(
    test_id="test_002",
    input={"query": "pricing", "user_tier": "enterprise"},
    expected_output={"price": "$99/mo", "features": ["unlimited", "support"]},
    actual_output={"price": "$99/mo", "features": ["unlimited"]},
    status="failed",
    failure_reason="Missing 'support' feature in response"
)

# Complete test result with all optional fields
test_result = TestResult(
    test_id="test_003",
    test_name="Greeting Test",
    input="Generate a greeting",
    expected_output="Hello! How can I help you today?",
    actual_output="Hello! How can I help you?",
    status="failed",
    failure_reason="Missing sign-off",
    category="greeting",
    tags=["user_interaction", "formatting"],
    execution_time_ms=1500,
    timestamp="2024-01-15T10:30:00Z",
    metadata={"environment": "production", "version": "1.2.3"}
)
```

## 🔍 How It Works

### 1. Input Processing
- Parses and validates test results
- Detects failure patterns using clustering algorithms
- Groups similar failures together

### 2. Architecture Discovery (Agentic Navigation with Shared Session)
- Uses **Claude Agent SDK** with Read, Grep, Glob tools to navigate your codebase
- **Shared session**: Starts a persistent session to maintain context across operations
- **Read tool**: Examines actual code files to understand structure
- **Grep tool**: Searches for patterns (LLM calls, agents, prompts, etc.)
- **Glob tool**: Finds relevant files matching patterns
- Claude remembers files read and architecture discovered for subsequent steps
- Detects frameworks (LangChain, LlamaIndex, etc.)
- Maps data flow and component relationships
- **READ-ONLY**: All navigation is read-only - no code is modified

### 3. Pattern Mapping (Sequential with Shared Context)
- Leverages **shared session context** from architecture discovery
- Claude remembers the architecture it already learned - no re-discovery needed
- Maps failure patterns to relevant code files sequentially to maintain context
- **Grep/Glob tools**: Find files related to each pattern
- More efficient than starting fresh for each pattern

### 4. Pattern Analysis (Parallel Execution)
- Uses **Claude Agent SDK** to analyze patterns in parallel for scalability
- **Read tool**: Reads actual code files to find issues
- **Grep tool**: Traces call chains and searches for related code
- **Glob tool**: Discovers all relevant files in failure paths
- Traces failure cascades through code paths (root causes and cascading issues)
- Compares failing vs passing tests
- Identifies root causes and code issues with specific file:line locations
- Each pattern analyzed concurrently with explicit architecture context

### 5. Recommendation Generation (READ-ONLY)
- Generates specific, actionable fixes based on actual code analysis
- Prioritizes by impact and effort
- Provides implementation details and code diffs
- **READ-ONLY mode**: Tells you how to fix issues but does NOT modify code

## 🧠 Context Management

Merit Analyzer uses **intelligent session management** for optimal performance:

- **Shared session** for sequential operations (architecture discovery → pattern mapping)
  - Claude maintains a single conversation context
  - Remembers files read, architecture discovered, and patterns identified
  - Eliminates redundant file reads and re-discovery
  - More efficient and provides better analysis quality

- **Parallel execution** for pattern analysis
  - Multiple patterns analyzed concurrently for scalability
  - Each analysis gets explicit context (architecture, code locations)
  - Handles large test suites (1000+ tests) efficiently

- **Automatic cleanup**
  - Session closed after analysis completes
  - Ensures no resource leaks
  - Fresh start for each analysis run

## 🎯 Use Cases

### 1. Debugging AI Agent Failures
```python
# Analyze why your pricing agent is giving vague responses
report = analyzer.analyze(pricing_test_results)
# Get specific prompt improvements and code fixes
```

### 2. Improving Test Coverage
```python
# Find patterns in test failures to improve coverage
patterns = report.patterns
for pattern_name, pattern in patterns.items():
    print(f"Pattern: {pattern_name} - {pattern.failure_count} failures")
```

### 3. Performance Optimization
```python
# Identify timeout and performance issues
timeout_patterns = [p for p in patterns.values() if "timeout" in p.name]
```

### 4. Prompt Engineering
```python
# Get recommendations for improving prompts
prompt_recs = [r for r in report.recommendations if r.type.value == "prompt"]
```

## 🛠️ Advanced Usage

### Custom Configuration

```python
from merit_analyzer import MeritAnalyzer

config = {
    "min_cluster_size": 3,
    "max_patterns": 15,
    "similarity_threshold": 0.4,
    "verbose": True
}

analyzer = MeritAnalyzer(
    project_path="./my-ai-app",
    api_key="sk-ant-...",
    config=config
)
```

### Integration with Test Frameworks

```python
# pytest integration example
import pytest
from merit_analyzer import MeritAnalyzer, TestResult

class MeritAnalyzerPlugin:
    def __init__(self):
        self.test_results = []
        self.analyzer = MeritAnalyzer(
            project_path=".",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        
        if call.when == "call":
            test_result = TestResult(
                test_id=item.nodeid,
                test_name=item.name,
                input=item.funcargs.get('input', ''),
                actual_output=item.funcargs.get('output', ''),
                status="passed" if report.passed else "failed",
                failure_reason=str(report.longrepr) if report.failed else None
            )
            self.test_results.append(test_result)
    
    def pytest_sessionfinish(self, session):
        if any(t.status == "failed" for t in self.test_results):
            analysis_report = self.analyzer.analyze(self.test_results)
            analysis_report.display()
```

### Batch Processing

```python
# Analyze multiple test result files
test_files = ["results_1.json", "results_2.json", "results_3.json"]
all_reports = []

for test_file in test_files:
    report = analyzer.analyze(test_file)
    all_reports.append(report)
    print(f"Analyzed {test_file}: {len(report.recommendations)} recommendations")
```

## 📈 Performance

- **Analysis Time**: 2-10 minutes depending on codebase size
- **Token Usage**: 50K-500K tokens per analysis
- **Memory Usage**: ~100MB for typical projects
- **Supported Projects**: Up to 1000 Python files

## 🔒 Security

- **No Data Retention**: All analysis is local or in your Claude account
- **API Key Security**: Never sent to Merit servers
- **File Exclusions**: Sensitive files can be excluded from analysis
- **On-Premise Support**: Can be deployed locally

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/merit-analyzer/merit-analyzer.git
cd merit-analyzer
pip install -e ".[dev]"
pytest
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.merit-analyzer.com](https://docs.merit-analyzer.com)
- **Issues**: [GitHub Issues](https://github.com/merit-analyzer/merit-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/merit-analyzer/merit-analyzer/discussions)
- **Email**: support@merit-analyzer.com

## 🗺️ Roadmap

### Phase 1: MVP ✅
- [x] Core analysis engine
- [x] Pattern detection
- [x] Claude Agent SDK integration with Read, Grep, Glob tools
- [x] Agentic codebase navigation (intelligent file system access)
- [x] READ-ONLY analysis mode (no code modifications)
- [x] Basic recommendations

### Phase 2: Enhancement (Q2 2024)
- [ ] Web UI for report viewing
- [ ] More framework support
- [ ] Regression detection
- [ ] Git integration

### Phase 3: Advanced Features (Q3 2024)
- [ ] Automated fix generation
- [ ] Continuous monitoring
- [ ] CI/CD integration
- [ ] Team collaboration features

## 🙏 Acknowledgments

- Built on [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) - Uses Read, Grep, Glob tools for intelligent codebase navigation
- Operates in **READ-ONLY mode** - Analyzes code and provides recommendations without modifying files
- Inspired by the need for better AI system debugging tools
- Thanks to the open-source community for foundational libraries

---

**Transform test failures into specific code changes in minutes, not hours of manual debugging.**
