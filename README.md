# Nano GraphRAG v2 - A lightweight implementation of GraphRAG

This project demonstrates a multi-layered agent architecture using LangGraph and Deep Agents framework, showcasing human-in-the-loop functionality for tool execution approval.

## Architecture

### Three-Layer Agent System

```
Top Supervisor (总经理)
├── Team1 Agent (乘法和除法团队)
│   ├── Multiplication Agent (乘法代理)
│   │   └── multiply_numbers 工具
│   └── Division Agent (除法代理)
│       └── divide_numbers 工具
└── Team2 Agent (加法和减法团队)
    ├── Addition Agent (加法代理)
    │   └── add_numbers 工具
    └── Subtraction Agent (减法代理)
        └── subtract_numbers 工具
```

## Features

- **Multi-layered Agent Architecture**: Hierarchical delegation of tasks
- **Human-in-the-loop**: All tool executions require human approval
- **State Persistence**: Uses LangGraph's checkpointer for reliable state management
- **Interrupt Handling**: Supports complex multi-agent interrupt scenarios
- **DashScope Integration**: Uses Alibaba's qwen-plus model

## Technology Stack

- **LangGraph 1.0**: Graph-based agent orchestration
- **Deep Agents**: Advanced agent composition framework
- **DashScope**: Alibaba's LLM service (qwen-plus model)
- **Python 3.12**: Modern Python features

## Setup

1. Install dependencies:
```bash
pip install langgraph deepagents langchain-openai python-dotenv
```

2. Configure environment variables:
```bash
# Create .env file
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## Usage

### Run Individual Agents

```bash
# Test basic math agents
python agents/deep_agent_add.py
python agents/deep_agent_multiplication.py
python agents/deep_agent_division.py
python agents/deep_agent_subtraction.py

# Test team agents
python agents/team1_agent.py
python agents/team2_agent.py

# Test top supervisor
python agents/top_supervisor.py
```

### Example Interaction

```python
from agents.top_supervisor import create_top_supervisor
import uuid

# Create the top supervisor
agent, checkpointer = create_top_supervisor()

# Create config with unique thread_id
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# Send a complex multi-step request
result = agent.invoke({
    "messages": [{"role": "user", "content": "Calculate 10 × 20, then 100 - 13, and finally 30 ÷ 2"}]
}, config=config)

# Handle human review if interrupted
if result.get("__interrupt__"):
    # Process interrupts and collect user decisions
    # ... (see code for full implementation)
```

## Key Concepts

### Human-in-the-loop Workflow

1. **Tool Call Triggered**: Agent decides to call a tool
2. **Interrupt Generated**: Execution pauses, state saved to checkpointer
3. **Human Review**: User sees tool call details and makes decision (approve/reject/edit)
4. **Execution Resumed**: Agent continues based on user decision

### State Persistence

- **Checkpointer**: Uses MemorySaver for development, PostgresSaver for production
- **Thread ID**: Unique identifier for conversation state
- **Config Consistency**: Same config must be used for initial call and resume

### Multi-Agent Interrupts

When multiple agents trigger interrupts simultaneously:

```python
# Multiple interrupt objects with unique IDs
resume_map = {
    'interrupt-id-1': {'decisions': [{'type': 'approve'}]},
    'interrupt-id-2': {'decisions': [{'type': 'reject'}]},
    'interrupt-id-3': {'decisions': [{'type': 'approve'}]}
}

result = agent.invoke(Command(resume=resume_map), config=config)
```

## Security Features

### Strict Agent Behavior

All agents are configured with strict system prompts to prevent bypassing human review:

```python
system_prompt="""
**严格规则**：
1. 禁止自己进行任何计算，无论多简单，必须调用工具/子代理处理
2. 如果工具调用被人工审核拒绝（rejected），不要重试该操作
3. 被拒绝的操作不要给出计算结果，直接报告该操作被拒绝
4. 不要尝试用其他方式计算或编造结果
"""
```

This prevents agents from bypassing the human review process by calculating results themselves.

## Project Structure

```
nano-graphragv2/
├── agents/
│   ├── deep_agent_add.py           # Addition agent
│   ├── deep_agent_subtraction.py   # Subtraction agent  
│   ├── deep_agent_multiplication.py # Multiplication agent
│   ├── deep_agent_division.py      # Division agent
│   ├── team1_agent.py              # Team1 (multiplication & division)
│   ├── team2_agent.py              # Team2 (addition & subtraction)
│   └── top_supervisor.py           # Top-level supervisor
├── .env                            # Environment variables
└── README.md                       # This file
```

## Development Notes

### Bug Fixes and Lessons Learned

1. **Multi-interrupt Resume Format**: Use `{interrupt_id: {'decisions': [...]}}` for multiple interrupts
2. **Config Consistency**: Always use the same config object for initial call and resume
3. **Agent Bypass Prevention**: Strict system prompts prevent LLMs from calculating results when tools are rejected
4. **Environment Variables**: Must call `load_dotenv()` to load `.env` file

### Testing Scenarios

- Single tool calls with human review
- Multi-step tasks requiring multiple agents
- Reject scenarios to test security measures
- Complex nested agent interactions

## License

MIT License - see LICENSE file for details.