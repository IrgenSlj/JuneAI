# june-brain

June's intelligence layer. A Python package that wraps a LangGraph agent, a three-store memory system, an MCP skills loader, and the Gemma/Gemini model providers.

## Install

```bash
pip install june-brain
```

## Use

```python
from june_brain import create_june_agent
from june_brain.memory import MemoryManager

memory = MemoryManager()
agent = create_june_agent(memory=memory)
for event in agent.stream(user_id="me", message="Hello"):
    print(event)
```

## See Also

- [Architecture overview](../../docs/architecture/overview.md)
- [ADR 0004: memory architecture](../../docs/decisions/0004-memory-architecture.md)
- [ADR 0005: skills as MCP](../../docs/decisions/0005-skills-as-mcp.md)
