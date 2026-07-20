# Week 1 Acceptance — ReAct

**Date:** YYYY-MM-DD

## Commands run

```bash
pytest tests/test_react_parse.py -v
python week02/react_loop.py "Training 目录下有哪些训练相关脚本？用工具确认后再回答。"
AGENT_VERBOSE=1 python -m research_agent.cli "Training 目录下有哪些训练相关脚本？先用工具确认再回答。"
```

## Checklist

- [ ] `parse_action` unit tests pass
- [ ] week02 printed `[step N] ...` and a grounded answer
- [ ] research_agent verbose showed `budget consume` + tools
- [ ] I can explain ReAct vs tool_calls using `week02/COMPARISON.md`

## My 3-sentence summary

(write after you finish)
