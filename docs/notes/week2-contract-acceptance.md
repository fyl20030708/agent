# Week 2 Acceptance — Prompt contracts

**Date:** YYYY-MM-DD

## Commands run

```bash
pytest tests/test_structured_parse.py tests/test_answer_contract.py -v
python week03/structured_output_demo.py
AGENT_VERBOSE=1 python -m research_agent.cli "这个仓库的训练入口大概在哪个文件？先用工具确认再回答。"
```

## Checklist

- [ ] structured parse tests pass
- [ ] week03 returns valid JSON (possibly after one retry)
- [ ] research_agent answer cites a real relative path
- [ ] I can explain system vs user vs repair vs tool schema using `week03/CONTRACT.md`

## My 3-sentence summary

(write after you finish)
