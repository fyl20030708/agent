"""CLI entry point for the research agent."""

from research_agent.agent import ask


def main() -> None:
    import sys

    question = " ".join(sys.argv[1:])
    if not question:
        print("Usage: python -m research_agent.cli <question>")
        raise SystemExit(1)
    print(ask(question))


if __name__ == "__main__":
    main()
