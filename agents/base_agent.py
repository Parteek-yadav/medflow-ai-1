class BaseAgent:
    """Minimal base agent abstraction for future agent implementations."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, task: str) -> str:
        return f"{self.name} received task: {task}"
