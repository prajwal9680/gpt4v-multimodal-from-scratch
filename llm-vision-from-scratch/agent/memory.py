class AgentMemory:
    """Stores the agent's reasoning history across steps."""
    def __init__(self):
        self.steps = []

    def add(self, step: str):
        self.steps.append(step)

    def context(self) -> str:
        return '\n'.join(self.steps)

    def clear(self):
        self.steps = []
