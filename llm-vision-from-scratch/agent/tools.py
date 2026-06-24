class Tool:
    """
    A named callable tool the agent can dispatch.

    Args:
        name: Tool identifier used in ACTION: lines.
        func: Callable that takes a string input and returns a string output.
    """
    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def run(self, input: str) -> str:
        return self.func(input)


class ToolRegistry:
    """
    Registry for managing available tools.
    The agent looks up tools by name before calling them.
    """
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())
