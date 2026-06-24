from agent.memory import AgentMemory
from agent.tools import ToolRegistry


def parse_action(output: str):
    """
    Parse model output to extract the intended action and its input.

    Expected format in model output:
        ACTION: <tool_name>
        ACTION_INPUT: <tool_input_string>
    """
    action, tool_input = "", ""
    for line in output.split('\n'):
        if "ACTION:" in line:
            action = line.split("ACTION:")[1].strip()
        elif "ACTION_INPUT:" in line:
            tool_input = line.split("ACTION_INPUT:")[1].strip()
    return action, tool_input


class Agent:
    """
    A ReAct-style agent loop.

    At each step the agent:
        1. Builds a prompt from memory + user query
        2. Calls the model to generate a response
        3. If "FINAL ANSWER" is in the output → return it
        4. Otherwise parse ACTION + ACTION_INPUT → run tool → record observation
        5. Repeat

    Args:
        model:         A language model with a .generate(prompt) method.
        tool_registry: A ToolRegistry with registered tools.
    """
    def __init__(self, model, tool_registry: ToolRegistry):
        self.model = model
        self.tools = tool_registry
        self.memory = AgentMemory()

    def run(self, query: str) -> str:
        while True:
            prompt = self.memory.context() + f"\nUser: {query}"
            output = self.model.generate(prompt)

            print("MODEL OUTPUT:\n", output)

            if "FINAL ANSWER" in output:
                return output

            action, tool_input = parse_action(output)
            tool = self.tools.get(action)

            if tool is None:
                return f"Error: Tool '{action}' not found. Available: {self.tools.list_tools()}"

            observation = tool.run(tool_input)
            self.memory.add(output)
            self.memory.add(f"Observation: {observation}")
