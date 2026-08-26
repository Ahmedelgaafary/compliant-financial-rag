# src/llm/client.py
class LLMClient:
    """Mock LLM client – replace with real API call."""

    def generate(self, prompt: str) -> str:
        # In production, call your LLM (e.g., OpenAI, Anthropic)
        if "verified claims" in prompt.lower():
            return (
                "Based on the verified data, the company's revenue in 2025 "
                "was $42.8 billion."
            )
        return "The company reported revenue of $42.8B in 2025."