"""LLM client for generating responses."""

import os
from typing import Optional

# Load environment variables from .env file - MUST BE FIRST
from dotenv import load_dotenv

load_dotenv()

# Option 1: OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Option 2: Anthropic Claude
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

# Option 3: Google Gemini (google-genai SDK)
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Option 4: Local model via Ollama
try:
    import requests
except ImportError:
    requests = None


class LLMClient:
    """LLM client that respects the evidence and prevents hallucinations."""

    def __init__(
        self,
        provider: str = "openai",  # or "anthropic", "gemini", "ollama"
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: "openai", "anthropic", "gemini", or "ollama"
            model: Model name (uses default if None)
            temperature: 0.0 for deterministic, 0.7 for creative
            max_tokens: Maximum tokens in response (default from env or 4096)
        """
        self.provider = provider
        self.temperature = temperature
        
        # Read max_tokens from environment or use default
        if max_tokens is None:
            max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
        self.max_tokens = max_tokens

        # Set default models
        if provider == "openai":
            self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.client = self._init_openai()
        elif provider == "anthropic":
            self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
            self.client = self._init_anthropic()
        elif provider == "gemini":
            self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
            self.client = self._init_gemini()
        elif provider == "ollama":
            self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
            self.client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_openai(self):
        """Initialize OpenAI client."""
        if OpenAI is None:
            raise ImportError("OpenAI not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        return OpenAI(api_key=api_key)

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        if Anthropic is None:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        return Anthropic(api_key=api_key)

    def _init_gemini(self):
        """Initialize Google Gemini client using the google-genai SDK."""
        if genai is None:
            raise ImportError("Google GenAI not installed. Run: pip install google-genai")

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        return genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The generated response as a string
        """
        if "EVIDENCE" not in prompt and "evidence" not in prompt:
            print("Warning: Prompt does not contain evidence. This may lead to hallucinations.")

        try:
            if self.provider == "openai":
                return self._generate_openai(prompt)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt)
            elif self.provider == "gemini":
                return self._generate_gemini(prompt)
            elif self.provider == "ollama":
                return self._generate_ollama(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"Error generating response: {e}")
            return "The system encountered an error. Please try again later."

    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial analyst. "
                        "Always use ONLY the evidence provided. "
                        "Do not invent information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,  # Now configurable
        )
        return response.choices[0].message.content.strip()

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate using Anthropic Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,  # Now configurable
            temperature=self.temperature,
            system=(
                "You are a financial analyst. "
                "Always use ONLY the evidence provided. "
                "Do not invent information."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _generate_gemini(self, prompt: str) -> str:
        """Generate using Google Gemini."""
        # Use model from environment or default
        model_names = [
            self.model,
            os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp"),
            "gemini-2.0-flash-exp",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro",
        ]

        # Remove duplicates while preserving insertion order
        seen = set()
        model_names = [m for m in model_names if not (m in seen or seen.add(m))]

        # Configure model parameters with higher token limit
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,  # Now configurable
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )

        last_error = None
        for model_name in model_names:
            try:
                print(f"Trying model: {model_name}")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text.strip()
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                last_error = e
                continue

        raise last_error or RuntimeError("All Gemini models failed")

    def _generate_ollama(self, prompt: str) -> str:
        """Generate using Ollama (local model)."""
        if requests is None:
            raise ImportError("Requests not installed. Run: pip install requests")

        system_prompt = (
            "You are a financial analyst. "
            "Always use ONLY the evidence provided. "
            "Do not invent information."
        )

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,  # Now configurable
            },
            timeout=120,  # Increased timeout for longer responses
        )
        response.raise_for_status()
        return response.json()["response"].strip()


def get_llm_client() -> LLMClient:
    """
    Factory function to get the LLM client based on environment.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Get max_tokens from environment
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

    if provider == "gemini":
        if os.environ.get("GEMINI_API_KEY"):
            print("Using Gemini as LLM provider (from LLM_PROVIDER)")
            return LLMClient(
                provider="gemini",
                max_tokens=max_tokens
            )
        raise ValueError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set")

    if provider == "openai":
        if os.environ.get("OPENAI_API_KEY"):
            print("Using OpenAI as LLM provider (from LLM_PROVIDER)")
            return LLMClient(
                provider="openai",
                max_tokens=max_tokens
            )
        raise ValueError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set")

    if provider == "anthropic":
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("Using Anthropic as LLM provider (from LLM_PROVIDER)")
            return LLMClient(
                provider="anthropic",
                max_tokens=max_tokens
            )
        raise ValueError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set")

    if provider == "ollama":
        print("Using Ollama as LLM provider (from LLM_PROVIDER)")
        return LLMClient(
            provider="ollama",
            max_tokens=max_tokens
        )

    if os.environ.get("GEMINI_API_KEY"):
        print("Using Gemini as LLM provider")
        return LLMClient(
            provider="gemini",
            max_tokens=max_tokens
        )
    if os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI as LLM provider")
        return LLMClient(
            provider="openai",
            max_tokens=max_tokens
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Using Anthropic as LLM provider")
        return LLMClient(
            provider="anthropic",
            max_tokens=max_tokens
        )

    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Using Ollama as LLM provider")
            return LLMClient(
                provider="ollama",
                max_tokens=max_tokens
            )
    except (requests.ConnectionError, requests.Timeout, ImportError):
        pass

    raise RuntimeError(
        "No LLM provider configured. "
        "Please set LLM_PROVIDER=gemini and GEMINI_API_KEY, "
        "or set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally."
    )


def list_gemini_models():
    """List all available Gemini models."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not set. Please set it in your .env file.")
            return

        client = genai.Client(api_key=api_key)

        print("\n" + "=" * 80)
        print("AVAILABLE GEMINI MODELS")
        print("=" * 80)

        models = client.models.list()
        model_list = list(models)

        for model in model_list:
            supported_methods = getattr(model, "supported_generation_methods", [])
            print(f"\nModel: {model.name}")
            print(f"  Display Name: {getattr(model, 'display_name', 'N/A')}")
            print(f"  Supported Methods: {supported_methods}")

    except Exception as e:
        print(f"Error listing models: {e}")


if __name__ == "__main__":
    list_gemini_models()

    print("\n" + "=" * 80)
    print("TESTING LLM CLIENT")
    print("=" * 80)

    try:
        client = get_llm_client()
        print(f"Provider: {client.provider}")
        print(f"Model: {client.model}")
        print(f"Max Tokens: {client.max_tokens}")
        response = client.generate("EVIDENCE: 2+2=4. What is 2+2?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")