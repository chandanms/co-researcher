"""
LLM Client for Azure Anthropic Foundry.
"""

import json
import os

from anthropic import AnthropicFoundry
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Client for Azure Anthropic Foundry."""

    def __init__(self):
        endpoint = os.getenv("AZURE_ANTHROPIC_ENDPOINT")
        api_key = os.getenv("AZURE_ANTHROPIC_API_KEY")
        self.model = os.getenv("AZURE_ANTHROPIC_MODEL", "claude-opus-4-5")

        if not endpoint:
            raise ValueError("AZURE_ANTHROPIC_ENDPOINT not found in .env")
        if not api_key:
            raise ValueError("AZURE_ANTHROPIC_API_KEY not found in .env")

        self.client = AnthropicFoundry(api_key=api_key, base_url=endpoint)
        self.max_tokens = 4096

    def call(self, prompt: str) -> dict:
        """
        Call the LLM and parse JSON response.

        Args:
            prompt: The prompt to send

        Returns:
            Parsed JSON dict, or error dict on failure
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            text = text.strip()

            # Handle potential markdown code blocks
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)

            return json.loads(text)

        except json.JSONDecodeError as e:
            return {"error": "parse_failed", "raw": text, "parse_error": str(e)}
        except Exception as e:
            return {"error": "api_failed", "message": str(e)}
