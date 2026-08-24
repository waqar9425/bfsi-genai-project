"""
Single factory for constructing the LLM client used across Argus.

Kept as one file so swapping providers (Gemini -> Anthropic/OpenAI, or a
cheaper/bigger model) later touches this file only, not every agent node.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Add it to a .env file at the project root."
        )
    return ChatGoogleGenerativeAI(
        # flash-lite: much higher free-tier daily quota than the top-tier
        # "gemini-flash-latest" alias, which hit a 20-requests/day wall
        # mid-Milestone-3 (preview/newest models get the tightest caps).
        # Fine for learning purposes -- structured output / tool-calling
        # quality difference vs. full flash is negligible for our graphs.
        model="gemini-flash-lite-latest",
        google_api_key=api_key,
        temperature=temperature,
    )


def get_token_usage(response) -> int:
    """Extract total token count from an LLM response, if available.
    usage_metadata isn't populated on the PARSED object returned by
    with_structured_output() unless you pass include_raw=True and read
    result["raw"] -- verified empirically (Milestone 9): by default that
    call site loses token usage entirely, silently. Degrades to 0 rather
    than crashing if usage_metadata is missing for any reason.
    """
    usage = getattr(response, "usage_metadata", None)
    return usage.get("total_tokens", 0) if usage else 0
