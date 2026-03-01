"""
Answer relevance scorer — uses an LLM to judge how relevant the answer is.

Non-deterministic, slower, costs tokens. Uses a cheap model by default.
"""

from __future__ import annotations

import litellm

from petal.eval.scorer import scorer


@scorer(name="answer_relevance", judge_model="gpt-4o-mini")
async def answer_relevance(input: str, output: str, judge_model: str = "gpt-4o-mini") -> float:
    """Use an LLM to rate how relevant the agent's answer is to the question.

    Sends the input/output pair to a cheap judge model and asks it to
    rate relevance from 0.0 to 1.0. Parses the float from the response.

    Returns 0.0 if parsing fails.
    """
    prompt = (
        "Rate how relevant this answer is to the question.\n"
        f"Question: {input}\n"
        f"Answer: {output}\n"
        "Respond with ONLY a number from 0.0 to 1.0, nothing else."
    )

    try:
        response = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = response.choices[0].message.content.strip()
        return max(0.0, min(1.0, float(score_text)))
    except Exception:
        return 0.0
