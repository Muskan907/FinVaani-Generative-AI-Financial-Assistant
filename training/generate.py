"""
generate.py — Shared generation utility used by all model scripts.
Handles prompt formatting and answer extraction for both English and Hindi.
"""

import re
import torch


def build_prompt(question: str, language: str, few_shot_examples: list = None) -> str:
    """
    Build the generation prompt for a given question.

    Args:
        question:           The question text.
        language:           "en" or "hi".
        few_shot_examples:  Optional list of (q, a) tuples to prepend.

    Returns:
        Formatted prompt string.
    """
    prompt = ""

    if few_shot_examples:
        for q, a in few_shot_examples:
            if language == "hi":
                prompt += f"### सवाल: {q}\n### जवाब: {a}\n\n"
            else:
                prompt += f"### Question: {q}\n### Answer: {a}\n\n"

    if language == "hi":
        prompt += f"### सवाल: {question}\n### जवाब:"
    else:
        prompt += f"### Question: {question}\n### Answer:"

    return prompt


def extract_answer(generated_text: str, language: str) -> str:
    """
    Extract only the answer portion from generated text.
    Strips the prompt prefix and any trailing generation artifacts.

    Args:
        generated_text: Full decoded model output.
        language:       "en" or "hi".

    Returns:
        Cleaned answer string.
    """
    # Split on the answer marker
    if language == "hi":
        marker = "### जवाब:"
    else:
        marker = "### Answer:"

    if marker in generated_text:
        answer = generated_text.split(marker)[-1]
    else:
        answer = generated_text

    # Stop at end-of-text token or next question marker
    for stop in ["<|endoftext|>", "### Question:", "### सवाल:", "\n\n\n"]:
        if stop in answer:
            answer = answer.split(stop)[0]

    return answer.strip()


def generate_answer(
    model,
    tokenizer,
    question: str,
    language: str = "en",
    few_shot_examples: list = None,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    device: str = "cpu",
) -> str:
    """
    Generate an answer for a given question using the provided model.

    Args:
        model:              HuggingFace causal LM model.
        tokenizer:          Corresponding tokenizer.
        question:           Input question.
        language:           "en" or "hi".
        few_shot_examples:  Optional list of (q, a) tuples for few-shot prompting.
        max_new_tokens:     Maximum tokens to generate.
        temperature:        Sampling temperature.
        device:             "cuda", "mps", or "cpu".

    Returns:
        Generated answer string.
    """
    prompt = build_prompt(question, language, few_shot_examples)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return extract_answer(generated, language)


def get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
