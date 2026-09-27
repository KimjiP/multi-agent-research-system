"""
Project A2 - Version 1 (L1 Baseline): Single API Call Research Agent
=====================================================================
Purpose: Establish a baseline showing what a single LLM call can and cannot do.
This output becomes the comparison point for the L2-L3 pipeline.
"""

import anthropic
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL = "claude-sonnet-4-20250514"

# EU AI Act vertical benchmark questions (test all 3, compare outputs)
BENCHMARK_QUESTIONS = [
    # Question 1: Specific compliance scenario (tests practical reasoning)
    "What are the current compliance requirements for a company deploying a high-risk AI hiring tool under the EU AI Act, and what enforcement actions have been taken so far?",

    # Question 2: Cross-jurisdictional comparison (tests breadth + synthesis)
    "How do the EU AI Act's requirements for foundation model providers compare to the US Executive Order on AI's approach to frontier model oversight?",

    # Question 3: Narrow technical question (tests depth + specificity)
    "What conformity assessment procedures are required for high-risk AI systems under the EU AI Act, and which notified bodies have been designated so far?",
]

SYSTEM_PROMPT = """You are a regulatory research analyst specializing in AI governance and compliance.

Given a research question, produce a structured research report with:
1. An executive summary (2-3 sentences)
2. Key findings organized by theme
3. Specific citations for every factual claim (include source name, author if known, publication date)
4. A confidence assessment noting what you're certain about vs. uncertain about
5. Identified gaps - what information would require further research

Be precise about what you know vs. what you're inferring. If you cannot verify a specific fact, say so explicitly rather than guessing."""

def run_v1_baseline(question: str, question_number: int) -> dict:
    """Run a single API call and capture the output + metadata."""
    client = anthropic.Anthropic()

    print(f"\n{'='*60}")
    print(f"QUESTION {question_number}: {question[:80]}...")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Research question: {question}"}
        ]
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    output_text = response.content[0].text

    result = {
        "question": question,
        "response": output_text,
        "model": MODEL,
        "duration_seconds": duration,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        "timestamp": start_time.isoformat(),
    }

    print(output_text)
    print(f"\n--- Stats ---")
    print(f"Duration: {duration:.1f}s")
    print(f"Tokens: {result['total_tokens']} (in: {result['input_tokens']}, out: {result['output_tokens']})")

    return result

def evaluate_output(result: dict) -> dict:
    """Manual evaluation checklist - fill this in after reviewing each output."""
    evaluation = {
        "question": result["question"][:80],
        # Score each 1-5 after reading the output
        "checklist": {
            "citations_verifiable": "TODO: Are the cited sources real and verifiable?",
            "claims_accurate": "TODO: Are factual claims correct (spot-check 3-5)?",
            "covers_key_aspects": "TODO: Does it address all parts of the question?",
            "acknowledges_uncertainty": "TODO: Does it flag what it doesn't know?",
            "hallucination_count": "TODO: How many fabricated facts/sources did you find?",
            "actionable": "TODO: Could a compliance officer use this as a starting point?",
        },
        "strengths": "TODO: What did the single call do well?",
        "weaknesses": "TODO: Where did it fall short? (These become your V2 requirements)",
    }
    return evaluation

if __name__ == "__main__":
    all_results = []

    # Run all 3 benchmark questions
    for i, question in enumerate(BENCHMARK_QUESTIONS, 1):
        result = run_v1_baseline(question, i)
        all_results.append(result)

    # Save raw outputs for later comparison with V2 and V3
    output_file = f"v1_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")

    # Print evaluation template
    print(f"\n\n{'='*60}")
    print("EVALUATION TEMPLATE")
    print("Review each output and fill in the checklist below.")
    print("Your notes here become the 'why V2 is needed' section of your README.")
    print(f"{'='*60}\n")

    for i, result in enumerate(all_results, 1):
        eval_template = evaluate_output(result)
        print(f"\nQuestion {i}: {eval_template['question']}...")
        for key, value in eval_template["checklist"].items():
            print(f"  [ ] {key}: {value}")
        print(f"  Strengths: {eval_template['strengths']}")
        print(f"  Weaknesses: {eval_template['weaknesses']}")