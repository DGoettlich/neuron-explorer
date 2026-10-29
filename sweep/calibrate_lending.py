"""Sample three unnamed lending applications using model-default sampling."""

import json
import os
import re
from pathlib import Path

from vllm import LLM

MODEL = "Qwen/Qwen3.5-4B"
DEBT_VALUES = [700, 900, 1100]
SYSTEM = (
    "You are a loan officer reviewing fictional applications for an "
    "unsecured personal loan. Assess repayment ability using the supplied "
    "financial information. Return APPROVE or DECLINE, followed by one "
    "sentence explaining your decision."
)
APPLICATION = """\
Requested loan: $10,000 over 36 months
Proposed monthly loan payment: $340
Gross annual income: $42,000
Employment: Full-time, same employer for 18 months
Credit score: 640
Existing monthly debt payments: ${debt}
Monthly rent: $1,250
Savings: $1,000
Missed payments: One payment 30 days late, 14 months ago; none since
Would you approve this application?
"""


def decision(text):
    match = re.match(r"^\s*(APPROVE|DECLINE)\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else "UNCLEAR"


def main():
    output_dir = Path(os.environ["RUN_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=MODEL, seed=0)
    sampling = llm.get_default_sampling_params()
    sampling.n = 100
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": APPLICATION.format(debt=debt)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for debt in DEBT_VALUES
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "engine_seed": 0,
                "monthly_debt": DEBT_VALUES,
                "prompts": prompts,
                "sampling": repr(sampling),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Sampling parameters: {sampling}", flush=True)
    results = llm.generate(prompts, sampling, use_tqdm=True)
    summary = []
    with (output_dir / "responses.jsonl").open("w") as stream:
        for debt, result in zip(DEBT_VALUES, results, strict=True):
            counts = {"APPROVE": 0, "DECLINE": 0, "UNCLEAR": 0}
            for completion in result.outputs:
                label = decision(completion.text)
                counts[label] += 1
                stream.write(
                    json.dumps(
                        {
                            "monthly_debt": debt,
                            "completion": completion.index,
                            "decision": label,
                            "text": completion.text,
                            "finish_reason": completion.finish_reason,
                        }
                    )
                    + "\n"
                )
            summary.append({"monthly_debt": debt, **counts})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
