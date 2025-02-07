import time
from typing import Literal

import google.generativeai as genai
import instructor
import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm


PCL_CATEGORIES = Literal[
    "Unbalanced power relations",
    "Shallow solution",
    "Presupposition",
    "Authority voice",
    "Metaphor",
    "Compassion",
    "The poorer, the merrier",
]


class PCLClassification(BaseModel):
    reasoning: str = Field(
        description="Detailed explanation of why the text is or isn't classified as PCL."
    )
    is_pcl: bool = Field(description="True if the text contains PCL, False otherwise.")
    categories: list[PCL_CATEGORIES] | None = Field(
        default=None,
        description="List of PCL categories the text falls into, if applicable.",
    )


SYSTEM_PROMPT = """
You are an expert language model trained to detect patronizing and condescending language (PCL).
Your task is to analyze the given text and determine whether it contains PCL.
If PCL is detected, classify it into one or more of the following categories:

1. Unbalanced power relations: The author distances themselves from the community or claims the power to give rights they do not own.
2. Shallow solution: A simplistic charitable action is presented as life-changing or a solution to a deep-rooted issue.
3. Presupposition: The author makes assumptions, generalizes experiences, or uses stereotypes without valid sources.
4. Authority voice: The author acts as a spokesperson or advisor for a vulnerable community.
5. Metaphor: The text uses euphemisms or comparisons to soften or obscure the true meaning of a situation.
6. Compassion: The author elicits pity through exaggerated, poetic, or flowery descriptions of vulnerability.
7. The poorer, the merrier: The text romanticizes poverty, suggesting that vulnerable communities are happier or morally superior due to their struggles.

For each input text:
- First, provide a detailed reasoning for your classification.
- Indicate whether the text contains PCL (True/False).
- If PCL is present, list all applicable categories.

Respond in the structured format of the PCLClassificationResponse model.
"""


class stop_after_attempt_without_ratelimit_error(stop_after_attempt):
    def __init__(self, rate_limit_error: Exception, max_attempt_number: int) -> None:
        super().__init__(max_attempt_number)
        self.rate_limit_error = rate_limit_error

    def __call__(self, retry_state: RetryCallState) -> bool:
        if retry_state.outcome.failed and isinstance(
            retry_state.outcome.exception(), self.rate_limit_error
        ):
            self.max_attempt_number += 1
        return super().__call__(retry_state=retry_state)


class wait_for_rate_limit(wait_random_exponential):
    def __init__(self, rate_limit_error: Exception, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rate_limit_error = rate_limit_error

    def __call__(self, retry_state: RetryCallState) -> float:
        if retry_state.outcome.failed and isinstance(
            retry_state.outcome.exception(), self.rate_limit_error
        ):
            return super().__call__(retry_state=retry_state)
        return 0.0


client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-2.0-flash",
    ),
    mode=instructor.Mode.GEMINI_JSON,
)

df = pd.read_csv("/Users/rino/Downloads/train.csv")


def get_pcl_classification(community: str, text: str) -> PCLClassification | None:
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Community: {community}\nText: {text}"},
            ],
            response_model=PCLClassification,
            max_retries=Retrying(
                stop=stop_after_attempt_without_ratelimit_error(
                    rate_limit_error=ResourceExhausted, max_attempt_number=10
                ),
                wait=wait_for_rate_limit(
                    rate_limit_error=ResourceExhausted, multiplier=1, max=60
                ),
                retry=retry_if_exception_type((ValidationError, ResourceExhausted)),
            ),
        )
        return response
    except Exception as e:
        print(e)
        return None


start, end = 5400, 6800
input_df = df[start:end]

results_file_path = "results.jsonl"
results = []
total_failures = 0

# Open the results file once in append mode.
with open(results_file_path, "a") as out_file:
    for index, row in tqdm(input_df.iterrows(), total=len(input_df)):
        time.sleep(2)
        community = row["community"]
        text = row["text"]
        pcl_classification = get_pcl_classification(community, text)
        if pcl_classification is not None:
            # Append the classification in JSONL format, ensuring one JSON per line.
            out_file.write(f"{index}\t{pcl_classification.model_dump_json()}\n")
            results.append({"index": index, **pcl_classification.model_dump()})
        else:
            print(f"Failed at {index}")
            total_failures += 1

print(f"Total failures: {total_failures}")
results_df = pd.DataFrame(results)
results_df.to_csv(f"results-{start}-{end}.csv", index=False)
