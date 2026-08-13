import os

from datasets import Dataset, Features, Sequence, Value, load_dataset


def load_and_process_dataset(data_name: str):
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt = (
            "{question}\nPlease reason step by step, and put your final answer "
            "within \\boxed{{}}."
        )
        dataset = dataset.map(lambda x: {"turns": [prompt.format(**x)]})
    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt = (
            "{problem}\nPlease reason step by step, and put your final answer "
            "within \\boxed{{}}."
        )
        dataset = dataset.map(lambda x: {"turns": [prompt.format(**x)]})
    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt = (
            "{problem}\nPlease reason step by step, and put your final answer "
            "within \\boxed{{}}."
        )
        dataset = dataset.map(lambda x: {"turns": [prompt.format(**x)]})
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (
                    f"{x['instruction']}\n\nInput:\n{x['input']}"
                    if x["input"]
                    else x["instruction"]
                )
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})
    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt = (
            "Write a solution to the following problem and make sure that it "
            "passes the tests:\n```python\n{prompt}\n```"
        )
        dataset = dataset.map(lambda x: {"turns": [prompt.format(**x)]})
    elif data_name == "mbpp":
        dataset = load_dataset(
            "google-research-datasets/mbpp", "sanitized", split="test"
        )
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    elif data_name == "livecodebench":
        local_arrow = os.environ.get("DARTREE_LCB_ARROW", "")
        if local_arrow and os.path.isfile(local_arrow):
            return Dataset.from_file(local_arrow)
        base = (
            "https://huggingface.co/datasets/livecodebench/"
            "code_generation_lite/resolve/main/"
        )
        files = [
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ]
        dataset = load_dataset(
            "json", data_files={"test": [base + name for name in files]}
        )["test"]

        def format_lcb(row):
            system = (
                "You are an expert Python programmer. You will be given a "
                "question (problem specification) and will generate a correct "
                "Python program that matches the specification and passes all "
                "tests. You will NOT return anything except for the program"
            )
            question = f"### Question:\n{row['question_content']}"
            if row.get("starter_code"):
                message = "### Format: Use the following code structure:"
                code = f"```python\n{row['starter_code']}\n```"
            else:
                message = "### Format: Write your code in the following format:"
                code = "```python\n# YOUR CODE HERE\n```"
            return (
                f"{system}\n\n{question}\n\n{message}\n{code}\n\n"
                "### Answer: (use the provided format with backticks)"
            )

        features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=features,
        )
    else:
        raise ValueError(f"unsupported dataset: {data_name}")
    return dataset
