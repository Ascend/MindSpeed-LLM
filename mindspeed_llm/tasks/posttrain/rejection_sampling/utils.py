import os

from datasets import interleave_datasets, load_dataset, load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm


def blending_datasets(
        datasets,
        probabilities,
        strategy=None,
        seed=42,
        max_count=5000000,
        stopping_strategy="first_exhausted",
        train_split="train"
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    if len(probabilities) != len(datasets):
        raise ValueError(f"Length of probabilities ({len(probabilities)}) must match the length of datasets ({len(datasets)})")

    train_data_list = []
    for _, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    return train_dataset


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key] if input_key in data.keys() else None
        if input_template:
            prompt = input_template.format(prompt)
    return prompt


class PromptGtAnswerDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer,
            strategy,
            input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template

        input_key = self.strategy.args.map_keys.get("prompt", None)
        gt_answer_key = self.strategy.args.map_keys.get("gt_answer", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.gt_answers = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            gt_answer = preprocess_data(data, input_key=gt_answer_key)
            self.prompts.append(prompt)
            self.gt_answers.append(gt_answer)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "gt_answer": self.gt_answers[idx]}


def apply_GenRM_template(prompt_text, response_text, ground_truth_answer_text):
    if ground_truth_answer_text:
        full_input = f"""你是一个判别推理正确性的专家。
                    [PROMPT]：{prompt_text}
                    [RESPONSE]：{response_text}
                    [REFERENCE]：{ground_truth_answer_text}
                    任务目标：根据给定的问题[PROMPT]和参考答案[REFERENCE]，评估回复质量[RESPONSE]，考虑语言一致性、\
                    格式正确性、最终结果正确性、推理合理性、回复无害性、语句重复冗余性，用简洁的文字说明原因，不需要给出改进答案。\
                    最后给出0到1之间的分数，分数以<score></score>的形式给出。
                    """
    else:
        full_input = f"""你是一个判别推理正确性的专家。
                    [PROMPT]：{prompt_text}
                    [RESPONSE]：{response_text}
                    任务目标：根据给定的问题[PROMPT]，评估回复质量[RESPONSE]，考虑语言一致性、格式正确性、\
                    最终结果正确性、推理合理性、回复无害性、语句重复冗余性，用简洁的文字说明原因，不需要给出改进答案。\
                    最后给出0到1之间的分数，分数以<score></score>的形式给出。
                    """
    return full_input


def rejection_sampling_processor(objs):
    out = {}
    for obj in tqdm(objs, desc="Rejection Sampling process...."):
        prompt = obj["prompt"]
        output = obj["output"]
        reward = float(obj["reward"])

        if reward > 0:
            if prompt not in out:
                out[prompt] = {"output": output, "reward": reward}
            elif reward > out[prompt]["reward"]:
                out[prompt]["reward"] = reward
                out[prompt]["output"] = output

    return [{"prompt": k, "output": v["output"], "reward": v["reward"]} for k, v in out.items()]
