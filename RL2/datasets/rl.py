import copy
from pathlib import Path
from RL2.datasets.base import BaseDataset


class RLDataset(BaseDataset):

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.data_path = str(getattr(config, 'path', ''))

        if 'planetarium' in self.data_path.lower():
            self.domain_prompt = "The domain for the planning problem is:"
            self.problem_prompt = "Provide me with the complete, valid problem PDDL file that describes the following planning problem in ```pddl markdown blocks:\n\n"
            self.domains_dict = self._load_domains()
        else:
            self.domain_prompt = None
            self.problem_prompt = None
            self.domains_dict = None

    def _load_domains(self) -> dict[str, str]:
        """Load PDDL domain files."""
        domains = {}
        base_path = Path.home() / "RL2" / "eval" / "pddl"
        for domain_name in ["blocksworld", "gripper", "floor-tile"]:
            domain_path = base_path / f"{domain_name}.pddl"
            with open(domain_path) as f:
                domains[domain_name] = f.read()
        return domains

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        data = {}

        if 'planetarium' in self.data_path.lower():
            data["prompt"] = self.tokenizer.apply_chat_template(
                self.apply_template(
                    ex["natural_language"],
                    self.domain_prompt,
                    self.domains_dict[ex["domain"]],
                    self.problem_prompt,
                    ex["problem_pddl"],
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            extra_info = ex.get("extra_info", {})
            extra_info["idx"] = ex["id"]
            extra_info["domain"] = ex["domain"]
            try:
                extra_info["domain_pddl"] = self.domains_dict[ex["domain"]]
            except KeyError:
                raise ValueError(f"Domain {ex['domain']} not found in domains_dict")
            extra_info["is_placeholder"] = ex["is_placeholder"]
            data["extra_info"] = extra_info

        elif 'multipl' in self.data_path.lower():
            if "prompt" in ex.keys():
                data["prompt"] = ex["prompt"]
            elif "messages" in ex.keys():
                data["prompt"] = self.tokenizer.apply_chat_template(
                    ex["messages"],
                    add_generation_prompt=True,
                    tokenize=False
                )
            extra_info = ex.get("extra_info", {})
            extra_info["idx"] = idx
            data["extra_info"] = extra_info

        else:
            raise ValueError(f"Unsupported dataset: {self.data_path}")

        return data

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]

    def apply_template(
        self,
        natural_language: str = "",
        domain_prompt: str = "",
        domain: str = "",
        problem_prompt: str = "",
        problem_pddl: str = "",
        include_answer: bool = False
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": f"{problem_prompt} {natural_language} "
                + f"{domain_prompt}\n{domain}\n",
            },
        ] + (
            [
                {
                    "role": "assistant",
                    "content": " " + problem_pddl,
                },
            ]
            if include_answer
            else []
        )