from __future__ import annotations

import os
from functools import lru_cache


class RerankerError(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 8, max_length: int = 512) -> None:
        if os.name == "nt":
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RerankerError(
                "Reranker dependencies are missing. Install them with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as exc:
            raise RerankerError(f"Could not load reranker model {model_name}: {exc}") from exc

        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            pairs = [[query, passage] for passage in batch]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
        return scores


@lru_cache(maxsize=4)
def get_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)
