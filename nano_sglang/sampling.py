from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1
