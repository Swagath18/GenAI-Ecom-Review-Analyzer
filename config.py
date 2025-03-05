from dataclasses import dataclass

@dataclass
class ReviewAnalysisConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 150
    min_length: int = 30
    do_sample: bool = False
    temperature: float = 0.7
    top_k: int = None  # Correct type hint
    top_p: float = None  # Correct type hint
    batch_size: int = 8  # ✅ Added batch processing config

    def __post_init__(self):
        # If both top_k and top_p are set, it might lead to conflicting behaviors, so we can give a warning if both are set.
        if self.top_k and self.top_p:
            print("Warning: Both top_k and top_p are set. Consider choosing one for controlling randomness.")