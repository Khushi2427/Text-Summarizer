from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


class PredictionPipeline:
    def __init__(self):
        # Load config
        self.config = ConfigurationManager().get_model_evaluation_config()

        # Force CPU (important for Render)
        self.device = 0 if torch.cuda.is_available() else -1

        # Load tokenizer & model ONCE
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)

        # Create pipeline ONCE
        self.pipe = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

    def predict(self, text: str) -> str:
        gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128
        }

        output = self.pipe(text, **gen_kwargs)[0]["summary_text"]
        return output
