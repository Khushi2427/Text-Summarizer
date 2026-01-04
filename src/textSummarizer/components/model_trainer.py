import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk



class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        # 🚨 HARD DISABLE MPS
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False

        device = torch.device("cpu")

        print("✅ Training on:", device)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )

        dataset = load_from_disk(self.config.data_path)

        # 🔽 EXTREMELY IMPORTANT FOR MAC
        dataset["train"] = dataset["train"].select(range(30))
        dataset["validation"] = dataset["validation"].select(range(10))

        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,

            gradient_accumulation_steps=1,
            save_strategy="no",

            fp16=False,
            bf16=False,

            logging_steps=20,
            report_to="none",
            no_cuda=True   # 🚨 THIS IS CRITICAL
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"]
        )

        trainer.train()

        model.save_pretrained(
            os.path.join(self.config.root_dir, "pegasus-samsum-model")
        )
        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
        )
