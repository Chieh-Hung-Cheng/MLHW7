from Utils import Utils
from Config import Config
from Trainer import Trainer
from TextDataParser import TextDataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from accelerate import Accelerator
import torch

def main():
    # Initialize
    Utils.initialize_time_path()
    Utils.fix_randomness()
    # Data
    Config.train_loader = DataLoader(TextDataset("train"), batch_size=8, shuffle=True, pin_memory=True)
    Config.valid_loader = DataLoader(TextDataset("valid"), batch_size=1, shuffle=False, pin_memory=True)
    # Model
    Config.model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese").to(Config.device)
    Config.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    Config.optimizer = torch.optim.AdamW(Config.model.parameters(), lr=Config.learning_rate)
    Config.accelerator = Accelerator()

    # Train
    trainer = Trainer()
    trainer.train_loop()


if __name__ == '__main__':
    main()
