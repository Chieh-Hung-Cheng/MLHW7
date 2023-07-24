from Config import Config
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from TextDataParser import TrainDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from accelerate import Accelerator


class Trainer:
    def __init__(self):
        pass

    def train_loop(self):
        for epoch in range(Config.epochs):
            mean_train_loss, mean_train_accuracy = self.train_one_epoch()
            self.valid_one_epoch()
            self.summarize_epoch()

    def train_one_epoch(self):
        Config.model.train()
        total_loss = 0
        success_pred_count = 0
        total_count = 0
        for input_ids_b, token_type_b, attention_mask_b, start_idx_b, end_idx_b in tqdm(Config.train_loader):
            # Put on GPU
            input_ids_b = input_ids_b.to(Config.device)
            token_type_b = token_type_b.to(Config.device)
            attention_mask_b = attention_mask_b.to(Config.device)
            start_idx_b = start_idx_b.to(Config.device)
            end_idx_b = end_idx_b.to(Config.device)
            # Model Forward
            prediction = Config.model(input_ids=input_ids_b,
                                      token_type_ids=token_type_b,
                                      attention_mask=attention_mask_b,
                                      start_positions=start_idx_b,
                                      end_positions=end_idx_b)
            start_pred = prediction.start_logits.argmax(dim=1)
            end_pred = prediction.end_logits.argmax(dim=1)
            # Calculate loss
            Config.optimizer.zero_grad()
            total_loss += prediction.loss
            Config.accelerator.backward(prediction.loss)
            Config.optimizer.step()
            # Calculate Accuracy
            accurate_b = (start_pred == start_idx_b) & (end_pred == end_idx_b)
            success_pred_count += torch.sum(accurate_b)
            total_count += len(input_ids_b)
        return total_loss / total_count, success_pred_count / total_count

    def valid_one_epoch(self):
        Config.model.eval()
        with torch.no_grad():
            for data in tqdm(Config.valid_loader):
                pass

    def summarize_epoch(self):
        pass


if __name__ == '__main__':
    Config.model = AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese")
    Config.model.to(Config.device)
    Config.optimizer = torch.optim.AdamW(Config.model.parameters(), lr=Config.learning_rate)
    Config.accelerator = Accelerator()
    Config.train_loader = DataLoader(TrainDataset(), batch_size=8, shuffle=True)
    trainer = Trainer()
    trainer.train_one_epoch()
