from Config import Config
import torch
from tqdm import tqdm
from Trainer import guess_transform_answer
import os
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from TextDataParser import TextDataset
from Utils import Utils

class Tester:
    def __init__(self):
        Config.model = AutoModelForQuestionAnswering.from_pretrained("10070725").to(Config.device)

    def infer(self):
        Config.model.eval()
        answers = []
        with torch.no_grad():
            for i, (input_ids_b, token_type_ids_b, attention_mask_b) in enumerate(tqdm(Config.test_loader)):
                # Test with batch_size 1
                input_ids = input_ids_b.squeeze(dim=0).to(Config.device)
                token_type_ids = token_type_ids_b.squeeze(dim=0).to(Config.device)
                attention_mask = attention_mask_b.squeeze(dim=0).to(Config.device)
                # Model forward pass
                prediction = Config.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                answers.append(guess_transform_answer(prediction, input_ids))
        self.save_answers_csv(answers)

    def save_answers_csv(self, answers):
        with open(os.path.join(Config.output_path, f"result_{Config.time_string}.csv"), "w", encoding="utf-8") as f:
            f.write("ID,Answer\n")
            for i, answer in enumerate(answers):
                f.write(f"{i},{answer.replace(',', '')}\n")


if __name__ == "__main__":
    Utils.initialize_time_path()
    Config.test_loader = DataLoader(TextDataset("test"), batch_size=1, shuffle=False, pin_memory=True)
    tester = Tester()
    tester.infer()