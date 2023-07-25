from Config import Config
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from TextDataParser import TextDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from accelerate import Accelerator
import os
import json


def guess_transform_answer(prediction, input_ids):
    max_prob = float("-inf")
    answer = ''
    for i in range(len(input_ids)):
        start_prob, start_idx = torch.max(prediction.start_logits[i], dim=0)
        end_prob, end_idx = torch.max(prediction.end_logits[i], dim=0)
        total_prob = start_prob + end_prob

        if total_prob > max_prob:
            max_prob = total_prob
            answer = Config.tokenizer.decode(input_ids[i][start_idx: end_idx + 1])
    answer = answer.replace(' ', '')
    return answer

def evaluate(data, output):
    ##### Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong

    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob

        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob:
            max_prob = prob
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = Config.tokenizer.decode(data[0][0][k][start_index : end_index + 1])

    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')


class Trainer:
    def __init__(self):
        pass

    def train_loop(self):
        for epoch in range(Config.epochs):
            mean_train_loss, mean_train_accuracy = self.train_one_epoch()
            self.valid_one_epoch()
            self.summarize_epoch()
        Config.model.save_pretrained(Config.time_string)

    def train_one_epoch(self):
        Config.model.train()
        logging_step = 100
        # Macroscope
        total_loss = 0
        success_pred_count = 0
        total_count = 0
        train_pbar = tqdm(Config.train_loader)
        # Microscope
        train_loss = 0
        train_acc = 0

        for i, (input_ids_b, token_type_b, attention_mask_b, start_idx_b, end_idx_b) in enumerate(train_pbar):
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
            train_loss += prediction.loss
            Config.accelerator.backward(prediction.loss)
            Config.optimizer.step()
            # Calculate Accuracy
            accurate_b = (start_pred == start_idx_b) & (end_pred == end_idx_b)
            success_pred_count += torch.sum(accurate_b)
            total_count += len(input_ids_b)
            train_acc += accurate_b.float().mean()

            if (i+1) % logging_step == 0:
                train_pbar.set_postfix({"train_loss": f"{train_loss.item() / logging_step: .3f}",
                                        "train_acc": f"{train_acc / logging_step:.3%}"})
                train_loss = 0
                train_acc = 0


        return total_loss / total_count, success_pred_count / total_count

    def valid_one_epoch(self):
        with open(os.path.join(Config.data_path, "hw7_dev.json"), "r", encoding="utf-8") as f:
            valid_dict = json.load(f)
        valid_questions_list = valid_dict["questions"]

        Config.model.eval()
        with torch.no_grad():
            success_count = 0
            for i, (input_ids_b, token_type_b, attention_mask_b) in enumerate(tqdm(Config.valid_loader)):
                # Put on GPU and squeeze dimension
                input_ids = input_ids_b.squeeze(dim=0).to(Config.device)
                token_type = token_type_b.squeeze(dim=0).to(Config.device)
                attention_mask = attention_mask_b.squeeze(dim=0).to(Config.device)
                # Model Forward
                prediction = Config.model(input_ids=input_ids, token_type_ids=token_type, attention_mask=attention_mask)
                success_count += guess_transform_answer(prediction, input_ids) == valid_questions_list[i]["answer_text"]
            print(f"Validation accuracy: {success_count / len(Config.valid_loader): .3%}")

    def summarize_epoch(self):
        pass


if __name__ == '__main__':
    Config.model = AutoModelForQuestionAnswering.from_pretrained("10460725").to(Config.device)
    Config.valid_loader = DataLoader(TextDataset(split="valid"), batch_size=1, shuffle=False, pin_memory=True)
    trainer = Trainer()
    trainer.valid_one_epoch()
