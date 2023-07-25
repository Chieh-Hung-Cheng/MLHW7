import json
import os
from Config import Config
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import random


class Preprocessor:
    def __init__(self):
        with open(os.path.join(Config.data_path, "hw7_train.json"), "r", encoding="utf-8") as f:
            self.train_dict = json.load(f)
        with open(os.path.join(Config.data_path, "hw7_dev.json"), "r", encoding="utf-8") as f:
            self.valid_dict = json.load(f)
        with open(os.path.join(Config.data_path, "hw7_test.json"), "r", encoding="utf-8") as f:
            self.test_dict = json.load(f)


class TextDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.json_dict = None
        if split == "train":
            with open(os.path.join(Config.data_path, "hw7_train.json"), "r", encoding="utf-8") as f:
                # Train dict with key: questions and paragraphs
                # questions: list of dicts, each with key id(index), paragraph_id, question_text, answer_text, answer_start, answer_end
                # paragraph: list of string, each the content
                self.json_dict = json.load(f)
        elif split == "valid":
            with open(os.path.join(Config.data_path, "hw7_dev.json"), "r", encoding="utf-8") as f:
                self.json_dict = json.load(f)
        elif split == "test":
            with open(os.path.join(Config.data_path, "hw7_test.json"), "r", encoding="utf-8") as f:
                self.json_dict = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        # tokenized: list of dict, with attribute: tokens(char), ids(int)
        # Questions
        self.questions_list = self.json_dict["questions"]
        self.questions_text = [i["question_text"] for i in self.questions_list]
        self.questions_tokenized = self.tokenizer(self.questions_text, add_special_tokens=False)
        # Paragraphs
        self.paragraphs_list = self.json_dict["paragraphs"]
        self.paragraph_tokenized = self.tokenizer(self.paragraphs_list, add_special_tokens=False)
        # Length Limitations
        self.max_question_length = 60
        self.max_paragraph_length = 150
        self.max_total_length = 1 + self.max_question_length + 1 + self.max_paragraph_length + 1 # 213

    def __len__(self):
        return len(self.questions_list)

    def __getitem__(self, item):
        # Question
        question_dict = self.questions_list[item]
        question_tokenized = self.questions_tokenized[item]
        # Paragraph
        paragraph_id = question_dict["paragraph_id"]
        paragraph_tokenized = self.paragraph_tokenized[paragraph_id]
        if self.split == "train":
            # Answer
            answer_start = question_dict["answer_start"] # char
            answer_end = question_dict["answer_end"] # char
            # char_to_token(original_index): Get the index of the token in the encoded output
            start_token_idx = paragraph_tokenized.char_to_token(answer_start) # int
            end_token_idx = paragraph_tokenized.char_to_token(answer_end) # int
            # Extract
            question_ids_extracted = question_tokenized.ids[0: self.max_question_length]
            paragraph_start, paragraph_end = self.calculate_interval(start_token_idx, end_token_idx, len(paragraph_tokenized))
            paragraph_token_extracted = paragraph_tokenized.ids[paragraph_start: paragraph_end]
            # Add special tokens
            input_ids_question = [101] + question_ids_extracted + [102]
            input_ids_paragraph = paragraph_token_extracted + [102]
            start_token_idx_shifted = start_token_idx + len(input_ids_question) - paragraph_start
            end_token_idx_shifted = end_token_idx + len(input_ids_question) - paragraph_start
            # Padding
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), \
                   start_token_idx_shifted, end_token_idx_shifted

        elif self.split == "valid" or self.split == "test":
            doc_stride = 150
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            for i in range(0, len(paragraph_tokenized), doc_stride):
                input_ids_question = [101] + question_tokenized.ids[0:self.max_question_length] + [102]
                input_ids_paragraph = paragraph_tokenized.ids[i: i+doc_stride] + [102]
                # Padding
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)
        else:
            raise ValueError(f"split {self.split} doesn't exist!")

    def calculate_interval(self, start_idx, end_idx, paragraph_token_length):
        '''end_idx += 1
        left_count = start_idx
        right_count = paragraph_token_length - end_idx
        answer_token_length = end_idx - start_idx
        if left_count < right_count:
            low_bound = max(0, self.max_paragraph_length - answer_token_length - right_count)
            if left_count >= low_bound:
                tokens_from_left = random.randint(low_bound, left_count+1)
            else:
                tokens_from_left = left_count
            tokens_from_right = self.max_paragraph_length - answer_token_length - tokens_from_left
        else:
            # if right count is smaller than left count
            low_bound = max(0, self.max_paragraph_length - answer_token_length - left_count)
            if right_count >= low_bound:
                tokens_from_right = random.randint(low_bound, right_count+1)
            else:
                tokens_from_right = right_count
            tokens_from_left = self.max_paragraph_length - answer_token_length - tokens_from_right
        paragraph_start = start_idx - tokens_from_left
        paragraph_end = end_idx + tokens_from_right
        assert(paragraph_end - paragraph_start == self.max_paragraph_length)'''
        mid = (start_idx + end_idx) // 2
        paragraph_start = max(0,min(mid - self.max_paragraph_length // 2, paragraph_token_length - self.max_paragraph_length))
        paragraph_end = paragraph_start + self.max_paragraph_length
        return paragraph_start, paragraph_end

    def padding(self, input_ids_question, input_ids_paragraph):
        padding_length = self.max_total_length - len(input_ids_question) - len(input_ids_paragraph)
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_length
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_length
        attention_mask = [1] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_length
        return input_ids, token_type_ids, attention_mask


if __name__ == '__main__':
    pass
