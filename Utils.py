import torch
import numpy as np
from Config import Config
from torch.utils.data import random_split
import os
import random
from torch.utils.data import DataLoader
from datetime import datetime, timedelta


class Utils:
    # Initialize time, path, randomness
    @staticmethod
    def initialize_time_path(local_flag=True):
        # Time Related
        time_now = datetime.now()
        if not local_flag:
            # Running on Google Colab
            time_now += timedelta(hours=8)
        Config.time_string = f"{time_now.hour:02d}{time_now.minute:02d}{time_now.month:02d}{time_now.day:02d}"
        # Path
        Config.base_path = os.getcwd()
        Config.save_path = os.path.join(Config.base_path, ".model")
        Config.output_path = os.path.join(Config.base_path, "output")
        if not os.path.isdir(Config.save_path):
            os.mkdir(Config.save_path)
        if not os.path.isdir(Config.output_path):
            os.mkdir(Config.output_path)
        if local_flag:
            # Data path on local
            Config.data_path = r"D:\ML_Dataset\HW7"
        else:
            # Data path on Colab
            Config.data_path = None
        # Check all set
        print(f"{Config.base_path=}")
        print(f"{Config.data_path=}")
        print(f"{Config.save_path=}")
        print(f"{Config.output_path=}")
        print(f"{Config.time_string=}")

    @staticmethod
    def fix_randomness():
        random.seed(Config.seed)
        torch.manual_seed(Config.seed)
        np.random.seed(Config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.seed)
            torch.cuda.manual_seed_all(Config.seed)
        print("Randomness Fixed")

    @staticmethod
    def set_train_valid_loader(train_dataset, valid_dataset, collate_function):
        Config.train_loader = DataLoader(train_dataset,
                                         batch_size=Config.batch_size,
                                         shuffle=True,
                                         num_workers=Config.num_worker,
                                         drop_last=True,
                                         pin_memory=True,  # Prevent VM paging
                                         collate_fn=collate_function)
        Config.valid_loader = DataLoader(valid_dataset,
                                         batch_size=Config.batch_size,
                                         shuffle=True,
                                         num_workers=Config.num_worker,
                                         drop_last=True,
                                         pin_memory=True,
                                         collate_fn=collate_function)
        print("Train, Valid DataLoader Complete")

    # Dataset, DataLoader Related
    @staticmethod
    def set_test_loader(test_dataset, collate_function):
        Config.test_loader = DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=Config.num_worker,
                                        drop_last=False,
                                        pin_memory=True,
                                        collate_fn=collate_function)
        print("Test DataLoader Complete")

    @staticmethod
    def split_train_valid_dataset(train_dataset):
        original_train_length = len(train_dataset)
        actual_valid_length = int(original_train_length * Config.valid_ratio)
        actual_train_length = original_train_length - actual_valid_length
        return random_split(train_dataset, [actual_train_length, actual_valid_length],
                            generator=torch.Generator().manual_seed(Config.seed))

    # Model Related
    @staticmethod
    def set_model_related(model_class):
        Config.model = model_class().to(Config.device)
        if Config.load_ckpt:
            Config.model.load_state_dict(torch.load(os.path.join(Config.save_path, f"model_{Config.load_name}.ckpt")))
            print(f"Model name {Config.load_name} Loaded")
        else:
            print("Training a new model are we?")

        Config.criterion = torch.nn.CrossEntropyLoss()
        Config.optimizer = torch.optim.AdamW(Config.model.parameters(), lr=Config.learning_rate)
        # Config.scheduler = get_cosine_schedule_with_warmup(Config.optimizer, Config.warmup_steps, Config.epochs)
        print("Model, Criterion, Optimizer Complete")
