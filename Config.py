import torch


class Config:
    # Time & Randomness
    time_string = None
    seed = 2

    # Paths
    base_path = None
    data_path = r"D:\ML_Dataset\HW7"
    save_path = None
    output_path = None

    # Load Models
    load_ckpt = True
    load_name = "05350717"

    # Dataset / DataLoader
    train_loader = None
    valid_loader = None
    test_loader = None
    valid_ratio = 0.1
    num_worker = 4

    # Training Related
    learning_rate = 1e-5
    epochs = 1
    batch_size = 32
    early_stop = 50
    valid_cycle = 2
    warmup_steps = 10

    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    scheduler = None
    accelerator = None



if __name__ == "__main__":
    print(Config.seed)
