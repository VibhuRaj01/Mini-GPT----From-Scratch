import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import json


class gptdataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(
        self,
    ):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = gptdataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


# if __name__ == "__main__":
#     json_path = "config.json"
#     with open(json_path, "r") as f:
#         config = json.load(f)
#     device = torch.device(config.get("device"))

#     torch.manual_seed(69)
#     txt = "Hello, how are you doing today? I hope you're having a great day!"
#     dataloader = create_dataloader(txt, batch_size=2, max_lenght=10, stride=5)

#     for input_ids, target_ids in dataloader:
#         print("Input IDs:", input_ids)
#         print("Target IDs:", target_ids)
#         break
