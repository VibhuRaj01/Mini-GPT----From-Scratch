import json
from model import GPTModel
from utils import (
    calc_loss_batch,
    evaluate_model,
)
from dataset import create_dataloader
import torch


def prepare_data(path, config, train_ratio=0.9):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text_data = f.read()
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_dataloader = create_dataloader(
        train_data, batch_size=4, max_length=config["context_length"], stride=32
    )
    val_dataloader = create_dataloader(
        val_data,
        batch_size=4,
        max_length=config["context_length"],
        stride=32,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
):
    train_losses, val_losses, track_token = [], [], []
    tokens_seen, global_step = 0, -1

    for _ in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token.append(tokens_seen)

                print(
                    f"Step {global_step}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Tokens Seen = {tokens_seen}"
                )
    return train_losses, val_losses, track_token


if __name__ == "__main__":
    json_path = "config.json"
    with open(json_path, "r") as f:
        config = json.load(f)

    train_loader, val_loader = prepare_data(
        path="the-verdict.txt", config=config, train_ratio=0.8
    )  # type: ignore
    num_epochs = 50
    eval_freq = 5

    device = torch.device(config.get("device"))
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader)
    )

    train_losses, val_losses, tokens_seen = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=5,
    )
