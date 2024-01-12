import tqdm
import uuid
import torch
import numpy as np
import wandb

def should_run_eval(total_steps, freq, current_step):
    return current_step % (total_steps // freq) == 0

def eval(model, val_data, wandb):
    print("evaluating model...\n")
    model.eval()
    losses = 0.0
    for step, batch in enumerate(val_data):
        batch = {
            "input_ids": batch["input_ids"].to(model.device),
            "labels": batch["labels"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        with torch.no_grad():
            outputs = model(**batch)
        
        # record loss
        loss = outputs.loss
        losses += loss.float()
    val_loss = losses / (step + 1)
    wandb.log(
        {
            "val_loss": val_loss
        }
    )

    return val_loss

def save_model(model, outpath: str, current_epoch: int, current_step: int):
    print(f"saving model at epoch: {current_epoch}, step: {current_step}")
    outpath += f"/model"
    model.save_pretrained(outpath)    

def train_model(model, epochs, train_dataloader, val_dataloader, train_steps, optimizer, lr_scheduler, save_path: str, wandb):
    pbar = tqdm(range(train_steps))

    run_id = str(uuid.uuid4())
    print(f"model id :: {run_id}")
    output_dir = f"{save_path}/outputs/bert/{run_id}"
    model.train()
    for epoch in range(epochs):
        current_epoch = epoch + 1
        train_batch_loss = []
        for step, batch in enumerate(train_dataloader):
            current_step = step + 1
            pbar.set_description(f"Epoch {current_epoch} :: Step {current_step}")

            batch = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }

            # forward
            outputs = model(**batch)
            loss = outputs.loss

            # write some kind of results logger / recording loss w wandb

            # backward
            loss.backward()

            # update weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # evaluate and save model
            if should_run_eval(len(train_dataloader), 5, current_step):
                val_loss = eval(model, val_dataloader, wandb)
                
                save_model(model, output_dir, current_epoch, current_step)
                model.train()
            pbar.update(1)
    # generate_loss_image(train_epoch_loss, val_epoch_loss, output_dir)