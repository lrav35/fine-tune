import tqdm
import uuid
import matplotlib as plt

def should_run_eval(total_steps, freq, current_step):
    return current_step % (total_steps // freq) == 0

def eval(model, val_data):
    print("evaluating model...\n")
    metric = evaluate.load("accuracy")
    preds_and_true = {'preds': [], 'labels': []}
    model.eval()
    for batch in val_data:
        batch = {
            "input_ids": batch["input_ids"].to(model.device),
            "labels": batch["labels"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
        }
        with torch.no_grad():
            outputs = model(**batch)
        
        # record loss
        val_loss = outputs.loss.item()

        # compute accuracy
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        metric.add_batch(predictions=preds, references=batch["labels"])
        preds_and_true['preds'].extend([p.item() for p in preds+1])
        preds_and_true['labels'].extend([l.item() for l in batch["labels"]+1])

        # pbar.update(1)
    acc_result = metric.compute()
    print(f"Accuracy: {acc_result['accuracy']}")
    return acc_result['accuracy'], preds_and_true, val_loss

def generate_loss_image(train_loss: list, val_loss: list, output_dir: str):
    iter_x_ind = np.linspace(0, len(val_loss)-1, num=len(train_loss))
    interp_val_loss = np.interp(iter_x_ind, np.arange(len(val_loss)), val_loss)
    plt.plot(np.array(train_loss), color='b', label='training loss')
    plt.plot(interp_val_loss, color='r', label='validation loss')
    plt.savefig(output_dir + '/loss_plot.jpeg')

def save_model(model, outpath: str, current_epoch: int, current_step: int, results: dict):
    print(f"saving model at epoch: {current_epoch}, step: {current_step}")
    outpath += f"/model"
    model.save_pretrained(outpath)    

def train_model(model, epochs, train_dataloader, val_dataloader, train_steps, optimizer, lr_scheduler, save_path: str):
    pbar = tqdm(range(train_steps))

    run_id = str(uuid.uuid4())
    print(f"model id :: {run_id}")
    output_dir = f"{save_path}/outputs/bert/{run_id}"
    model.train()
    best_accuracy = 0.0
    train_epoch_loss = []
    val_epoch_loss = []
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

            train_batch_loss.append(loss.item())

            # backward
            loss.backward()

            # update weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # evaluate and save model
            if should_run_eval(len(train_dataloader), 5, current_step):
                accuracy, results, val_loss = eval(model, val_dataloader)
                val_epoch_loss.append(val_loss)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    save_model(model, output_dir, current_epoch, current_step, results)
                else:
                    print('skipping model save...')
                print(f"current best accuracy: {best_accuracy}\n")
                model.train()
            pbar.update(1)
        train_epoch_loss.extend(train_batch_loss)
    generate_loss_image(train_epoch_loss, val_epoch_loss, output_dir)