import argparse
import time

from model_transformer import PolymerModel

import pandas as pd
import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from torchmetrics import R2Score
import random

from PolymerSmilesTokenization import PolymerSmilesTokenizer

from dataset import Downstream_Dataset, DataAugmentation

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


from copy import deepcopy
random.seed(1)
np.random.seed(seed=1)

torch.manual_seed(0)
torch.cuda.manual_seed(0)


"""Model"""

class DownstreamRegression(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.TransformerModel = deepcopy(TransformerModel)

        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.TransformerModel.embed_dim, self.TransformerModel.embed_dim),
            nn.ReLU(),
            nn.Linear(self.TransformerModel.embed_dim, 1)
        )

    def forward(self, input_ids, attention_mask):
        # self.TransformerModel.mask = attention_mask
        outputs = self.TransformerModel(x=input_ids, mask=attention_mask)
        logits = outputs.mean(dim=1)

        output = self.Regressor(logits)
        return output

"""Train"""

def train(model, optimizer, scheduler, loss_fn, train_dataloader, device):
    model.train()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prop = batch["prop"].to(device).float()
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask).float()
        loss = loss_fn(outputs.squeeze(), prop.squeeze())
        loss.backward()
        optimizer.step()
        scheduler.step()

    return None

def test(model, loss_fn, train_dataloader, test_dataloader, device, scaler, optimizer, scheduler, epoch):
    r2score = R2Score()
    train_loss = 0
    test_loss = 0
    # count = 0
    model.eval()
    with (torch.no_grad()):
        train_pred, train_true, test_pred, test_true = torch.tensor([]), torch.tensor([]), torch.tensor(
            []), torch.tensor([])

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            train_loss += loss.item() * len(prop)
            train_pred = torch.cat([train_pred.to(device), outputs.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().to("cpu"), train_true.flatten().to("cpu")).item()
        print("train RMSE = ", np.sqrt(train_loss))
        print("train r^2 = ", r2_train)

        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prop = batch["prop"].to(device).float()
            outputs = model(input_ids, attention_mask).float()
            outputs = torch.from_numpy(scaler.inverse_transform(outputs.cpu().reshape(-1, 1)))
            prop = torch.from_numpy(scaler.inverse_transform(prop.cpu().reshape(-1, 1)))
            loss = loss_fn(outputs.squeeze(), prop.squeeze())
            test_loss += loss.item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), outputs.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])

        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().to("cpu"), test_true.flatten().to("cpu")).item()
        print("test RMSE = ", np.sqrt(test_loss))
        print("test r^2 = ", r2_test)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("r^2/train", r2_train, epoch)
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("r^2/test", r2_test, epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch}
    torch.save(state, finetune_config['save_path'])

    return train_loss, test_loss, r2_train, r2_test

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main(finetune_config):
    best_r2 = 0.0  
    """Data"""
    print("Train Test Split")
    """train-validation split"""
    def split(file_path):
        dataset = pd.read_csv(file_path).values
        data_ratio = float(finetune_config['data_ratio'])
        if 0 < data_ratio < 1:
            dataset, _ = train_test_split(dataset, test_size=1.0 - data_ratio, random_state=1)
            print("data_ratio is " + str(data_ratio))
        elif data_ratio == 1.0:
            print("data_ratio is 1.0")
        else:
            raise ValueError(
                "Invalid data_ratio value. It should be between 0 and 1 (exclusive). meaning it can be up to 1 but not 0. Provided value: {}".format(
                    finetune_config['data_ratio']))
        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=1)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        return train_data, test_data

    train_data, test_data = split(finetune_config['file_path'])


    if finetune_config['aug_flag']:
        print("Data Augmentation")
        DataAug = DataAugmentation(finetune_config['aug_indicator'])
        train_data = DataAug.smiles_augmentation(train_data)
        train_data = DataAug.combine_columns(train_data)
        test_data = DataAug.combine_columns(test_data)

    scaler = StandardScaler()
    train_data.iloc[:, 1] = scaler.fit_transform(train_data.iloc[:, 1].values.reshape(-1, 1))
    test_data.iloc[:, 1] = scaler.transform(test_data.iloc[:, 1].values.reshape(-1, 1))

    train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'])
    test_dataset = Downstream_Dataset(test_data, tokenizer, finetune_config['blocksize'])
    train_dataloader = DataLoader(train_dataset, finetune_config['batch_size'], shuffle=True,
                                  num_workers=finetune_config["num_workers"])
    test_dataloader = DataLoader(test_dataset, finetune_config['batch_size'], shuffle=False,
                                 num_workers=finetune_config["num_workers"])

    """Parameters for scheduler"""
    steps_per_epoch = train_data.shape[0] // finetune_config['batch_size']
    training_steps = steps_per_epoch * finetune_config['num_epochs']
    warmup_steps = int(training_steps * finetune_config['warmup_ratio'])

    """Train the model"""
    model = DownstreamRegression(drop_rate=finetune_config['drop_rate']).to(device)
    model = model.double()
    loss_fn = nn.MSELoss()

    optimizer = AdamW(
        [
            {"params": model.TransformerModel.parameters(), "lr": finetune_config['lr_rate'],
             "weight_decay": 0.0},
            {"params": model.Regressor.parameters(), "lr": finetune_config['lr_rate_reg'],
             "weight_decay": finetune_config['weight_decay']},
        ]
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)
    torch.cuda.empty_cache()
    train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0 
    count = 0
    for epoch in range(finetune_config['num_epochs']):
        start_time = time.time()
        print("epoch: %s/%s" % (epoch + 1, finetune_config['num_epochs']))
        train(model, optimizer, scheduler, loss_fn, train_dataloader, device)
        train_loss, test_loss, r2_train, r2_test = test(model, loss_fn, train_dataloader,
                                                        test_dataloader, device, scaler,
                                                        optimizer, scheduler, epoch)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if r2_test > best_test_r2:
            best_train_r2 = r2_train
            best_test_r2 = r2_test
            train_loss_best = train_loss
            test_loss_best = test_loss
            count = 0
        else:
            count += 1

        if r2_test > best_r2:
            best_r2 = r2_test
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'epoch': epoch}
            torch.save(state, finetune_config['best_model_path'])  # save the best model
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        if count >= finetune_config['tolerance']:
            print("Early stop")
            if best_test_r2 == 0:
                print("Poor performance with negative r^2")
            break

    print('\n', 'test RMSE =', np.sqrt(test_loss_best), '\n', 'test r^2 =', best_test_r2)
    writer.flush()

if __name__ == "__main__":
    finetune_config = yaml.load(open("config_Transformer.yaml", "r"), Loader=yaml.FullLoader)
    print(finetune_config)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.is_available()


    print("No Pretrain")

    parser = argparse.ArgumentParser() 
    parser.add_argument('-v', '--vocab_size', default=50265, type=int)
    parser.add_argument('-e', '--embed_dim', default=384, type=int)
    parser.add_argument('-f', '--ffn_dim', default=384, type=int)
    parser.add_argument('-t', '--n_transformer_blocks', default=6, type=int)
    parser.add_argument('-H', '--n_heads', default=12, type=int)
    args = parser.parse_args()

    TransformerModel = PolymerModel(embed_dim=args.embed_dim,
                                num_heads=args.n_heads,
                                num_blocks=args.n_transformer_blocks,
                                vocab_size=args.vocab_size,
                                ffn_dim=args.ffn_dim)
    
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])

    max_token_len = finetune_config['blocksize']

    """Run the main function"""
    main(finetune_config)


