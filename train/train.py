import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm

def main_train(model, loss_fn, optimizer, n_epochs, dataloader, device) -> None:
    # Put model in train mode
    model.train()
    n = len(dataloader)

    start_time_global = time.time()

    # Loop through data loader data batches
    loss_list = np.zeros(n_epochs)
    for epoch in range(n_epochs):

        start_time_epoch = time.time()

        train_loss = 0

        for X, organ, y in tqdm(dataloader):
            print(X.shape, organ, y.shape)
            # 1. Forward pass
            model_output = model(X.to(device))

            # 2. Calculate and accumulate loss
            loss = loss_fn(model_output,y.to(device)) #model_output[:,1,:,:] correspond au masque de la classe 2 = la zone d'intérêt, la classe 1 correpsond ua background
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        loss_list[epoch] = train_loss
        print(
            f"epoch {epoch+1}/{n_epochs},"
            f" train_loss = {train_loss/n:.2e},"
            #f" validation_loss = {test_model(model, loss_fn):.2e},"
            f" time spent during this epoch = {time.time() - start_time_epoch:.2f}s,"
            f" total time spent = {time.time() - start_time_global:.2f}s"
        )

    return loss_list

def main_train_batch1(model, loss_fn, optimizer, n_epochs, dataset, device) -> None:
    # Put model in train mode
    model.train()
    n = len(dataset)

    start_time_global = time.time()

    # Loop through data loader data batches
    loss_list = np.zeros(n_epochs)
    for epoch in range(n_epochs):

        start_time_epoch = time.time()

        train_loss = 0

        for X, organ, y in tqdm(dataset):
            # 1. Forward pass
            X = torch.unsqueeze(X,0)
            model_output = model(X.to(device))

            # 2. Calculate and accumulate loss
            loss = loss_fn(model_output,y.to(device)) #model_output[:,1,:,:] correspond au masque de la classe 2 = la zone d'intérêt, la classe 1 correpsond ua background
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        loss_list[epoch] = train_loss
        print(
            f"epoch {epoch+1}/{n_epochs},"
            f" train_loss = {train_loss/n:.2e},"
            #f" validation_loss = {test_model(model, loss_fn):.2e},"
            f" time spent during this epoch = {time.time() - start_time_epoch:.2f}s,"
            f" total time spent = {time.time() - start_time_global:.2f}s"
        )

    return loss_list