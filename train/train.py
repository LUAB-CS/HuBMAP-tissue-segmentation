import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

def main_train(model, loss_fn, o_classif_loss_fn, p_s_classif_loss_fn, optimizer, n_epochs, dataloader, device) -> None:
    # Put model in train mode
    model.train()
    n = len(dataloader)

    start_time_global = time.time()

    # Loop through data loader data batches
    loss_list = np.zeros(n_epochs)
    o_classif_loss_list = np.zeros(n_epochs)
    p_s_classif_loss_list = np.zeros(n_epochs)

    for epoch in range(n_epochs):

        start_time_epoch = time.time()

        train_loss = 0

        for X, organ, y, pixel_size in tqdm(dataloader):
            # 1. Forward pass
            model_output, organ_output, p_s_output = model(X.to(device))
            print(organ_output, p_s_output)
            # 2. Calculate and accumulate loss
            loss = loss_fn(model_output,y.to(device)) #model_output[:,1,:,:] correspond au masque de la classe 2 = la zone d'intérêt, la classe 1 correpsond ua background
            o_classif_loss = o_classif_loss_fn(organ_output, organ.to(device))
            p_s_classif_loss = p_s_classif_loss_fn(p_s_output, pixel_size.to(device))
            train_loss = 0.8*loss.item() + 0.1*o_classif_loss.item() + 0.1*p_s_classif_loss.item()
            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            full_loss = loss+o_classif_loss+p_s_classif_loss
            full_loss.backward()

            # 5. Optimizer step
            optimizer.step()

        loss_list[epoch] = loss.item()
        o_classif_loss_list[epoch] = o_classif_loss.item()
        p_s_classif_loss_list[epoch] = p_s_classif_loss.item()

        print(
            f"epoch {epoch+1}/{n_epochs},"
            f" train_loss = {train_loss/n:.2e},"
            #f" validation_loss = {test_model(model, loss_fn):.2e},"
            f" time spent during this epoch = {time.time() - start_time_epoch:.2f}s,"
            f" total time spent = {time.time() - start_time_global:.2f}s"
        )
        if epoch % 10 == 0 :

            torch.save(model, f"../model_save/save_epoch_{epoch}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.pt")

    return loss_list, o_classif_loss_list, p_s_classif_loss_list

def main_train_batch1(model, loss_fn, o_classif_loss_fn, p_s_classif_loss_fn, optimizer, n_epochs, dataset, device) -> None:
    # Put model in train mode
    model.train()
    n = len(dataset)

    start_time_global = time.time()

    # Loop through data loader data batches
    loss_list = np.zeros(n_epochs)
    o_classif_loss_list = np.zeros(n_epochs)
    p_s_classif_loss_list = np.zeros(n_epochs)

    for epoch in range(n_epochs):

        start_time_epoch = time.time()

        train_loss = 0

        for X, organ, y, pixel_size in tqdm(dataset):
            # 1. Forward pass
            X = torch.unsqueeze(X,0)
            model_output, organ_output, p_s_output = model(X.to(device))

            # 2. Calculate and accumulate loss
            loss = loss_fn(model_output,y.to(device)) #model_output[:,1,:,:] correspond au masque de la classe 2 = la zone d'intérêt, la classe 1 correpsond ua background
            o_classif_loss = o_classif_loss_fn(organ_output, organ.to(device))
            p_s_classif_loss = p_s_classif_loss_fn(p_s_output, pixel_size.to(device))
            train_loss = loss.item() + o_classif_loss.item() + p_s_classif_loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            full_loss = loss+o_classif_loss+p_s_classif_loss
            full_loss.backward()

            # 5. Optimizer step
            optimizer.step()


        loss_list[epoch] = loss.item()
        o_classif_loss_list[epoch] = o_classif_loss.item()
        p_s_classif_loss_list[epoch] = p_s_classif_loss.item()

        print(
            f"epoch {epoch+1}/{n_epochs},"
            f" train_loss = {train_loss/n:.2e},"
            #f" validation_loss = {test_model(model, loss_fn):.2e},"
            f" time spent during this epoch = {time.time() - start_time_epoch:.2f}s,"
            f" total time spent = {time.time() - start_time_global:.2f}s"
        )
        if epoch % 10 == 0 :

            torch.save(model, f"../model_save/save_epoch_{epoch}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.pt")

    return loss_list, o_classif_loss_list, p_s_classif_loss_list