import os
import torch
import copy 
import numpy as np
import pandas as pd
from tqdm import tqdm

class Trainer:
    def __init__(self,gpu, args, get_matric):
        self.args         = args
        self.gpu          = gpu
        self.im_size      = args.im_size
        self.get_matric   = get_matric
        self.batch_size   = args.batch_size
        self.best_acc     = 0.0
        self.not_improved = 0
        self.train_df     = self.store_results()
        self.test_df      = self.store_results()
        self.trainig_dataset_name = args.dataset
        self.create_dir_to_save()
        
    def store_results(self):
        df = pd.DataFrame(columns=['epoch','loss', 'acc', 'apcer', 'bpcer'])
        return df
    
    def create_dir_to_save(self):
        csv_path = f"{self.args.csv_dir}/csv/patch_based_cnn/{self.args.dataset}/{self.args.protocol}/{self.args.protocol_type}/{self.im_size}"
        os.makedirs(csv_path,exist_ok=True)
        checkpoint_path = f"{self.args.output_dir}/patch_based_cnn/{self.args.dataset}/{self.args.protocol}/{self.args.protocol_type}/{self.im_size}"
        os.makedirs(checkpoint_path,exist_ok=True)
        
        
    def eval_model(self, model, dataloader, criterion, best_acc, epoc_no):

        total = 0
        running_corrects = 0
        running_loss = 0.0
        tq = tqdm(dataloader, desc="Validating")
        y_hat_all, y_true_all = [], []

        for idx, (im, label, acc_label) in enumerate(tq):
            im = im.reshape(-1, im.size(2), self.im_size, self.im_size)
            label = label.reshape(-1)
            im = im.cuda(self.gpu)
            label = label.type(torch.LongTensor).cuda(self.gpu)
            out = model(im)
            outputs = model(im)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, label)
            running_loss += loss.item() * im.size(0)
            preds     = preds.cpu().detach().numpy().reshape(-1).tolist()
            acc_label = acc_label.cpu().detach().numpy().reshape(-1).tolist()
            predictions = []
            
            for b in range(0, len(preds), int(self.args.no_patch) ):
                pred = preds[b:b+int(self.args.no_patch)]
                pred = np.mean(pred)
                pred = 1 if float(pred)>=0.5 else 0
                predictions.extend([pred])
            acc, apcer, bpcer = self.get_matric(acc_label, predictions)
            
            y_hat_all.extend(predictions)
            y_true_all.extend(acc_label)
            total += im.size(0)
            tq.set_postfix(iter=idx, loss=running_loss, acc=acc, apcer=apcer, bpcer=bpcer)

        epoch_loss = running_loss / total
        epoch_acc, epoch_apcer, epoch_bpcer = self.get_matric(acc_label, predictions)
        if epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
            self.save_model(f"{self.args.output_dir}/patch_based_cnn/{self.args.dataset}/{self.args.protocol}/{self.args.protocol_type}/{self.im_size}/BEST.pth", best_model)
            self.not_improved = 0
        else:
            self.not_improved +=1
            
        if (epoc_no % int(self.args.save_epoch_freq)) == 0:
            epoch_model = copy.deepcopy(model.state_dict())
            self.save_model(f"{self.args.output_dir}/patch_based_cnn/{self.args.dataset}/{self.args.protocol}/{self.args.protocol_type}/{self.im_size}/EPOCH_{epoc_no}.pth", epoch_model)
            
        return epoch_loss, epoch_acc, epoch_apcer, epoch_bpcer

    def save_model(self, save_path, model):
        print(f"Best Acc: {self.best_acc * 100:.4f}% and Model Saved")
        torch.save(model, f"{save_path}")
        
    def save_csv(self, df, types="train"):
        csv_path = f"{self.args.csv_dir}/csv/patch_based_cnn/{self.args.dataset}/{self.args.protocol}/{self.args.protocol_type}/{self.im_size}"
        df.to_csv(f"{csv_path}/{types}.csv", index=False)

    def train_model(self, model, criterion, dataloaders, optimizer, epochs):
        best_acc = 0.0
        train_loader = dataloaders["train"]
        val_loader   = dataloaders["val"]
        
        for i in range(epochs):
            total            = 0
            running_loss     = 0.0
            running_corrects = 0
            model.train()
            y_hat_all, y_true_all = [], []
            tq = tqdm(train_loader, desc="Training")
            for idx, (im, label, acc_label) in enumerate(tq):
                im    = im.reshape(-1, im.size(2), self.im_size, self.im_size)
                label = label.reshape(-1)
                im    = im.cuda(self.gpu)
                label = label.type(torch.LongTensor).cuda(self.gpu)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs   = model(im)
                    loss      = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                    _, preds  = torch.max(outputs, 1)

                running_loss += loss.item() * im.size(0)
                preds         = preds.cpu().detach().numpy().reshape(-1).tolist()
                acc_label     = acc_label.cpu().detach().numpy().reshape(-1).tolist()
                predictions   = []

                for b in range(0, len(preds), int(self.args.no_patch) ):
                    pred = preds[b:b+int(self.args.no_patch)]
                    pred = np.mean(pred)
                    pred = 1 if float(pred)>=0.5 else 0
                    predictions.extend([pred])
                acc, apcer, bpcer = self.get_matric(acc_label, predictions)

                y_hat_all.extend(predictions)
                y_true_all.extend(acc_label)
                total += im.size(0)
                tq.set_postfix(iter=idx, loss=running_loss, acc=acc, apcer=apcer, bpcer=bpcer)

            epoch_loss = running_loss / total
            epoch_acc, epoch_apcer, epoch_bpcer = self.get_matric(acc_label, predictions)
            # storing results
            row_to_add = [i, epoch_loss, epoch_acc, epoch_apcer, epoch_bpcer]
            self.train_df.loc[i] = row_to_add
            self.save_csv(self.train_df, types="train")

            print(f'Epoch: {i + 1} -- Train Loss: {epoch_loss:.4f} -- Acc: {epoch_acc:.4f}% -- Apcer: {epoch_apcer:.4f} -- Bpcer: {epoch_bpcer:.4f}')

            model = model.eval()

            v_epoch_loss, v_epoch_acc, v_epoch_apcer, v_epoch_bpcer = self.eval_model(model, val_loader, criterion, self.best_acc, epoc_no=i)
            # storing results
            row_to_add = [i, v_epoch_loss, v_epoch_acc, v_epoch_apcer, v_epoch_bpcer]
            self.test_df.loc[i] = row_to_add
            self.save_csv(self.test_df, types="test")
            
            print(f'Epoch: {i + 1} -- Valid Loss: {v_epoch_loss:.4f} -- Acc: {v_epoch_acc:.4f}% -- Apcer: {v_epoch_apcer:.4f} -- Bpcer: {v_epoch_bpcer:.4f}')
            if self.not_improved == 5:
                print("Model accbest_accuracy didn't improved for 12 consecutive epocs.\n")
                break #df.to_csv(f"{csv_path}/{types}.csv", index=False)
                
    def test_model(self, model, dataloader, criterion):

        total = 0
        running_corrects = 0
        running_loss = 0.0
        tq = tqdm(dataloader, desc="Testing")
        y_hat_all, y_true_all = [], []

        for idx, (im, label, acc_label) in enumerate(tq):
            im = im.reshape(-1, im.size(2), self.im_size, self.im_size)
            label = label.reshape(-1)
            im = im.cuda(self.gpu)
            label = label.type(torch.LongTensor).cuda(self.gpu)
            out = model(im)
            outputs = model(im)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, label)
            running_loss += loss.item() * im.size(0)
            preds     = preds.cpu().detach().numpy().reshape(-1).tolist()
            acc_label = acc_label.cpu().detach().numpy().reshape(-1).tolist()
            predictions = []
            
            for b in range(0, len(preds), int(self.args.no_patch) ):
                pred = preds[b:b+int(self.args.no_patch)]
                pred = np.mean(pred)
                pred = 1 if float(pred)>=0.5 else 0
                predictions.extend([pred])
            acc, apcer, bpcer = self.get_matric(acc_label, predictions)
            
            y_hat_all.extend(predictions)
            y_true_all.extend(acc_label)
            total += im.size(0)
            tq.set_postfix(iter=idx, loss=running_loss, acc=acc, apcer=apcer, bpcer=bpcer)

        epoch_loss = running_loss / total
        epoch_acc, epoch_apcer, epoch_bpcer = self.get_matric(acc_label, predictions)
        
        return epoch_loss, epoch_acc, epoch_apcer, epoch_bpcer
