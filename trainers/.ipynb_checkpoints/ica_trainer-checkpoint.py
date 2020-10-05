import numpy as np
import torch
import torchvision
import copy

batch_size = 256

class TestingModel(torch.nn.Module):
    def __init__(self, filters, set_weights="combined"):
        super(TestingModel, self).__init__()
        self.set_weights = set_weights
        if self.set_weights == "combined" or self.set_weights == "imagenet":
            self.conv1 = torch.nn.Conv2d(3, 27, 3)
            self.conv1.weight.data = torch.nn.Parameter(torch.tensor(filters).float())
            # self.conv2 = torch.nn.Conv2d(27, 128, 4, stride=2)
            self.fc = torch.nn.Linear(1330668, 2)

        elif self.set_weights == "separated":
            self.conv1 = torch.nn.Conv2d(3, 54, 3)
            self.conv1.weight.data = torch.nn.Parameter(torch.tensor(filters).float())
            self.fc = torch.nn.Linear(2661336, 2)
            # self.conv2 = torch.nn.Conv2d(54, 128, 4, stride=2)

        elif self.set_weights == "random":
            self.conv1 = torch.nn.Conv2d(3, 27, 3)
            self.fc = torch.nn.Linear(1330668, 2)
            # self.conv2 = torch.nn.Conv2d(54, 128, 4, stride=2)

        # self.conv3 = torch.nn.Conv2d(128, 128, 4, stride=2)
        # self.conv4 = torch.nn.Conv2d(128, 64, 4, stride=2)
        # self.fc1 = torch.nn.Linear(43264, 512)
        
        # self.fc2 = torch.nn.Linear(512, 2)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        #         print(x.shape)
        x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))
# #         print(x.shape)
        if self.set_weights == "combined" or self.set_weights == "imagenet" or self.set_weights == "random":
            x = x.view(-1, 27*222*222)
        elif self.set_weights == "separated":
            x = x.view(-1, 54*222 * 222)

        # x = self.relu(self.fc1(x))
        x = self.soft(self.fc(x))
        
        return x



class Trainer:
    def __init__(self, filters, set_weights="ica", freeze_layer=False, resume_training=False):
        self.set_weights = set_weights
        self.resume_training = resume_training
        self.model = self.load_model(filters)
        self.freeze_layer = freeze_layer

        self.model.cuda()

    def load_model(self, filters):
        model = TestingModel(filters, self.set_weights)
        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
        if self.resume_training:
            model.load_state_dict(torch.load(f"../ckpts/ica/{self.set_weights}.pth"))
            for param in model.parameters():
                param.requires_grad = True

        if self.freeze_layer:
            for param in model.parameters():
                param.requires_grad = False
                break

        return model
    def load_dataset(self, data_path, shuffle, transform):
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )
        #     print(dataset[0])
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=16,
            shuffle=shuffle
        )
        return data_loader


    def eval_model(self, model, dataloader, criterion, best_acc):

        # corrects = 0
        total = 0
        running_corrects = 0
        running_loss = 0.0
        for idx, (im, label) in enumerate(dataloader):
            im = im.cuda()
            label = label.type(torch.LongTensor).cuda()
            out = model(im)
            outputs = model(im)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, label)

            running_loss += loss.item() * im.size(0)
            running_corrects += torch.sum(preds == label.data)
            total += im.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
            self.save_model(f"../ckpts/ica/{self.set_weights}.pth", best_model)
        print(total, running_corrects, running_loss)
        print(f'Validation Loss: {epoch_loss:.4f} -- Acc: {epoch_acc*100:.4f}%')
        return best_acc

    def train_model(self, model, criterion, dataloaders, optimizer, epochs):
        
        best_acc = 0.0
        
        train_loader = dataloaders["train"]
        val_loader = dataloaders["validation"]
        
        for i in range(epochs):
            running_loss = 0.0
            running_corrects = 0
            total = 0
            model.train()
            print(f"number of batches {len(train_loader)}")
            for idx, (im, label) in enumerate(train_loader):
                im = im.cuda()
                label = label.type(torch.LongTensor).cuda()
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):

                    outputs = model(im)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * im.size(0)
                running_corrects += torch.sum(preds == label.data)
                total += im.size(0)
                if idx == 10:
                    print(f"{idx*batch_size} images are done!!!")
            print(total, running_corrects, running_loss)
            epoch_loss = running_loss /  total
            epoch_acc = running_corrects.double() / total


            print(f'Epoch: {i + 1} -- Train Loss: {epoch_loss:.4f} -- Acc: {epoch_acc*100:.4f}%')

            print("Evaluating the model...")
            model.eval()

            best_acc = self.eval_model(model, val_loader, criterion, best_acc)

            #  model.train()

     

    def run(self, transformers, data_path, epochs):
        loss = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        dataloaders = {}
        dataloaders["train"] = self.load_dataset(f"{data_path}/Train", True, transformers["train"])
        dataloaders["validation"] = self.load_dataset(f"{data_path}/Test", True, transformers["validation"])
        self.train_model(self.model, loss, dataloaders, optimizer, epochs)
        return self.model

    def save_model(self, save_path, model):
        torch.save(model, f"{save_path}")
