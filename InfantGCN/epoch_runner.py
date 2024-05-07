import tqdm
import torch
from utils.logger import Logger

class EpochRunner:
    def __init__(self, model, device, logger, optimizer=None, scheduler_func=None, loss_func=None, EPOCHS=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler_func
        self.loss_func = loss_func
        self.EPOCHS = EPOCHS
        self.device = device
        self.log_handler = logger

    def run_epoch(self, mode, epoch_num, dataloader):
        if mode == 'train':
            return self.training_epoch(epoch_num, dataloader)
        else:
            return self.validation_epoch(epoch_num, dataloader)

    def training_epoch(self, epoch_num, dataloader):
        self.model.train()
        running_loss = 0
        total_predictions = 0
        correct_predictions = 0
        with torch.enable_grad():
            for batch_X, batch_y in tqdm.tqdm(dataloader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                batch_pred = self.model(batch_X)
                _, predicted_labels = torch.max(batch_pred, 1)
                loss = self.loss_func(batch_pred, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                scheduler = self.scheduler(epoch_num)
                scheduler.step()

                running_loss += loss.item()
                correct_predictions += (predicted_labels == batch_y).sum().item()
                total_predictions += batch_y.size(0)

            accuracy = correct_predictions / total_predictions
            epoch_loss = running_loss / len(dataloader)
            self.log_handler.log_training(epoch_num+1,epoch_loss,accuracy,scheduler.get_last_lr()[0])
            return accuracy

    def validation_epoch(self, epoch_num, dataloader):
        self.model.eval()
        all_preds, all_gts, all_feats = [], [], []
        total_predictions = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch_X, batch_y in tqdm.tqdm(dataloader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                batch_pred, batch_feats = self.model.extract_features(batch_X)
                _, predicted_labels = torch.max(batch_pred, 1)

                all_preds.extend(predicted_labels.cpu().numpy())
                all_gts.extend(batch_y.cpu().numpy())
                all_feats.extend(batch_feats.cpu().numpy())

                correct_predictions += (predicted_labels == batch_y).sum().item()
                total_predictions += batch_y.size(0)

            accuracy = correct_predictions / total_predictions
            self.log_handler.log_validation(epoch_num+1,accuracy)
            return accuracy, all_preds, all_gts, all_feats
