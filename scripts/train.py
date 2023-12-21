import sys
sys.path.append('..')
from loss import calculate_bce
import torch.optim as optim
import torch
from tqdm import tqdm
from metrics import calculate_precision, calculate_recall, calculate_cf
import warnings
import wandb
warnings.filterwarnings('ignore')

def train_script(model, train_loader, val_loader, epochs, device, note):
    wandb.init(project='dog_cats', notes=note)
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model = model.to(device)
    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        train_prec, val_prec = 0.0, 0.0
        train_rec, val_rec = 0.0, 0.0
        train_tp, train_fp, train_fn, train_tn = 0.0, 0.0, 0.0, 0.0
        val_tp, val_fp, val_fn, val_tn = 0.0, 0.0, 0.0, 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_prec, running_rec = 0.0, 0.0, 0.0
            running_tp, running_fp, running_fn, running_tn = 0.0, 0.0, 0.0, 0.0
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for img, labels in tepoch:
                    img, labels = img.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(img)
                        loss = calculate_bce(outputs, labels)
                        precision = calculate_precision(outputs, labels)
                        recall = calculate_recall(outputs, labels)
                        tp, fn, fp, tn = calculate_cf(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    running_prec += precision.item()
                    running_rec += recall.item()
                    running_tp += tp.item()
                    running_fn += fn.item()
                    running_fp += fp.item()
                    running_tn += tn.item()
                    tepoch.set_postfix(loss = loss.item(), precision = precision.item(), recall = recall.item(), tp = tp.item(), fn = fn.item(), fp = fp.item(), tn = tn.item())
            if phase == 'train':
                train_loss = running_loss / len(dataloaders[phase])
                train_prec = running_prec / len(dataloaders[phase])
                train_rec = running_rec / len(dataloaders[phase])
                train_tp = running_tp 
                train_fn = running_fn 
                train_fp = running_fp 
                train_tn = running_tn 
                # print(f'Train loss: {train_loss}')
                # print(f'Train prec: {train_prec}')
                # print(f'Train rec: {train_rec}')
                # print(f'Train tp: {train_tp}')
                # print(f'Train fn: {train_fn}')
                # print(f'Train fp: {train_fp}')
                # print(f'Train tn: {train_tn}')
            else:
                val_loss = running_loss / len(dataloaders[phase])
                val_prec = running_prec / len(dataloaders[phase])
                val_rec = running_rec / len(dataloaders[phase])
                val_tp = running_tp 
                val_fn = running_fn 
                val_fp = running_fp 
                val_tn = running_tn 
                # print(f'Val loss: {val_loss}')
                # print(f'Val prec: {val_prec}')
                # print(f'Val rec: {val_rec}')
                # print(f'Val tp: {val_tp}')
                # print(f'Val fn: {val_fn}')
                # print(f'Val fp: {val_fp}')
                # print(f'Val tn: {val_tn}')
        wandb.log({'train_loss': train_loss, 
                   'train_prec': train_prec, 
                   'train_rec': train_rec, 
                   'train_tp': train_tp, 
                   'train_fn': train_fn, 
                   'train_fp': train_fp, 
                   'train_tn': train_tn, 
                   'val_loss': val_loss, 
                   'val_prec': val_prec, 
                   'val_rec': val_rec, 
                   'val_tp': val_tp, 
                   'val_fn': val_fn, 
                   'val_fp': val_fp, 
                   'val_tn': val_tn})