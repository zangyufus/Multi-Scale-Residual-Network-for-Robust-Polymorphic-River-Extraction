from __future__ import division
import copy
import torch.optim as optim
from utils.utils import *
from pathlib import Path
from newloader import TifDataset
from model.Net import Net
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

def calculate_iou(pred, target, n_classes):
    ious = []
    for cls in range(n_classes):
        # 计算每个类的 IoU
        intersection = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        iou = intersection / (union + 1e-6)
        ious.append(iou)
    return sum(ious) / len(ious)

def calculate_accuracy(pred, target):
    correct = (pred == target).sum().float()
    total = target.numel()
    accuracy = correct / total
    return accuracy

def calculate_masked_loss(pred, target):
    # Create a mask for valid pixels (non-NaN)
    valid_mask = ~torch.isnan(target)
    masked_pred = pred[valid_mask]
    masked_target = target[valid_mask]
    # Calculate loss only on valid pixels
    loss = criteria(masked_pred, masked_target)
    return loss


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setup_seed(42)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

data_tra_path = config['path_to_tradata']
data_val_path = config['path_to_valdata']

print({data_tra_path})

DIR_IMG_tra  = os.path.join(data_tra_path, 'images')
DIR_MASK_tra = os.path.join(data_tra_path, 'masks')

DIR_IMG_val  = os.path.join(data_val_path, 'images')
DIR_MASK_val = os.path.join(data_val_path, 'masks')

img_names_tra  = [path.name for path in Path(DIR_IMG_tra).glob('*.tif')]
mask_names_tra = [path.name for path in Path(DIR_MASK_tra).glob('*.tif')]

img_names_val  = [path.name for path in Path(DIR_IMG_val).glob('*.tif')]
mask_names_val = [path.name for path in Path(DIR_MASK_val).glob('*.tif')]

train_dataset = TifDataset(img_dir=DIR_IMG_tra, img_fnames=img_names_tra, mask_dir=DIR_MASK_tra, mask_fnames=mask_names_tra, isTrain=True)
valid_dataset = TifDataset(img_dir=DIR_IMG_val, img_fnames=img_names_val, mask_dir=DIR_MASK_val, mask_fnames=mask_names_val, resize=True)
print(f'train_dataset:{len(train_dataset)}')
print(f'valiant_dataset:{len(valid_dataset)}')

train_loader  = DataLoader(train_dataset, batch_size = int(config['batch_size_tr']), shuffle= True,  drop_last=True)
val_loader    = DataLoader(valid_dataset, batch_size = int(config['batch_size_va']), shuffle= False, drop_last=True)

Net = Net(n_classes = number_classes)
flops, params = get_model_complexity_info(Net, (16, 256, 256), as_strings=True, print_per_layer_stat=False)
print('flops: ', flops, 'params: ', params)
message = 'flops:%s, params:%s' % (flops, params)

Net = Net.to(device)

#load model
if os.path.exists(config['saved_model']):
    checkpoint = torch.load(config['saved_model'], map_location='cpu')
    Net.load_state_dict(checkpoint['model_weights'])
    best_val_loss = checkpoint.get('val_loss', float('inf')) 
    print("Loaded pretrained model successfully.")
else:
    print("Warning: Pretrained model not found, using random initialization.")

    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    Net.apply(init_weights)
    best_val_loss = float('inf')

#optimizer = optim.Adam(Net.parameters(), lr= float(config['lr']))
optimizer = optim.Adam(Net.parameters(), lr= float(config['lr']), weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = config['patience'])
criteria  = TverskyBCELoss(alpha=0.3, beta=0.7)


# visual
visualizer = Visualizer(isTrain=True)
log_name = os.path.join('./checkpoints', config['loss_filename'])
with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)


best_val_loss = float('inf')
best_epoch = -1
best_model = None


for ep in range(int(config['epochs'])):
    # ================== train ==================
    Net.train()
    epoch_loss = 0.0

    for batch in train_loader:
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask'].to(device, dtype=torch.long)

        pred, _ = Net(img, istrain=True)
        loss = calculate_masked_loss(pred, msk)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    mean_train_loss = epoch_loss / len(train_loader)
    print(f'tra_mode:Epoch>> {ep+1}, mean loss: {mean_train_loss:.6f}')

    # ================== validation ==================
    Net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device, dtype=torch.long)

            pred = Net(img)
            loss = calculate_masked_loss(pred, msk)
            val_loss += loss.item()

    mean_val_loss = val_loss / len(val_loader)
    print(f'validation on epoch>> {ep+1}, mean tloss>> {mean_val_loss:.6f}')
    visualizer.print_current_losses(ep + 1, mean_val_loss, isVal=True)

    # ================== save best model ==================
    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        best_epoch = ep + 1
        best_model = copy.deepcopy(Net.state_dict())

        print(f'****** New best model at epoch {best_epoch}, val_loss={best_val_loss:.6f} ******')

        torch.save({
            'model_weights': best_model,
            'val_loss': best_val_loss,
            'best_epoch': best_epoch
        }, config['saved_model'])

    scheduler.step(mean_val_loss)

# ================== training end ==================
if best_model is not None:
    visualizer.print_end(best_epoch, best_val_loss)
    torch.save({
        'model_weights': best_model,
        'val_loss': best_val_loss,
        'best_epoch': best_epoch
    }, config['saved_model_final'])
else:
    print('Warning: no best model found, saving last epoch model.')
    torch.save({
        'model_weights': Net.state_dict(),
        'val_loss': mean_val_loss,
        'best_epoch': int(config['epochs'])
    }, config['saved_model_final'])


print('Train finished')  



