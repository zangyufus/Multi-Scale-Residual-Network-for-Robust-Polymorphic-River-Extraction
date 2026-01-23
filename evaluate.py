import time
import argparse
import codecs
import yaml
from tqdm import tqdm
from newloader import *
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from model.Net import Net
from utils.utils import get_img_patches, merge_pred_patches
from pathlib import Path
import torch
import cv2
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./results.prf')
parser.add_argument('--thresh_step', type=float, default=0.01)
args = parser.parse_args()

config = yaml.load(open('./config.yml'), Loader=yaml.FullLoader)
batch_size_va = int(config['batch_size_va']) 


def calculate_metrics(mskp, msk):

    mskp = (mskp > 0.5).astype(np.uint8) 
    msk = (msk > 0).astype(np.uint8)

    tp = np.sum((mskp == 1) & (msk == 1))
    fp = np.sum((mskp == 1) & (msk == 0))
    fn = np.sum((mskp == 0) & (msk == 1))
    tn = np.sum((mskp == 0) & (msk == 0))
  
    # Accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

    # IoU
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    return accuracy, iou, precision, recall, f1_score, 

def linear_stretch(img, lower_percent=2, upper_percent=98):
    if np.any(np.isnan(img)) or np.any(np.isinf(img)):
        img = np.nan_to_num(img, nan=0, posinf=255, neginf=0)

    lower_value = np.percentile(img, lower_percent)
    upper_value = np.percentile(img, upper_percent)

    stretched_img = (img - lower_value) / (upper_value - lower_value) * 255
    stretched_img = np.clip(stretched_img, 0, 255).astype(np.uint8)

    if np.any(np.isnan(stretched_img)) or np.any(np.isinf(stretched_img)):
        stretched_img = np.nan_to_num(stretched_img, nan=0, posinf=255, neginf=0)

    return stretched_img

def save_results(input_list, output_path):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)

def save_sample(img_path, msk, msk_pred, fname, name=''):

    with rasterio.open(img_path) as src:
        img_array = src.read() 
    
    img_height, img_width = img_array.shape[1], img_array.shape[2]

    if img_array.shape[0] >= 3:
        img_rgb = img_array[:3, :, :] 
        img_rgb = np.moveaxis(img_rgb, 0, -1) 
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # OpenCV 使用 BGR，转换为 RGB
    else:
        print(f"Warning: The image at {img_path} does not have enough channels for RGB.")
        return

    img_rgb = linear_stretch(img_rgb)

    msk = msk.astype(int)
    mskp = (msk_pred > 0.5).astype(np.uint8)


    if msk.shape != (img_height, img_width):
        msk = cv2.resize(msk, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    if mskp.shape != (img_height, img_width):
        mskp = cv2.resize(mskp, (img_width, img_height), interpolation=cv2.INTER_LINEAR)

    # 保存预测 mask 结果
    fname = fname[0]
    fname_without = os.path.splitext(fname)[0]  
    mask_pred_path = os.path.join(config['save_result'], f"{fname_without}.png")
    cv2.imwrite(mask_pred_path, mskp * 255)
    print(f"Prediction mask saved at {mask_pred_path}")

    # 可视化 rgb mask maskp
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].imshow(img_rgb)

    axs[1].axis('off')
    axs[1].imshow(msk * 255, cmap='gray')

    axs[2].axis('off')
    axs[2].imshow(mskp * 255, cmap='gray') 

    result_path = os.path.join(config['save_result'], f"{fname_without}" + '_combined.png')
    plt.savefig(result_path)
    print(f"Sample saved at {result_path}")

    #plt.show()  # 显示结果

    plt.close()
    plt.close()


# 读取配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = config['path_to_testdata']
DIR_IMG  = os.path.join(data_path, 'images')
DIR_MASK = os.path.join(data_path, 'masks')
img_names  = [path.name for path in Path(DIR_IMG).glob('*.tif')]
mask_names = [path.name for path in Path(DIR_MASK).glob('*.tif')]
total_test_samples = len(list(Path(config['path_to_testdata']).glob('images/*.tif')))
print(f"total_test_samples: {total_test_samples}")

test_dataset = TifDataset(img_dir=DIR_IMG, img_fnames=img_names, mask_dir=DIR_MASK, mask_fnames=mask_names, isTrain=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size_va, shuffle=False, drop_last=True)
print(f'test_dataset:{len(test_dataset)}')

# 加载模型
Net = Net(n_classes=int(config['number_classes']))
Net = Net.to(device)
Net.load_state_dict(torch.load(config['saved_model_final'], map_location='cpu')['model_weights'])

save_samples = True  # 是否保存样本
total_accuracy = 0
total_iou = 0
total_precision = 0
total_recall = 0
total_f1_score = 0
total_kappa = 0


with torch.no_grad():
    print('val_mode')
    val_loss = 0
    times = 0
    Net.eval()

    for itter, batch in enumerate(tqdm(test_loader)):
        img = batch['image'].to(device, dtype=torch.float)
        img_path = batch['img_path'][0]
        fname = batch['fname']
        msk = batch['mask']
        
        patch_totensor = TifToTensor()
        preds = []
        
        start = time.time()
        
        for i in range(img.size(0)): 
            img_single = img[i].cpu().numpy()
            img_single = np.transpose(img_single, (1, 2, 0)) 
            patches, patch_locs = get_img_patches(img_single)
            for patch in patches:
                #patch_n = patch_totensor(Image.fromarray(patch))  # 将patch转换为tensor
                patch_n = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32)
                X = patch_n.unsqueeze(0).to(device, dtype=torch.float) 
                msk_pred = torch.sigmoid(Net(X))
                mask = msk_pred.cpu().detach().numpy()[0, 0]
                preds.append(mask)
        
        mskp = merge_pred_patches(img_single, preds, patch_locs) 

        kernel = np.array(
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ], dtype=np.uint8)
        mskp = cv2.morphologyEx(mskp, cv2.MORPH_CLOSE, kernel, iterations=1).astype(float)  # 闭运算
        
        end = time.time()
        times += (end - start)
        
        if itter < total_test_samples and save_samples:
            save_sample(img_path, msk.numpy()[0, 0], mskp, fname, name=str(itter + 1))
            accuracy, iou, precision, recall, f1_score = calculate_metrics(mskp, msk.numpy()[0, 0])
            total_accuracy += accuracy
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score
            # total_kappa += kappa
            print(f"Accuracy: {accuracy:.4f}")
            print(f"IoU: {iou:.4f}")
            print(f"f1_score: {f1_score:.4f}")
            # print(f"kappa: {kappa:.4f}")

    avg_accuracy = total_accuracy / total_test_samples
    avg_iou = total_iou / total_test_samples
    avg_f1_score = total_f1_score / total_test_samples
    avg_precision = total_precision / total_test_samples
    avg_recall = total_recall / total_test_samples
    print(f"Overall Accuracy: {avg_accuracy:.4f}")
    print(f"Overall IoU: {avg_iou:.4f}")
    print(f"Overall precision: {avg_precision:.4f}")
    print(f"Overall recall: {avg_recall:.4f}")
    print(f"Overall F1 Score: {avg_f1_score:.4f}")

    print(f'Running time of each image: {times/total_test_samples:.4f}s')

