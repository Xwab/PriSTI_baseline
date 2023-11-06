import imp
import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
from sklearn.preprocessing import StandardScaler
import random
from utils import get_randmask


def get_mask_rm(sample, k):
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = np.ones(sample.shape)
    length_index = np.array(range(mask.shape[0]))  # lenght of series indexes
    for channel in range(mask.shape[1]):
        perm = torch.randperm(len(length_index))
        perm = np.array(perm)
        idx = perm[0:k]
        mask[:, channel][idx] = 0

    return mask

def get_mask_block(sample, k):
    mask = np.ones(sample.shape)
    channel = random.randint(0, mask.shape[1]-1)
    series_id = random.randint(0, mask.shape[0]-k)
    mask[:,channel][series_id:series_id + k] = 0


    return mask



class ELE_Dataset(Dataset):    #默认缺失率为百分之25，为随机缺失
    def __init__(self, eval_length=16, target_dim=427, mode="train", missing_ratio = 0.1, is_interpolate = False):

        self.eval_length = eval_length
        self.target_dim = target_dim
        self.is_interpolate = is_interpolate

        '''path = "./data/hangzhou/hangzhou_meanstd.npy"
        with open(path, "rb") as f:
            self.mean = np.load(f)[0]
            self.std = np.load(f)[1]'''
        
        #train_data = np.load('./data/hangzhou/hangzhou_train.npy')
        #test_data = np.load('./data/hangzhou/hangzhou_test.npy')
        #valid_data = np.load('./data/hangzhou/hangzhou_valid.npy')

        #train_data = np.array(np.split(train_data[:6292], 242, 0))
        #test_data = np.array(np.split(test_data[:624], 24, 0))
        #valid_data = np.array(np.split(valid_data[:936], 36, 0))
        
        #train_data = np.array(np.split(train_data, 60, 0))
        #test_data = np.array(np.split(test_data, 6, 0))
        #valid_data = np.array(np.split(valid_data, 9, 0))
        train_data1 = np.load('./data/electricity/16/ele_train1.npy')
        train_data2 = np.load('./data/electricity/16/ele_train3.npy')
        train_data3 = np.load('./data/electricity/16/ele_train3.npy')
        train_data4 = np.load('./data/electricity/16/ele_train4.npy')
        train_data5 = np.load('./data/electricity/16/ele_train5.npy')
        train_data = np.concatenate((train_data1, train_data2, train_data3, train_data4, train_data5),axis=0)
        #train_data = np.load('./data/electricity/16/ele_train_16.npy')
        train_mask = np.load('./data/electricity/16/ele_train_randmask25_16.npy').astype('float')
        valid_data = np.load('./data/electricity/16/ele_valid_16.npy')
        valid_mask = np.load('./data/electricity/16/ele_valid_randmask25_16.npy').astype('float')
        test_data = np.load('./data/electricity/16/ele_test_16.npy')
        test_mask = np.load('./data/electricity/16/ele_test_randmask25_16.npy').astype('float')

        if mode == "train":
            observed_values = train_data
            origin_mask = train_mask
        elif mode == "test":
            observed_values = test_data
            origin_mask = test_mask
        else:
            observed_values = valid_data
            origin_mask = valid_mask

        data_shape = observed_values.shape
        scaler = StandardScaler().fit(observed_values.reshape(-1, observed_values.shape[-1]))
        observed_values = scaler.transform(observed_values.reshape(-1, observed_values.shape[-1])).reshape(data_shape)

        observed_masks = []
        gt_masks = []

        if mode == "train" or mode == "valid":
            for i in range(len(observed_values)):
                #ratio_list = [0.1, 0.15]
                observed_mask = origin_mask[i]
                #observed_mask = get_mask_rm(observed_values[i], int(eval_length * 0.25))
                observed_masks.append(observed_mask)
                masks = observed_mask.reshape(-1).copy()
                obs_indices = np.where(masks)[0].tolist()
                miss_indices = np.random.choice(
                obs_indices, 3, replace=False)
                masks[miss_indices] = 0
                gt_mask = masks.reshape(observed_mask.shape)
                gt_masks.append(gt_mask)

        else:
            for i in range(len(observed_values)):
                #ratio_list = [0.1, 0.15]
                observed_mask = origin_mask[i]
                observed_masks.append(observed_mask)
                masks = observed_mask.reshape(-1).copy()
                obs_indices = np.where(masks)[0].tolist()
                miss_indices = np.random.choice(
                obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False)
                masks[miss_indices] = 0
                gt_mask = masks.reshape(observed_mask.shape)
                gt_masks.append(gt_mask)


        observed_masks = np.array(observed_masks)
        ob_mask_tensor = torch.tensor(observed_masks).float()
        if mode == 'train':
            gt_masks = get_randmask(ob_mask_tensor)
        else:
            gt_masks = ob_mask_tensor
        #gt_masks = np.array(gt_masks)

        self.observed_values = observed_values
        self.observed_masks = observed_masks.astype("float32")
        #self.gt_masks = gt_masks.astype("float32")
        self.gt_masks = gt_masks
        self.scaler = scaler

        self.use_index_list = np.arange(len(self.observed_values))

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        if self.is_interpolate:
            tmp_data = torch.tensor(self.observed_values[index]).to(torch.float64)
            itp_data = torch.where(self.gt_masks[index] == 0, float('nan'), tmp_data).to(torch.float32)
            itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            s["coeffs"] = itp_data.numpy()
        return s

    def __len__(self):
        return len(self.use_index_list)

def get_dataloader(batch_size=16, device="cuda:0",is_interpolate=False, num_workers=4):

    # only to obtain total length of dataset
    dataset = ELE_Dataset()
    indlist = np.arange(len(dataset))

    np.random.seed(42)
    np.random.shuffle(indlist)

    dataset = ELE_Dataset(
        mode = "train", is_interpolate = is_interpolate
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1, num_workers=num_workers)
    valid_dataset = ELE_Dataset(
        mode = "valid", is_interpolate = is_interpolate
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0, num_workers=num_workers)
    test_dataset = ELE_Dataset(
        mode = "test", is_interpolate = is_interpolate
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    train_scaler = dataset.scaler
    valid_scaler = valid_dataset.scaler
    test_scaler = test_dataset.scaler

    return train_loader, valid_loader, test_loader, train_scaler, valid_scaler, test_scaler




        







        

        