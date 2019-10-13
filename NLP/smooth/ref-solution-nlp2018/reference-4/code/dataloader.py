import copy
import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader


class BertDataset(Dataset):
    def __init__(self, file_path, mode):
        super(BertDataset, self).__init__()
        with open(file_path, "r") as f:
            self.data_dict = json.load(f)
        self.mode = mode
        if self.mode == "test":
            self.X = []
            self.ID = []
            for key, value in self.data_dict.items():
                self.ID.append(key)
                self.X.append(value["ids"])
        else:
            self.X = []
            self.Y = []
            for key, value in self.data_dict.items():
                self.X.append(value["ids"])
                self.Y.append(int(value["label"]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.mode == "test":
            return self.X[index], self.ID[index]
        else:
            return self.X[index], self.Y[index]

class BertDataLoader(DataLoader):
    def __init__(self, dataset, mode, max_len, **kwargs):
        super(BertDataLoader, self).__init__(
            dataset=dataset,
            collate_fn=self.collate_fn,
            **kwargs
        )
        self.mode = mode
        self.max_len = max_len

    def pad(self, x, max_len):
        if len(x) > max_len - 2:
            x = x[:(max_len - 2)]

        x = [101] + x + [102]
        tokens_type = [0] * len(x)
        input_mask = [1] * len(x)

        while len(x) < max_len:
            x.append(0)
            tokens_type.append(0)
            input_mask.append(0)

        assert len(x) == max_len
        assert len(tokens_type) == max_len
        assert len(input_mask) == max_len

        return x, tokens_type, input_mask

    def collate_fn(self, batches):
        X_list, Y_list = list(zip(*batches))
        lens = [len(item) for item in X_list]

        Y_list = np.hstack(Y_list)

        max_len = min(max(lens), self.max_len)

        ids_all = []
        type_all = []
        mask_all = []
        for x in X_list:
            tokens_ids, tokens_type, input_mask = self.pad(x, max_len)
            ids_all.append(tokens_ids)
            type_all.append(tokens_type)
            mask_all.append(input_mask)

        ids_all = np.array(ids_all)
        type_all = np.array(type_all)
        mask_all = np.array(mask_all)

        if self.mode == "test":
            return torch.LongTensor(ids_all), torch.LongTensor(type_all), torch.LongTensor(mask_all), Y_list
        else:
            return torch.LongTensor(ids_all), torch.LongTensor(type_all), torch.LongTensor(mask_all), torch.tensor(Y_list)


if __name__ == "__main__":
    dataset = BertDataset("preprocessed_data/dev.json", "train")
    dataloader = BertDataLoader(dataset, mode="train", batch_size=4, num_workers=4, shuffle=False)
    for index, (ids, type, mask, label) in enumerate(dataloader):
        print(ids)
        print(type)
        print(mask)
        print(label)
        break