import os, h5py, numpy as np
from torch.utils.data import Dataset


class S3DISDataset_HDF5(Dataset):
    """Chopped Scene"""

    def __init__(self, split='train', test_area=5):
        self.root = '/mnt/sdb/public/data/common-datasets/indoor3d_sem_seg_hdf5_data'
        self.all_files = self.getDataFiles(os.path.join(self.root, 'all_files.txt'))
        self.room_filelist = self.getDataFiles(os.path.join(self.root, 'room_filelist.txt'))
        self.scene_points_list = []
        self.semantic_labels_list = []

        for h5_filename in self.all_files:
            data_batch, label_batch = self.loadh5DataFile(os.path.join(self.root, h5_filename))
            self.scene_points_list.append(data_batch)
            self.semantic_labels_list.append(label_batch)

        self.data_batches = np.concatenate(self.scene_points_list, 0)
        self.label_batches = np.concatenate(self.semantic_labels_list, 0)

        test_area = 'Area_' + str(test_area)
        train_idxs, test_idxs = [], []

        for i, room_name in enumerate(self.room_filelist):
            if test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        assert split in ['train', 'test']
        if split == 'train':
            self.data_batches = self.data_batches[train_idxs, ...]
            self.label_batches = self.label_batches[train_idxs]
        else:
            self.data_batches = self.data_batches[test_idxs, ...]
            self.label_batches = self.label_batches[test_idxs]

    @staticmethod
    def getDataFiles(list_filename):
        return [line.rstrip() for line in open(list_filename)]

    @staticmethod
    def loadh5DataFile(PathtoFile):
        f = h5py.File(PathtoFile, 'r')
        return f['data'][:], f['label'][:]

    def __getitem__(self, index):
        points = self.data_batches[index, :]
        labels = self.label_batches[index].astype(np.int32)

        return points, labels

    def __len__(self):
        return len(self.data_batches)
