import torch
import random
import numpy as np
from tqdm import tqdm 

from datasets.data import *

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from utils import build_model, build_ft_cls
from parser import args


print(f'====== Fewshot evaluation on {args.ft_dataset} ======')
print(f'------ n_runs: {args.n_runs}')
print(f'------ k_way: {args.k_way}')
print(f'------ n_shot: {args.n_shot}')
print(f'------ n_query: {args.n_query}')


device = torch.device("cuda:%d" % args.rank)
# ------ 1. used finetuned model
# model = build_ft_cls()
# ------ 2. used pretrained model
model = build_model()
model = model.to(device)

save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', 'model_best.pth')
state_dict = torch.load(save_path)
model.load_state_dict(state_dict)

if args.ft_dataset == 'ModelNet40':
    # ModelNet40 - Few Shot Learning
    data_train, label_train = load_modelnet_data('train')
    data_test, label_test = load_modelnet_data('test')
    n_cls = 40
    
elif args.ft_dataset == 'ScanObjectNN':
    # ScanObjectNN - Few Shot Learning
    data_train, label_train = load_ScanObjectNN('train')
    data_test, label_test = load_ScanObjectNN('test')
    n_cls = 15

label_idx = {}
for key in range(n_cls):
    label_idx[key] = []
    for i, label in enumerate(label_train):
        # if label[0] == key:
        if label == key:
            label_idx[key].append(i)

acc = []
for run in tqdm(range(args.n_runs)):
    k = args.k_way ; n_shot = args.n_shot ; n_q = args.n_query

    k_way = random.sample(range(n_cls), k)

    data_support = [] ; label_support = [] ; data_query = [] ; label_query = []
    for i, class_id in enumerate(k_way):
        support_id = random.sample(label_idx[class_id], n_shot)
        query_id = random.sample(list(set(label_idx[class_id]) - set(support_id)), n_q)

        pc_support_id = data_train[support_id]
        pc_query_id = data_train[query_id]
        data_support.append(pc_support_id)
        label_support.append(i * np.ones(n_shot))
        data_query.append(pc_query_id)
        label_query.append(i * np.ones(n_q))

    data_support = np.concatenate(data_support)
    label_support = np.concatenate(label_support)
    data_query = np.concatenate(data_query)
    label_query = np.concatenate(label_query)

    model = model.eval()
    feats_train = []
    labels_train = []
    for i in range(k * n_shot):
        data = torch.from_numpy(np.expand_dims(data_support[i], axis = 0))
        label = int(label_support[i])
        data = data.to(device)
        with torch.no_grad():
            ''' # NOTE: use model `backbone_feats` '''
            feat = model(data)[1]
        feat = feat.tolist()
        feats_train.append(feat)
        labels_train.append(label)

    feats_train = np.array(feats_train)
    # squeeze the dimension whose value is 1
    feats_train = np.squeeze(feats_train)
    labels_train = np.array(labels_train)

    feats_test = []
    labels_test = []
    for i in range(k * n_q):
        data = torch.from_numpy(np.expand_dims(data_query[i], axis = 0))
        label = int(label_query[i])
        data = data.to(device)
        with torch.no_grad():
            ''' # NOTE: use model `backbone_feats` '''
            feat = model(data)[1]
        feat = feat.tolist()
        feats_test.append(feat)
        labels_test.append(label)

    feats_test = np.array(feats_test)
    # squeeze the dimension whose value is 1
    feats_test = np.squeeze(feats_test)
    labels_test = np.array(labels_test)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(feats_train)
    model_tl = SVC(C=args.svm_coff, kernel='linear')
    model_tl.fit(train_scaled, labels_train)
    
    test_scaled = scaler.transform(feats_test)
    accuracy = model_tl.score(test_scaled, labels_test) * 100
    acc.append(accuracy)

    # print(f"C = {c} : {model_tl.score(test_scaled, labels_test)}")
    # print(f"Run - {run + 1} : {accuracy}")

print(f'------ Acc: {np.mean(acc)} +/- {np.std(acc)}\n')