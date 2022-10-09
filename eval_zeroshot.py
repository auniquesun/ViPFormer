import os, datetime
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV

import torch
from torch.utils.data import DataLoader

from datasets.data import ModelNet40SVM, ScanObjectNNSVM

from utils import build_model

from parser import args

from fvcore.nn import FlopCountAnalysis


device = torch.device("cuda")
save_path = os.path.join('runs', args.proj_name, args.exp_name, 'models', args.pc_model_file)
state_dict = torch.load(save_path)

pc_model, _ = build_model(device)
model = pc_model
print('\n')
print(model)
print('\n')
model.load_state_dict(state_dict)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total parameters:', pytorch_total_params)
# total parameters: 8,490,240

if args.pt_dataset == "ModelNet40":
    train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)
elif args.pt_dataset == "ScanObjectNN":
    train_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)
    test_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=args.num_test_points),
                                batch_size=args.test_batch_size, shuffle=True)

model = model.eval()
with torch.no_grad():
    feats_train = []
    labels_train = []
    for i, (data, label) in enumerate(train_loader):
        if args.pt_dataset == "ModelNet40":
            labels = list(map(lambda x: x[0],label.tolist()))
        elif args.pt_dataset == "ScanObjectNN":
            labels = label.tolist()
        data = data.to(device)
        # model(data)[1] is the features output by CrossFormer backbone
        feats = model(data)[1].tolist()
        feats_train.extend(feats)
        labels_train.extend(labels)

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)
    print('feats_train.shape:', feats_train.shape)

    feats_test = []
    labels_test = []
    for i, (data, label) in enumerate(test_loader):
        if args.pt_dataset == "ModelNet40":
            labels = list(map(lambda x: x[0],label.tolist()))
        elif args.pt_dataset == "ScanObjectNN":
            labels = label.tolist()
        data = data.to(device)
        # model(data)[1] is the features output by CrossFormer backbone
        feats = model(data)[1].tolist()
        feats_test.extend(feats)
        labels_test.extend(labels)

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
    print('feats_test.shape:', feats_test.shape)

flops = FlopCountAnalysis(model, data)
print('fvcore - total flops:', flops.total())
# total flops scanobjectnn: 122,689,855,488.0 / test_batch_size(160)
# total flops modelnet40: 82,603,294,784.0 /test_batch_size(160)

# ------ Linear SVM
# Linear SVM parameter C, can be tuned
c = args.svm_coff 
linear_svm = svm.SVC(C = c, kernel ='linear')
linear_svm.fit(feats_train, labels_train)
print(f"Linear SVM, C = {c} : {linear_svm.score(feats_test, labels_test)}")

# ------ RBF SVM
rbf_svm = svm.SVC(C = c, kernel ='rbf')
rbf_svm.fit(feats_train, labels_train)
print(f"RBF SVM, C = {c} : {rbf_svm.score(feats_test, labels_test)}")

# ------ grid search parameters for SVM
print("\n\n")
print("="*37)
svm_clsf = svm.SVC()
# [1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1]
C_range = np.outer(np.logspace(-1, 1, 3), np.array([1, 5])).flatten()
parameters = {'kernel': ['linear', 'rbf'], 'C': C_range}
grid_clsf = GridSearchCV(estimator=svm_clsf, param_grid=parameters, n_jobs=8, verbose=1)

start_time = datetime.datetime.now()
print('Start Param Searching at {}'.format(str(start_time)))
grid_clsf.fit(feats_train, labels_train)
print('Elapsed time, param searching {}'.format(str(datetime.datetime.now() - start_time)))
sorted(grid_clsf.cv_results_.keys())

# scores = grid_clsf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
labels_pred = grid_clsf.best_estimator_.predict(feats_test)
print("Best Params via Grid Search Cross Validation on Train Split is: ", grid_clsf.best_params_)
print("Best Model's Accuracy on Test Dataset: {}".format(metrics.accuracy_score(labels_test, labels_pred)))
