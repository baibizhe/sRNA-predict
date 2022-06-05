from __future__ import print_function

import argparse
import time

import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from ACTINN import Classifier, CSV_IO
from ACTINN.utils import *


class Inferdataset(data.Dataset):
    def __init__(self, data):
        self.imgs = data

    def __getitem__(self, index):
        img = self.imgs[index]
        return torch.tensor(img)

    def __len__(self):
        return len(self.imgs)


"""
Written based on the original data processing done by ACTINN 
to preserve compatibility with datasets processed by the TF version
"""


class Mycelldataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # self.transforms = transform

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        return torch.tensor(img), torch.tensor(label)

    def __len__(self):
        return self.data.shape[0]


def type2label_dict(types):
    """
    Turn types into labels
    INPUT:
        types-> types of cell present in the data

    RETURN
     celltype_to_label_dict-> type_to_label dictionary

    """
    all_celltype = list(np.unique(types))
    print("all_celltype", all_celltype)
    celltype_to_label_dict = {}

    for i in range(len(all_celltype)):
        celltype_to_label_dict[all_celltype[i]] = i
    return celltype_to_label_dict


def convert_type2label(types, type_to_label_dict):
    """
    Convert types to labels
    INPUTS:
        types-> list of types
        type_to_label dictionary-> dictionary of cell types mapped to numerical labels

    RETURN:
        labels-> list of labels

    """
    types = list(types)
    labels = list()
    for type in types:
        labels.append(type_to_label_dict[type])
    #         labels.append(type_to_label_dict[type[0]])
    return labels


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def main():
    parser = argparse.ArgumentParser()
    # classifier options
    parser.add_argument('--ClassifierEpochs', type=int, default=50,
                        help='number of epochs to train the classifier, default = 50')
    parser.add_argument('--data_type', type=str, default="csv", help='type of train/test data, default="scanpy"')
    parser.add_argument("--save_iter", type=int, default=1, help="Default=1")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--print_frequency', type=int, default=8,
                        help='frequency of training stats printing, default=5')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.005')
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument('--clip', type=float, default=100, help='the threshod for clipping gradient')
    parser.add_argument("--step", type=int, default=1000,
                        help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=1000")
    parser.add_argument('--cuda', default=True, help='enables cuda, default = True')
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed, default = 0')
    parser.add_argument('--tensorboard', default=True, action='store_true', help='enables tensorboard, default True')
    parser.add_argument('--outf', default='./TensorBoard/', help='folder to output training stats for tensorboard')
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
    parser.add_argument("--expName", default="Adamax_TestB_18000TopGenes_0.4drop_lr0.0001", type=str, help="expName")
    parser.add_argument("--n_fold", default=5, type=int, help="n_fold")

    opt = parser.parse_args()
    print(opt)
    if not os.path.exists(os.path.join("expOutput", opt.expName)):
        try:
            os.makedirs(os.path.join("expOutput", opt.expName))
        except:
            pass

    # determin the device for torch
    ## if we are allowed to run things on CUDA
    if opt.cuda and torch.cuda.is_available():
        device = "cuda";
        print('==> Using GPU (CUDA)')
    else:
        device = "cpu"
        print('==> Using CPU')
        print('    -> Warning: Using CPUs will yield to slower training time than GPUs')
    train_path = "../cellPredict/data/train.h5ad"
    train_lab_path = "../cellPredict/data/train.anno.csv"
    # test_path = "../data/fusai/train.h5ad"

    test_path = "../cellPredict/data/test.hidden.h5ad"
    hca_data_path = "../data/heart.h5ad"
    submit_path = "../cellPredict/data/submit_A_example_fusai.csv"

    test_lab_path = "/home/ubuntu/SCRealVAE_68K/ACTINN_Data/68K_h5/test_lab.csv"
    print("建构数据集")
    infer_data_loader, cf_model, type_to_label_dict = None, None, None
    models = []
    n_fold = opt.n_fold
    data, data_label, infer_set = CSV_IO(train_path,
                                         train_lab_path,
                                         test_path,
                                         test_lab_path,
                                         batchSize=opt.batchSize,
                                         workers=opt.workers,
                                         hca_data_path=hca_data_path,
                                         slices=800000
                                         )
    #     data_label = np.vstack((data.obs.level1,data.obs.level2,data.obs.level3,data.obs.level4)).T
    data_label = data.obs.level4

    result_CSV = pd.DataFrame({"cell_id": infer_set.obs.barcode.values})
    type_to_label_dict = type2label_dict(data_label)
    print(f"    -> Cell types in training set: {type_to_label_dict}")

    data, infer_set = data.X, infer_set.X
    skf = StratifiedShuffleSplit(n_splits=int(n_fold), test_size=0.2)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(data_label, data_label)):

        print("==> training on fold", fold)
        train_set = data[train_idx]
        test_set = data[valid_idx]
        train_label = data_label[train_idx]
        test_label = data_label[valid_idx]
        print(f"    -> # trainng cells: {train_label.shape[0]}")
        names = list(np.unique(train_label))
        count_train = []
        count_valid = []
        for i in names:
            count_train.append((train_label == i).sum())
            count_valid.append((test_label == i).sum())
        for i in range(len(names)):
            print(names[i], " {} in train ".format(count_train[i]), "{} in valid".format(count_valid[i]))
        train_label = convert_type2label(train_label, type_to_label_dict)
        test_label = convert_type2label(test_label, type_to_label_dict)
        with open('../cellPredict/type_to_label_dict.txt', 'w') as f:
            f.writelines(str(type_to_label_dict))
        f.close()
        print(f"    *** Remember we the data is formatted as Cells X Genes ***")
        # inp_size = train_set[0].shape[0]

        # create DataLoaders
        train_dataset = Mycelldataset(train_set, train_label)
        test_dataset = Mycelldataset(test_set, test_label)
        infer_dataset = Inferdataset(infer_set)
        print(" train length {} | valid length {} | infer length {}".format(len(train_dataset), len(test_dataset),
                                                                            len(infer_dataset)))

        train_data_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, sampler=None,
                                       batch_sampler=None, num_workers=0, collate_fn=None,
                                       pin_memory=True)

        valid_data_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, sampler=None,
                                       batch_sampler=None, num_workers=0, collate_fn=None,
                                       pin_memory=True)
        infer_data_loader = DataLoader(infer_dataset, batch_size=12, shuffle=False, num_workers=1, drop_last=False)
        print("结束建构数据集")

        # get input output information for the network
        number_of_classes = len(type_to_label_dict)
        print(number_of_classes)


        start_time = time.time()
        cur_iter = 0

        print("Building the classifier model")
        cf_model = Classifier(output_dim=number_of_classes, input_size= 3635).to(device)
        # initilize the weights in our model
        cf_model.apply(init_weights)
        cf_criterion = torch.nn.CrossEntropyLoss()

        #     cf_criterion = torch.nn.BCEWithLogitsLoss()
        cf_optimizer = torch.optim.Adamax(params=cf_model.parameters(),
                                          lr=opt.lr,
                                          betas=(0.9, 0.999),
                                          eps=1e-08,
                                          # amsgrad=False,
                                          weight_decay=2e-2,
                                          )
        cf_decayRate = 0.95
        cf_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=cf_optimizer, gamma=cf_decayRate)
        print("\n Classifier Model \n")
        print(cf_model)

        """
        Training as the classifier (Should be done when we are warm-starting the VAE part)
        """

        # def train_classifier(cf_epoch, iteration, batch, cur_iter):
        # TRAIN
        print("---------------- ")
        print("==> Trainig Started ")
        print(f"    -> lr decaying after every {opt.step} steps")
        print(f"    -> Training stats printed after every {opt.print_frequency} epochs")
        for epoch in tqdm(range(0, opt.ClassifierEpochs + 1), desc="Classifier Training"):
            # save models
            if (epoch % opt.print_frequency == 0 and epoch != 0) or epoch == 0:
                print("==> evaluate on trainning set :")
                evaluate_classifier(train_data_loader, cf_model)
                print("==> evaluate on evaluation set :")
                evaluate_classifier(valid_data_loader, cf_model, classification_report=True)
                save_epoch = (epoch // opt.save_iter) * opt.save_iter
                save_checkpoint_classifier(cf_model, save_epoch, 0, 'PCAfold_' + str(fold))
            cf_model.train()
            for iteration, batch in enumerate(train_data_loader):
                # ============train Classifier Only============
                cf_optimizer.zero_grad()
                labels = batch[1]
                batch = batch[0]
                batch_size = batch.size(0)
                features = Variable(batch).to(device)
                true_labels = Variable(labels).to(device).long()
                info = f"\n====> Classifier Cur_iter: [{cur_iter}]: Epoch[{cur_iter}]({iteration}/{len(train_data_loader)}): time: {time.time() - start_time:4.4f}: "
                # =========== Update the classifier ================
                with autocast():
                    pred_cluster = cf_model(features)
                    loss = cf_criterion(pred_cluster.squeeze(), true_labels)
                loss.backward()
                cf_optimizer.step()
                if cur_iter % opt.step == 0 and cur_iter != 0:
                    cf_lr_scheduler.step()
                info += f'Loss: {loss.data.item():.4f} '
                if cur_iter == 0:
                    print("    ->Initial stats:", info)
                if epoch % opt.print_frequency == 0 and iteration == (len(train_data_loader) - 1):
                    print(info)
                cur_iter += 1

        save_epoch = (epoch // opt.save_iter) * opt.save_iter

        save_checkpoint_classifier(cf_model, save_epoch, 0, 'LASTPCA')
        print("==> Final evaluation on validation data: ")
        evaluate_classifier(valid_data_loader, cf_model, classification_report=True)
        print(f"==> Total training time {time.time() - start_time}");
        models.append(cf_model)

    result_softmax = []
    result = []
    my_dict2 = {y: x for x, y in type_to_label_dict.items()}

    with torch.no_grad():
        for _, batch in enumerate(infer_data_loader):
            outputs = torch.tensor([])
            for i in range(n_fold):
                models[i].eval()
                outputs = torch.cat((outputs, models[i](batch.to(device)).argmax(1).cpu().unsqueeze(1)), axis=1)

            outputs = torch.mode(outputs).values.numpy()
            result.extend(list(outputs))
            print(len(result))

    result = list(map(lambda x: my_dict2[x], list(result)))
    result_CSV["level4"] = result
    result_CSV.to_csv(os.path.join("expOutput", opt.expName, "submission_val_level4.csv"), index=False)

    mapLvel4toLevel1 = returnMaplevel4To1()
    convertedMateix = np.zeros(shape=(len(result_CSV.level4.values), 4), dtype=object)
    for i in range(len(result_CSV.level4.values)):
        for g in mapLvel4toLevel1:
            if result_CSV.level4.values[i] in g:
                stopIndex = g.index(result_CSV.level4.values[i])
                # print(stopIndex)
                convertedMateix[i][0:stopIndex] = g[0:stopIndex]
                convertedMateix[i][stopIndex:] = g[stopIndex]

    # print(convertedMateix)
    result_CSV["level1"] = convertedMateix[:, 0]
    result_CSV["level2"] = convertedMateix[:, 1]
    result_CSV["level3"] = convertedMateix[:, 2]
    result_CSV["level4"] = convertedMateix[:, 3]

    result_CSV = result_CSV[["cell_id", "level1", "level2", "level3", "level4"]]

    result_CSV.to_csv(os.path.join("expOutput", opt.expName, "submission_val.csv"), index=False)


#     print(result)


if __name__ == "__main__":
    main()
