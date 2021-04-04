# -*- coding: utf-8 -*-
import torch
from torch import nn, optim
import torchvision
from torchvision import models
import os
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

from OriginalDataset import OriginalDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 適当な数字で設定すればいいらしいがよくわかっていない

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 正誤確認関数(正解:ans=1, 不正解:ans=0)
def eval_ans(y_hat, label):
    true_label = int(label)
    if(y_hat == true_label):
        ans = 1
    else:
        ans = 0
    return ans

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def train(model, rank, loss_fn, optimizer, train_loader):
    model.train()
    train_class_loss = 0.0
    correct_num = 0

    for (input_tensor, class_label) in train_loader:
        # Bagとbatchのギャップを吸収して学習
        for bag_num in range(input_tensor.shape[0]):
            bag = input_tensor[bag_num].to(rank, non_blocking=True)
            bag_class_label = class_label[bag_num].to(rank, non_blocking=True)
            class_prob, class_hat = model(bag)
            class_loss = loss_fn(class_prob, bag_class_label)
            train_class_loss += class_loss.item()
            optimizer.zero_grad()
            class_loss.backward()
            optimizer.step() 
            correct_num += eval_ans(class_hat, bag_class_label)

    train_class_loss = train_class_loss / len(train_loader)
    train_acc = correct_num / len(train_loader)

    return train_class_loss, train_acc


def valid(model, rank, loss_fn, valid_loader):
    model.eval()
    valid_class_loss = 0.0
    correct_num = 0
    for (input_tensor, class_label) in valid_loader:
        # Bagとbatchのギャップを吸収して学習
        for bag_num in range(input_tensor.shape[0]):
            bag = input_tensor[bag_num].to(rank, non_blocking=True)
            bag_class_label = class_label[bag_num].to(rank, non_blocking=True)
            with torch.no_grad():
                class_prob, class_hat = model(bag)
            class_loss = loss_fn(class_prob, bag_class_label)
            valid_class_loss += class_loss.item()
            correct_num += eval_ans(class_hat, bag_class_label)

    valid_class_loss = valid_class_loss / len(valid_loader)
    valid_acc = correct_num / len(valid_loader)

    return valid_class_loss, valid_acc


# マルチプロセス (GPU) で実行される関数
# rank : mp.spawnで呼び出すと勝手に追加される引数で, GPUが割り当てられている
# world_size : mp.spawnの引数num_gpuに相当
def train_model(rank, world_size, train_slide, valid_slide):
    setup(rank, world_size)

    EPOCHS = 20
    
    # 訓練用と検証用に症例を分割
    import dataset as ds
    train_A, train_B, train_C, valid_A, valid_B, valid_C = ds.slide_split(train_slide, valid_slide)

    # 訓練slideにクラスラベル付与
    train_dataset = []
    for slideID in train_A:
        train_dataset.append([slideID, 0])
    for slideID in train_B:
        train_dataset.append([slideID, 1])
    for slideID in train_C:
        train_dataset.append([slideID, 2])

    valid_dataset = []
    for slideID in valid_A:
        valid_dataset.append([slideID, 0])
    for slideID in valid_B:
        valid_dataset.append([slideID, 1])
    for slideID in valid_C:
        valid_dataset.append([slideID, 2])

    torch.backends.cudnn.benchmark=True #cudnnベンチマークモード

    # 出力先
    root = f'./outputs/train_{train_slide}'

    if rank == 0:
        if not os.path.exists(root):
            os.mkdir(root)
        with open(f'{root}/train_result.txt', mode='w') as f:
                f.write(
                    'train_result_file' + '\n'
                )
        with open(f'{root}/valid_result.txt', mode='w') as f:
                f.write(
                    'valid_result_file' + '\n'
                )

    # model読み込み
    from MIL import feature_extractor, class_predictor, MIL

    # 特徴抽出器の作成 (今回はResNet50を使用)
    encoder = models.resnet50(pretrained=True)
    encoder.fc = nn.Identity()
    for p in encoder.parameters():
        p.required_grad = True

    # MILモデルの構築
    feature_ex = feature_extractor(encoder)
    class_pred = class_predictor()
    model = MIL(feature_ex, class_pred)
    model = model.to(rank)
    process_group = torch.distributed.new_group([i for i in range(world_size)])
    # modelのBatchNormをSyncBatchNormに変更してくれる
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    # modelをmulti GPU対応させる
    ddp_model = DDP(model, device_ids=[rank])

    # クロスエントロピー損失関数使用
    loss_fn = nn.CrossEntropyLoss()
    # SGDmomentum法使用
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    # 前処理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    
    train_loss = []
    train_acc  = []

    valid_loss = []
    valid_acc  = []

    # Train
    for epoch in range(EPOCHS):
        # Train bag作成(epochごとにbag再構築)
        original_dataset = OriginalDataset(
            transform=transform,
            dataset=train_dataset,
            train=True
        )

        # Datasetをmulti GPU対応させる
        deta_sampler = torch.utils.data.distributed.DistributedSampler(original_dataset, rank=rank)

        # batch_sizeで設定したbatch_sizeで各GPUに分配
        train_loader = torch.utils.data.DataLoader(
            original_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=2,
            sampler=deta_sampler
        )

        # 学習
        class_loss, acc = train(ddp_model, rank, loss_fn, optimizer, train_loader)
        train_loss.append(class_loss)
        train_acc.append(acc)
        
        with open(f'{root}/train_result.txt', mode='a') as f:
            f.write(
                'epoch='  + str(epoch) + ', ' +
                'rank='   + str(rank) + ', ' +
                'loss='   + str(class_loss)  + ', ' +
                'acc='    + str(acc) + '\n'
            )

        # Valid
        # Valid bag作成(epochごとにbag再構築)
        original_dataset = OriginalDataset(
            transform=transform,
            dataset=valid_dataset,
            train=False
        )

        # Datasetをmulti GPU対応させる
        deta_sampler = torch.utils.data.distributed.DistributedSampler(original_dataset, rank=rank)

        # batch_sizeで設定したbatch_sizeで各GPUに分配
        valid_loader = torch.utils.data.DataLoader(
            original_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            num_workers=2,
            sampler=deta_sampler
        )
        class_loss, acc = valid(ddp_model, rank, loss_fn, valid_loader)
        valid_loss.append(class_loss)
        valid_acc.append(acc)

        with open(f'{root}/valid_result.txt', mode='a') as f:
            f.write(
                'epoch='  + str(epoch) + ', ' +
                'rank='   + str(rank) + ', ' +
                'loss='   + str(class_loss)  + ', ' +
                'acc='    + str(acc)  + '\n'
            )

            model_file = f'./model/train_{train_slide}/{epoch+1}.wgt'
            Path(model_file).parent.mkdir(parents=True, exist_ok=True)
            torch.save(ddp_model.module.state_dict(), model_file)


if __name__ == '__main__':

    num_gpu = 8 # GPU数
    args = sys.argv
    train_slide = args[1]
    valid_slide = args[2]

    # マルチプロセスで実行するために呼び出す
    # train_model : マルチプロセスで実行する関数
    # args : train_modelの引数
    # nprocs : プロセス (GPU) の数
    mp.spawn(train_model, args=(num_gpu, train_slide, valid_slide), nprocs=num_gpu, join=True)
