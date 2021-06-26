import os
import csv
import time
import torch
import argparse
import numpy as np
from data.dataset import HandDataset
from resnet.resnet import resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2
from utils.utils import pose_pck_accuracy, save_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Hand Keypoint Training")
    parser.add_argument('--data_dir', type=str, default='/data/yx/hand_db',
                        help='Directory containing data for training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='total batch size for all GPUs')
    parser.add_argument('--workers',
                        type=int,
                        default=16,
                        help='number of data loading workers')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=5e-4,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.9,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2',
                        type=float,
                        default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='Weight decay for optimizer')
    args = parser.parse_args()


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    print("=> creating model")
    best_pck = 0
    gpu = 0
    model = resnet50(filter_size=5, num_classes=42)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = model.cuda()
    model.load_state_dict(torch.load("resnet/models/resnet50.pth"), strict=False)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 args.lr,
                                 betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[args.epochs // 4, args.epochs // 2],
                                                     gamma=0.5)
    torch.backends.cudnn.benchmark = True
    train_dataset = HandDataset(args.data_dir, training=True)
    train_dataset.add_anno_file(os.path.join(args.data_dir, "onehand10k/onehand10k_train.json"))
    train_dataset.add_anno_file(os.path.join(args.data_dir, "coco/coco_hand_train.json"))
    train_dataset.add_anno_file(os.path.join(args.data_dir, "coco/coco_hand_val.json"))
    # train_dataset.add_anno_file()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_dataset = HandDataset(args.data_dir, training=False)
    val_dataset.add_anno_file(os.path.join(args.data_dir, "onehand10k/onehand10k_test.json"))
    # val_dataset.add_anno_file()
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    for epoch in range(args.epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']
        print("Epoch: {} / {}, LR: {:.5f}".format(epoch + 1, args.epochs, lr))
        # train for one epoch
        train_pck = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_pck = validate(val_loader, model, criterion)
        is_best = best_pck < val_pck
        if is_best:
            best_pck = val_pck
        scheduler.step()
        epoch_end = time.time()
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict()}, is_best)
        elapsed_time = (epoch_end - epoch_start) / 60.
        print("Epoch: {} finish, Elapsed Time: {} minutes, Pck_train:{:.4f}, Pck_val:{:.4f}".format(epoch + 1, int(
            round(elapsed_time)), train_pck, val_pck))


def train(train_loader, model, criterion, optimizer):
    # switch to train mode
    model.train()
    preds = []
    kpts = []
    normalizes = []
    for i, items in enumerate(train_loader):
        # measure data loading time
        image = items["image"].cuda()
        target = items["warp_kpt"].cuda()
        weight = items["weight"].cuda()
        kpt = items["kpt"]
        meta = items["meta"]
        normalize = items["normalize"]
        weight = weight

        # compute output
        output = model(image)
        loss = criterion(output * weight, target[..., :2] * weight)
        # pred, maxval = keypoints_from_heatmaps(output.detach().cpu().numpy(),
        #                                        meta.numpy())
        # preds.append(torch.from_numpy(pred).float())
        preds.append(output.detach())
        kpts.append(kpt)
        normalizes.append(normalize)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print("Iter: {} / {}, Loss: {:.5f}".format(i + 1, len(train_loader), loss.item()))

    preds = torch.cat(preds, 0)
    kpts = torch.cat(kpts, 0)
    normalizes = torch.cat(normalizes, 0)
    _, avg_pck, _ = pose_pck_accuracy(preds.cpu().numpy(),
                                      kpts.numpy(),
                                      normalizes.numpy(),
                                      thr=0.2)
    return avg_pck


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    preds = []
    kpts = []
    normalizes = []
    with torch.no_grad():
        for i, items in enumerate(val_loader):
            image = items["image"].cuda()
            target = items["warp_kpt"].cuda()
            weight = items["weight"].cuda()
            kpt = items["kpt"]
            meta = items["meta"]
            normalize = items["normalize"]
            weight = weight

            # compute output
            output = model(image)
            # loss = criterion(output * weight, target[..., :2] * weight)
            # pred, maxval = keypoints_from_heatmaps(output.cpu().numpy(),
            #                                        meta.numpy())
            # preds.append(torch.from_numpy(pred).float())
            preds.append(output.detach())
            kpts.append(kpt)
            normalizes.append(normalize)
            # if (i + 1) % 10 == 0:
            #     print("Iter: {} / {}, Loss: {:.5f}".format(i + 1, len(val_loader), loss.item()))

        preds = torch.cat(preds, 0)
        kpts = torch.cat(kpts, 0)
        normalizes = torch.cat(normalizes, 0)
        _, avg_pck, _ = pose_pck_accuracy(preds.cpu().numpy(),
                                          kpts.numpy(),
                                          normalizes.numpy(),
                                          thr=0.2)
    return avg_pck


if __name__ == "__main__":
    main()
