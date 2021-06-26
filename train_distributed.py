import os
import csv
import time
import torch
import argparse
import torch.distributed as dist

from data.dataset import HandDataset
from models.defaults import _C as cfg
from models.hrnet import HighResolutionNet
from utils.transform import keypoints_from_heatmaps
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
                        default=160,
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
    world_size = int(
        os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    main_worker(args.local_rank, world_size, args)


def main_worker(gpu, ngpus_per_node, args):
    dist.init_process_group(backend='nccl')
    if args.local_rank == 0:
        print("=> creating model")
    model = HighResolutionNet(cfg)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    args.batch_size = int(args.batch_size / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss().cuda(gpu)
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
    # train_dataset.add_anno_file()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
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
        train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            print("Epoch: {} / {}, LR: {:.5f}".format(epoch + 1, args.epochs, lr))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, gpu)
        # evaluate on validation set
        validate(val_loader, model, criterion, gpu)
        scheduler.step()
        epoch_end = time.time()
        if args.local_rank == 0:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.module.state_dict()}, str(epoch + 1) + "_state.pth")
            print("Epoch: {} finish, Elapsed Time: {}".format(epoch + 1, epoch_end - epoch_start))


def train(train_loader, model, criterion, optimizer, epoch, gpu):
    # switch to train mode
    model.train()
    for i, items in enumerate(train_loader):
        # measure data loading time
        images = items["image"].cuda(gpu, non_blocking=True)
        targets = items["target"].cuda(gpu, non_blocking=True)
        kpts = items["kpt"].cuda(gpu, non_blocking=True)
        metas = items["meta"].cuda(gpu, non_blocking=True)
        normalizes = items["normalize"].cuda(gpu, non_blocking=True)
        weights = items["weight"].cuda(gpu, non_blocking=True)
        weights = weights.unsqueeze(-1)

        # compute output
        output = model(images)
        loss = criterion(output * weights, targets * weights)
        preds, maxvals = keypoints_from_heatmaps(output.detach().cpu().numpy(),
                                                 metas.cpu().numpy())
        _, avg_pck, _ = pose_pck_accuracy(preds,
                                          kpts.cpu().numpy(),
                                          normalizes.cpu().numpy(),
                                          thr=0.2)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1 == 0 and args.local_rank == 0:
            print("Iter: {} / {}, Loss: {:.5f}, Pck: {:.4f}".format(i + 1, len(train_loader), loss.item(), avg_pck))


def validate(val_loader, model, criterion, gpu):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, items in enumerate(val_loader):
            images = items["image"].cuda(gpu, non_blocking=True)
            targets = items["target"].cuda(gpu, non_blocking=True)
            kpts = items["kpt"].cuda(gpu, non_blocking=True)
            metas = items["meta"].cuda(gpu, non_blocking=True)
            normalizes = items["normalize"].cuda(gpu, non_blocking=True)
            weights = items["weight"].cuda(gpu, non_blocking=True)
            weights = weights.unsqueeze(-1)

            # compute output
            output = model(images)
            loss = criterion(output * weights, targets * weights)
            preds, maxvals = keypoints_from_heatmaps(output.cpu().numpy(),
                                                     metas.cpu().numpy())

            _, avg_pck, _ = pose_pck_accuracy(preds,
                                              kpts.cpu().numpy(),
                                              normalizes.cpu().numpy(),
                                              thr=0.2)
            if args.local_rank == 0:
                print("Iter: {} / {}, Loss: {:.5f}, Pck: {:.4f}".format(i + 1, len(val_loader), loss.item(), avg_pck))


if __name__ == "__main__":
    main()
