import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from eval_utils import compute_original_accuracies
import argparse
import os
from training_utils import symmetric_cross_entropy, LinearSubspace, AverageMeter
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

class FeatureDataset(Dataset):
    def __init__(self, img_i_features, img_t_features, img_ti_features,
                 txt_i_features, txt_t_features, class_id, real_or_fake):
        self.img_i_features = img_i_features
        self.img_t_features = img_t_features
        self.img_ti_features = img_ti_features
        self.txt_i_features = txt_i_features
        self.txt_t_features = txt_t_features
        self.txt_i_features_tensor = torch.tensor(txt_i_features)
        self.real_or_fake = real_or_fake
        self.class_id = class_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.img_i_features)

    def __getitem__(self, idx):
        return (self.img_i_features[idx], self.img_t_features[idx],
                self.img_ti_features[idx], self.txt_i_features[self.class_id[idx]],
                self.txt_t_features[idx], self.class_id[idx], self.real_or_fake[idx])


def train(epoch, model, train_loader, scheduler):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    regularizer = AverageMeter('Weight norm', ':.4e')
    model.train()
    for i, batch in enumerate(train_loader):
        # compute output
        img_i, img_t, img_ti, txt_i, txt_t, class_idx_batch, _ = batch
        img_i, img_t, img_ti, txt_i, txt_t = model(img_i.to(device), img_t.to(device), img_ti.to(device), txt_i.to(device), txt_t.to(device), class_idx_batch.to(device))
        im_i_txt_i_loss = symmetric_cross_entropy(img_i, txt_i, device)
        im_t_txt_t_loss = symmetric_cross_entropy(img_t, txt_t,device)
        im_ti_txt_i_loss = symmetric_cross_entropy(img_ti, txt_i,device)
        im_ti_txt_t_loss = symmetric_cross_entropy(img_ti, txt_t,device)
        im_ti_img_i_loss = symmetric_cross_entropy(img_ti, img_i,device)
        im_ti_img_t_loss = symmetric_cross_entropy(img_ti, img_t,device)
        loss = args.l_im_i_txt_i * im_i_txt_i_loss + args.l_im_t_txt_t * im_t_txt_t_loss + args.l_im_ti_txt_i * im_ti_txt_i_loss + args.l_im_ti_txt_t * im_ti_txt_t_loss + args.l_im_ti_im_i * im_ti_img_i_loss + args.l_im_ti_im_t * im_ti_img_t_loss
        reg = model._weight_norm(device)
        losses.update(loss.item(), img_i.shape[0])
        regularizer.update(reg.item(), img_i.shape[0])

        wandb.log({'epoch': epoch, 'train_im_i_txt_i_loss': im_i_txt_i_loss, 'train_im_t_txt_t_loss': im_t_txt_t_loss,
                   'train_im_ti_txt_i_loss': im_ti_txt_i_loss, 'train_im_ti_txt_t_loss' : im_ti_txt_t_loss,
                   'train_im_ti_img_i_loss': im_ti_img_i_loss, 'train_im_ti_img_t_loss': im_ti_img_t_loss,
                   'train_regularizer': reg})

        total_loss = loss + args.reg*reg
        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if i % 2000 == 0:
            print(f'Train Epoch {epoch}, Step {i}, Loss: {loss}')
        scheduler.step()



def validate(model, val_loader, val_txt_i_features, U):
    model.eval()
    val_img_i_features, val_img_t_features, val_img_ti_features, val_txt_t_features = [], [], [], []
    class_idx = []
    real_or_fake = []
    for i, batch in enumerate(val_loader):
        img_i, img_t, img_ti, txt_i, txt_t, class_idx_batch, rf = batch
        img_i, img_t, img_ti, txt_i, txt_t = model(img_i.to(device), img_t.to(device), img_ti.to(device), txt_i.to(device), txt_t.to(device), class_idx_batch.to(device))

        im_i_txt_i_loss = symmetric_cross_entropy(img_i, txt_i, device)
        im_t_txt_t_loss = symmetric_cross_entropy(img_t, txt_t, device)
        im_ti_txt_i_loss = symmetric_cross_entropy(img_ti, txt_i,device)
        im_ti_txt_t_loss = symmetric_cross_entropy(img_ti, txt_t,device)
        im_ti_img_i_loss = symmetric_cross_entropy(img_ti, img_i,device)
        im_ti_img_t_loss = symmetric_cross_entropy(img_ti, img_t,device)

        wandb.log({'val_im_i_txt_i_loss': im_i_txt_i_loss, 'val_im_t_txt_t_loss': im_t_txt_t_loss,
                   'val_im_ti_txt_i_loss': im_ti_txt_i_loss, 'val_im_ti_txt_t_loss' : im_ti_txt_t_loss,
                   'val_im_ti_img_i_loss': im_ti_img_i_loss, 'val_im_ti_img_t_loss': im_ti_img_t_loss})


        val_img_i_features.append(img_i)
        val_img_t_features.append(img_t)
        val_img_ti_features.append(img_ti)
        val_txt_t_features.append(txt_t)
        class_idx.append(class_idx_batch)
        real_or_fake.append(rf)

    if True:
        files = {"image_with_text": torch.cat(val_img_ti_features),
                  "text_image": torch.cat(val_img_t_features),
                  'text_features': torch.cat(val_txt_t_features),
                  "text_class_features": val_txt_i_features.to(device)@model.W.t(),
                  'class_id': torch.cat(class_idx).to(device),
                  'real_or_fake': torch.cat(real_or_fake),
                  'image': torch.cat(val_img_i_features),}
        out_dict = compute_original_accuracies('val', files, U, suffix='')
        wandb.log(out_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', help='Part of experiment name [Learning/Forgetting]')
    parser.add_argument('--out_dim', type=int)
    parser.add_argument('--exp_info', type=str)
    parser.add_argument('--reg', type=float, default=0.)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--l_im_i_txt_i', type=float, default=0.0)
    parser.add_argument('--l_im_t_txt_t', type=float, default=0.0)
    parser.add_argument('--l_im_ti_txt_i', type=float, default=0.0)
    parser.add_argument('--l_im_ti_txt_t', type=float, default=0.0)
    parser.add_argument('--l_im_ti_im_i', type=float, default=0.0)
    parser.add_argument('--l_im_ti_im_t', type=float, default=0.0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--root', help='Root path of the data folder')
    parser.add_argument('--save_path', help='Path to folder to save the model checkpoint')



    args = parser.parse_args()
    print(args)
    SET = args.set
    out_dim = args.out_dim

    exp_name = f"{args.set}_spelling_{args.exp_info}_out_dim_{args.out_dim}_reg_{args.reg}"
    wandb.init(project="spelling", entity="joaanna")
    wandb.run.name = exp_name
    wandb.config.update(args)

    root = args.root

    train_ft = [np.load(xxx) for xxx in [f'{root}/train_img.npy',  f'{root}/train_text_img.npy',
                              f'{root}/train_with_text_img.npy', f'{root}/train_text_class.npy',
                              f'{root}/train_text.npy', f'{root}/train_class_id.npy',
                                    f'{root}/train_real_or_fake.npy']]
    val_set = 'val'
    val_ft = [np.load(xxx) for xxx in [f'{root}/{val_set}_img.npy',  f'{root}/{val_set}_text_img.npy',
                            f'{root}/{val_set}_with_text_img.npy',
                            f'{root}/{val_set}_text_class.npy', f'{root}/{val_set}_text.npy',
                                       f'{root}/{val_set}_class_id.npy', f'{root}/{val_set}_real_or_fake.npy']]
    U = len(np.unique(np.array([i[:-1] for i in open(f'{root}/{val_set}_text.txt', 'r').readlines()])[val_ft[-1]==1]))

    (train_img_i_features, train_img_t_features,
     train_img_ti_features, train_txt_i_features,
     train_txt_t_features, train_class_id, train_real_or_fake) = train_ft


    (val_img_i_features, val_img_t_features,
     val_img_ti_features, val_txt_i_features,
     val_txt_t_features, val_class_id, val_real_or_fake) = val_ft

    files = {"image_with_text": val_img_ti_features,
             "text_image": val_img_t_features,
             'text_features': val_txt_t_features,
             "text_class_features": val_txt_i_features,
             'class_id': val_class_id,
             'real_or_fake': val_real_or_fake,
             'image': val_img_i_features }


    batch_size = 128
    num_epochs = args.num_epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LinearSubspace(out_dim=out_dim).to(device)
    lr = 0.0001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=4000)
    # Log model training
    wandb.watch(model, log="all")




    train_dataset = FeatureDataset(train_img_i_features, train_img_t_features,
                       train_img_ti_features, train_txt_i_features,
                       train_txt_t_features, train_class_id, train_real_or_fake)
    val_dataset = FeatureDataset(val_img_i_features,
                                 val_img_t_features,
                       val_img_ti_features, val_txt_i_features,
                       val_txt_t_features, val_class_id, val_real_or_fake)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4)
    print(f'Number of training examples : {len(train_dataset)} \n', f'Number of validation examples : {len(val_dataset)} \n')

    if args.test:
        model.load_state_dict(
            torch.load(f'{args.save_path}/{exp_name}.pth'))

        validate(model, val_loader, torch.tensor(val_txt_i_features), U)
        sys.exit()


    for epoch in range(num_epochs):
        train(epoch, model, train_loader, scheduler)
        validate(model, val_loader, torch.tensor(val_txt_i_features), U)

    os.makedirs(f'{args.save_path}/', exist_ok=True)
    torch.save(model.state_dict(), f'{args.save_path}/{exp_name}.pth')

