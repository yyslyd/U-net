import argparse
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss, FocalLoss
from utils.segmentation import AMTOWN02_MASK_VALUES
from utils.splits import (create_split_ids, create_split_manifest, dataset_file_records,
                          file_sha256, save_split_manifest)

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


class AugmentedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, augment=True):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        if not self.augment:
            return item
        import torchvision.transforms.functional as TF
        img, mask = item['image'], item['mask']
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, [1, 2])
            mask = torch.rot90(mask, k, [0, 1])
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, 0.8 + random.random() * 0.4)
        if random.random() > 0.5:
            img = TF.adjust_contrast(img, 0.8 + random.random() * 0.4)
        return {'image': img, 'mask': mask}


def build_data_loaders(dataset, train_indices, val_indices, batch_size, seed, augment_train=False):
    train_subset = Subset(dataset, train_indices)
    if augment_train:
        train_subset = AugmentedSubset(train_subset, augment=True)
    val_subset = Subset(dataset, val_indices)
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_subset, shuffle=True, generator=torch.Generator().manual_seed(seed), **loader_args
    )
    val_loader = DataLoader(val_subset, shuffle=False, drop_last=False, **loader_args)
    return train_loader, val_loader


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        images_dir: Path = dir_img,
        masks_dir: Path = dir_mask,
        checkpoint_dir: Path = dir_checkpoint,
        seed: int = 0,
        use_wandb: bool = False,
):
    random.seed(seed)
    torch.manual_seed(seed)
    # 1. Create dataset (with augmentation for train, without for val)
    dataset = BasicDataset(images_dir, masks_dir, img_scale, augment=False,
                           mask_values=AMTOWN02_MASK_VALUES)
    if model.n_classes != len(dataset.mask_values):
        raise ValueError(
            f'Model has {model.n_classes} classes but AMtown02 requires {len(dataset.mask_values)}'
        )

    # 2. Split into train / validation partitions
    train_ids, val_ids = create_split_ids(dataset.ids, val_percent, seed)
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.ids)}
    train_indices = [index_by_id[sample_id] for sample_id in train_ids]
    val_indices = [index_by_id[sample_id] for sample_id in val_ids]
    n_train, n_val = len(train_indices), len(val_indices)
    file_records = dataset_file_records(dataset)

    # 3. Create data loaders
    train_loader, val_loader = build_data_loaders(
        dataset, train_indices, val_indices, batch_size, seed, augment_train=True
    )

    # (Initialize logging)
    if use_wandb and HAS_WANDB:
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
        )
    else:
        experiment = None
        if use_wandb and not HAS_WANDB:
            logging.warning('Weights & Biases requested but wandb is not installed; continuing without it')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Split seed:      {seed}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = FocalLoss(gamma=2.0) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    logging.info(f'Using Focal Loss (gamma=2.0) + Dice Loss + CosineAnnealingLR')
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if experiment is not None:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        if experiment is not None:
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        if experiment is not None:
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    **histograms
                                })
                            except:
                                pass

        # Step cosine scheduler per epoch
        scheduler.step()

        if save_checkpoint:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch{epoch}.pth'
            torch.save(state_dict, str(checkpoint_path))
            manifest = create_split_manifest(
                dataset.ids, val_percent, seed, checkpoint_path.name, file_sha256(checkpoint_path),
                dataset.mask_values, img_scale, model.bilinear, dataset_files=file_records,
            )
            if manifest['train_ids'] != train_ids or manifest['val_ids'] != val_ids:
                raise RuntimeError('Saved split manifest does not match the training split')
            save_split_manifest(checkpoint_path.with_suffix('.split.json'), manifest)
            logging.info(f'Checkpoint {epoch} saved! (lr={optimizer.param_groups[0]["lr"]:.2e})')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=Path,
                        help='Initialize model weights from a checkpoint; starts a new run and saved split')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=15, help='Number of classes (AMtown02: 15)')
    parser.add_argument('--images-dir', type=Path, default=dir_img, help='Training image directory')
    parser.add_argument('--masks-dir', type=Path, default=dir_mask, help='Training mask directory')
    parser.add_argument('--checkpoint-dir', type=Path, default=dir_checkpoint,
                        help='Directory for checkpoints and split sidecars')
    parser.add_argument('--seed', type=int, default=0, help='Dataset split and random augmentation seed')
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument('--wandb', dest='use_wandb', action='store_true',
                             help='Enable Weights & Biases logging')
    wandb_group.add_argument('--no-wandb', dest='use_wandb', action='store_false',
                             help='Disable Weights & Biases logging (default)')
    parser.set_defaults(use_wandb=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device, weights_only=True)
        mask_values = state_dict.pop('mask_values', None)
        if mask_values != list(AMTOWN02_MASK_VALUES):
            raise ValueError(
                f'Initial checkpoint mask_values must equal {list(AMTOWN02_MASK_VALUES)}, '
                f'got {mask_values}'
            )
        model.load_state_dict(state_dict)
        logging.info(f'Model weights initialized from {args.load}; creating a new run and split')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            use_wandb=args.use_wandb,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            use_wandb=args.use_wandb,
        )
