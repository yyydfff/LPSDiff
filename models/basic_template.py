import os
import os.path as osp
import argparse
import torch
import tqdm as tqdm_module
from tqdm import tqdm

from utils.dataset import dataset_dict
from utils.loggerx import LoggerX

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TrainTask(object):
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggerX(save_root=osp.join(
            osp.dirname(osp.dirname(osp.abspath(__file__))),
            'output',
            '{}_{}'.format(opt.model_name, opt.run_name)
        ))

        self.rank = 0

        if hasattr(opt, 'gpu_id') and opt.gpu_id is not None:
            torch.cuda.set_device(opt.gpu_id)
            self.device = torch.device(f'cuda:{opt.gpu_id}')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.set_loader()
        self.set_model()

        self.pbar = None
        self.console_log_every = 0

    @staticmethod
    def build_default_options():
        parser = argparse.ArgumentParser('Default arguments for training of different methods')
        parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID to use (default: 0)')
        parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
        parser.add_argument('--test_freq', type=int, default=1, help='validation frequency in epochs')
        parser.add_argument('--save_freq', type=int, default=1, help='save frequency (epochs)')
        parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
        parser.add_argument('--test_batch_size', type=int, default=1, help='validation/test batch_size')
        parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
        parser.add_argument('--max_iter', type=int, default=540001, help='number of training iterations')
        parser.add_argument('--resume_iter', type=int, default=0, help='resume from iter')
        parser.add_argument('--test_iter', type=int, default=540000, help='number of epochs for test')
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'train_osl_framework', 'test_osl_framework'])
        parser.add_argument('--wandb', action='store_true')

        parser.add_argument('--run_name', type=str, default='default', help='each run name')
        parser.add_argument('--model_name', type=str, default='LPSDiff', help='the type of method')

        parser.add_argument('--osl_max_iter', type=int, default=3001)
        parser.add_argument('--osl_batch_size', type=int, default=8)
        parser.add_argument('--index', type=int, default=10)
        parser.add_argument('--unpair', action='store_true')
        parser.add_argument('--patch_size', type=int, default=256)

        parser.add_argument('--train_dataset', type=str, default='mayo_2016')
        parser.add_argument('--val_dataset', type=str, default='val')
        parser.add_argument('--test_dataset', type=str, default='mayo_2016')
        parser.add_argument('--test_id', type=int, default=9, help='test patient index for Mayo 2016')


        parser.add_argument('--context', dest='context', action='store_true', default=True, help='use contextual information')
        parser.add_argument('--no_context', dest='context', action='store_false', help='disable contextual information')

        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--dose', type=int, default=25, help='dose percentage')

        parser.add_argument('--console_log_every', type=int, default=100)
        parser.add_argument('--progress_mode', choices=['epoch', 'batch'], default='epoch',
                            help='epoch: one progress bar per epoch; batch: progress by batch')
        parser.add_argument('--warmup_epochs', type=int, default=2, help='linear warmup epochs before cosine')

        return parser

    @staticmethod
    def build_options():
        pass

    def set_loader(self):

        opt = self.opt

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_images = None
        self.test_dataset = None
        self.val_dataset = None

        if opt.mode == 'train':
            train_dataset = dataset_dict['train'](
                dataset=opt.train_dataset,
                test_id=opt.test_id,
                dose=opt.dose,
                context=opt.context,
            )
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            if opt.val_dataset not in dataset_dict:
                raise KeyError(f"val_dataset='{opt.val_dataset}' is not in dataset_dict. Available keys: {list(dataset_dict.keys())}")

            val_dataset = dataset_dict[opt.val_dataset](
                dataset=opt.train_dataset,
                test_id=opt.test_id,
                dose=opt.dose,
                context=opt.context,
            )
            self.val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=opt.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            self.val_dataset = val_dataset

            print(f"[Loader] train samples: {len(train_dataset)}")
            print(f"[Loader] val samples: {len(val_dataset)}")
            print("[Loader] train mode: test_loader is NOT created.")

        elif opt.mode == 'test':
            if opt.test_dataset not in dataset_dict:
                raise KeyError(f"test_dataset='{opt.test_dataset}' is not in dataset_dict. Available keys: {list(dataset_dict.keys())}")

            test_dataset = dataset_dict[opt.test_dataset](
                dataset=opt.test_dataset,
                test_id=opt.test_id,
                dose=opt.dose,
                context=opt.context,
            )
            self.test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=opt.test_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=opt.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            self.test_dataset = test_dataset
            print(f"[Loader] test samples: {len(test_dataset)}")

    def set_model(self):
        raise NotImplementedError

    def train(self, inputs, n_iter):
        raise NotImplementedError

    @torch.no_grad()
    def eval_step(self, batch):
        return 0.0

    @torch.no_grad()
    def validate(self, n_iter, epoch=None):
        raise NotImplementedError

    @torch.no_grad()
    def test(self, n_iter, epoch=None):
        pass

    def adjust_learning_rate(self, n_iter):
        return

    def _parse_train_ret(self, ret):
        loss = None
        lr = None
        if isinstance(ret, dict):
            loss = ret.get('loss', None)
            lr = ret.get('lr', None)
        elif isinstance(ret, (list, tuple)) and len(ret) >= 1:
            loss = ret[0]
            if len(ret) >= 2:
                lr = ret[1]
        return loss, lr

    def fit(self):
        opt = self.opt

        if opt.mode == 'train':
            n_iter = getattr(opt, 'resume_iter', 0)

            for epoch in range(opt.epochs):
                self.model.train()
                pbar = tqdm(
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}/{opt.epochs}",
                    leave=True,
                    dynamic_ncols=True
                )

                for batch_idx, inputs in enumerate(self.train_loader):
                    n_iter += 1
                    ret = self.train(inputs, n_iter)

                    cur_lr = float(ret.get('lr', self.optimizer.param_groups[0]['lr'])) if isinstance(ret, dict) else self.optimizer.param_groups[0]['lr']
                    cur_loss = float(ret.get('loss', 0.0)) if isinstance(ret, dict) else 0.0

                    pbar.set_postfix(lr=self._format_lr(cur_lr, 6), loss=self._format_lr(cur_loss, 6))
                    pbar.update(1)

                pbar.close()

                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Validate once after each epoch by default. This never touches test_loader.
                if (epoch + 1) % max(1, opt.test_freq) == 0:
                    self.validate(n_iter=n_iter, epoch=epoch)

        elif opt.mode == 'test':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.test(opt.test_iter, epoch=None)

        elif opt.mode == 'train_osl_framework':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.train_osl_framework(opt.test_iter)

        elif opt.mode == 'test_osl_framework':
            self.logger.load_test_checkpoints(opt.test_iter)
            self.test_osl_framework(opt.test_iter)

    def _reset_bar(self, total, desc):
        total = int(total)
        if total <= 0:
            total = 1

        if self.pbar is None:
            self.pbar = tqdm_module.tqdm(total=total, desc=desc, position=0, leave=True, dynamic_ncols=True)
        else:
            self.pbar.reset(total=total)
            self.pbar.set_description(desc)

        self.pbar.miniters = 1
        self.pbar.mininterval = 0
        self.pbar.smoothing = 0.0

    @staticmethod
    def _to_float(x):
        try:
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item())
            return float(x)
        except Exception:
            return None

    def transfer_calculate_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-1000, cut_max=1000):

        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = 255 * (img - cut_min) / (cut_max - cut_min)
        return img

    def transfer_display_window(self, img, MIN_B=-1024, MAX_B=3072, cut_min=-100, cut_max=200):

        img = img * (MAX_B - MIN_B) + MIN_B
        img[img < cut_min] = cut_min
        img[img > cut_max] = cut_max
        img = (img - cut_min) / (cut_max - cut_min)
        return img

    @staticmethod
    def _format_lr(x, decimals: int = 6, strip: bool = True) -> str:
        try:
            s = f"{float(x):.{decimals}f}"
            return s.rstrip('0').rstrip('.') if strip else s
        except Exception:
            return str(x)
