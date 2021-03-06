import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")

parser.add_argument("--save_path", type=str, default=r"/home/share/Han/novatek")
#parser.add_argument("--save_name", type=str, default=r'fingerNet_kiaraNoise_v1.0')
parser.add_argument("--save_annotation", type=str, default='')
parser.add_argument("--dataset_path", type=str, default="")
parser.add_argument("--train_path", type=str, default="")
parser.add_argument("--val_path", type=str, default="")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument('--model', type=str, default='fingerNet_v2')
parser.add_argument('--model_name', type=str, default='myModel')
parser.add_argument('--dataset', type=str, default='kiaraNoise')
parser.add_argument('--datasetfilename', type=str, default='dataset')
parser.add_argument("--singleInOut", action='store_true')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument("--input_shape", type=int, default=50)
parser.add_argument("--debug", action="store_true")

parser.add_argument("--losses", action="append" )
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta_1', type=float, default=0.9)
parser.add_argument('--beta_2', type=float, default=0.999)

parser.add_argument('--print_rate', type=int, default=100)
#=======================for perceptual loss===========================================

parser.add_argument('--lambda_tv', '-l_tv', default=10e-4, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', '-l_feat', default=1e0, type=float)
parser.add_argument('--lambda_style', '-l_style', default=1e1, type=float)
#=========================
'''
parser.add_argument('dataset', type=str)

parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--special_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll',"MSELoss","BCELoss", "CrossEntropyLoss"])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')

parser.add_argument('--concat', type = str, default = "", choices = ['All', 'First'], help = 'use concatenation after shifting')
parser.add_argument('--data_fuse', default = False, action = "store_true", help = 'concatenate skeleton to depth')
parser.add_argument('--extra_temporal_modeling', default=False, action="store_true", help = 'subtract the feature map')

parser.add_argument('--prune', type = str, default = "", choices = ["input", "output", "inout"], help='use prune for models')


parser.add_argument('--clipnums', type = str, default = "", help='numbers of clips')
parser.add_argument('--decoder_resume', default='', type=str, metavar='PATH',
                    help='path to latest decoder checkpoint (default: none)')
parser.add_argument("--debug", action="store_true")
'''