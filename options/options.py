import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=140, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['derain', 'dehaze', 'desnow', 'derain_heavy', 'dehaze_heavy', 'desnow_heavy'],
                    help='which type of degradations is training and testing for.')
                    

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# paths
parser.add_argument('--derain_dir', type=str, default='/data/data_awracle/Train/Derain/',
                    help='training images of deraining.')
parser.add_argument('--dehaze_dir', type=str, default='/data/data_awracle/Train/Dehaze/',
                    help='training images of dehazing')
parser.add_argument('--desnow_dir', type=str, default='/data/data_awracle/Train/Desnow/',
                    help='training images of desnowing')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="", help='checkpoint load path for resuming')
parser.add_argument('--resume', action='store_true', default=False, help='Resume or not')
parser.add_argument("--wblogger",type=str,default="Smile",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="./train_ckpt/",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 2,help = "Number of GPUs to use for training")

options = parser.parse_args()

