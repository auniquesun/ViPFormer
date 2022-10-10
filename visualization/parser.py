import argparse


# common settings for pretraining CrossPoint
parser = argparse.ArgumentParser(description='General Attention for Point Cloud Understanding')

parser.add_argument('--proj_name', type=str, default='Perceiver', metavar='N',
                    help='Name of the project')
parser.add_argument('--exp_name', type=str, default='1CL1SB6SL_4', metavar='N',
                    help='Name of the experiment')

parser.add_argument('--main_program', type=str, default='main.py', metavar='N',
                    help='Name of main program')
parser.add_argument('--model_name', type=str, default='modules.py', metavar='N',
                    help='Name of model file')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--modality', type=str, default='both', metavar='N',
                    choices=['cmc-only', 'imc-only', 'both'], 
                    help='which modality to use, cmc: cross-modal, imc: intra-modal')

parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--pc_model_file', type=str, default='pc_best_model.pth', metavar='N',
                    help='saved point model name')
parser.add_argument('--img_model_file', type=str, default='img_best_model.pth', metavar='N',
                    help='saved image model name')

parser.add_argument('--eval', action='store_true',  help='evaluate the model')

parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch')
parser.add_argument('--test_batch_size', type=int, default=160, metavar='test_batch_size',
                    help='Size of test batch')
parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                    help='number of processes to load data in host memory')

# optimizer and learning schedule
parser.add_argument('--optim', type=str, default='sgd', metavar='N',
                    choices=['sgd', 'adam', 'adamw'],
                    help='optimizer to choose')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--max_lr', type=float, default=0.1, metavar='LR',
                    help='maximum learning rate')
parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                    help='minimum learning rate')
parser.add_argument('--warm_epochs', type=int, default=10,
                    help='number of iterations for the first restart')
parser.add_argument('--factor', type=float, default=0.1, 
                    help='factor by which the learning rate will be reduced. new_lr = lr * factor')
parser.add_argument('--patience', type=int, default=10,
                    help='number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument('--step_size', type=int, default=30,
                    help='adapt learning rate every step_size epochs during training')
parser.add_argument('--gamma', type=float, default=0.1, 
                    help='lr = lr * gamma, every step_size epochs')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'coswarm', 'plateau', 'step'],
                    help='Scheduler to use, [cos, step, coswarm, plateau]')

parser.add_argument('--num_pt_points', type=int, default=2048,
                    help='Number of points when pretraining')
parser.add_argument('--num_ft_points', type=int, default=1024,
                    help='Number of points when finetuning')
parser.add_argument('--num_test_points', type=int, default=1024,
                    help='Number of points when svm test')
parser.add_argument('--point_channels', type=int, default=3,
                    help='the dimension of the feature of a point')

# --------- Model specifics
parser.add_argument('--num_groups', type=int, default=128,
                    help='Number of groups to divide')
parser.add_argument('--group_size', type=int, default=32,
                    help='Number of nearest neighbors to use')

parser.add_argument('--num_pc_latents', type=int, default=128,
                    help='array length of latent point cloud')
parser.add_argument('--num_img_latents', type=int, default=128,
                    help='array length of latent image')
parser.add_argument('--num_latent_channels', type=int, default=256,
                    help='channels of element in latent array')

parser.add_argument('--num_ca_layers', type=int, default=1, metavar='N', help='Number of cross attention layers')
parser.add_argument('--num_ca_heads', type=int, default=4, metavar='N', help='Number of heads in cross attention layer')
parser.add_argument('--num_sa_layers_per_block', type=int, default=6, metavar='N', help='Number of layers in each block')
parser.add_argument('--num_sa_blocks', type=int, default=1, metavar='N', help='Number of self attention blocks')
parser.add_argument('--num_sa_layers', type=int, default=1, metavar='N', help='Number of self attention layers')
parser.add_argument('--num_sa_heads', type=int, default=4,
                    help='number of heads in self attention')

parser.add_argument('--mlp_widen_factor', type=int, default=2, metavar='N',
                    help='dimension factor of hidden layer in MLP')
parser.add_argument('--max_dpr', type=float, default=0.5,
                    help='max drop path rate')
parser.add_argument('--atten_drop', type=float, default=0.1,
                    help='dropout rate in Attention')
parser.add_argument('--mlp_drop', type=float, default=0.5,
                    help='dropout rate in MLP')
parser.add_argument('--layer_idx', nargs='+', type=int, help='Specify the layers whose feature will be extracted')

parser.add_argument('--cmid_weight', type=float, default=1.0,
                    help='weight of loss_cmid')

parser.add_argument('--img_height', type=int, default=224, help='input image height')
parser.add_argument('--img_width', type=int, default=224, help='input image width')
# ---------

parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')

# training on single GPU device 
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu_id', type=int, default=0, help='specify the GPU device'
                    'to train of finetune model')

# distributed training on multiple GPUs
parser.add_argument('--rank', type=int, default=-1, help='the rank for current GPU or process, '
                    'ususally one process per GPU')
parser.add_argument('--backend', type=str, default='nccl', help='DDP communication backend')
parser.add_argument('--world_size', type=int, default=6, help='number of GPUs')
parser.add_argument('--master_addr', type=str, default='localhost', help='ip of master node')
parser.add_argument('--master_port', type=str, default='12355', help='port of master node')

parser.add_argument('--pt_dataset', type=str, default='ModelNet40', help='the dataset used for '
                    'evaluating the pretrained model')
parser.add_argument('--ft_dataset', type=str, default='ModelNet40', help='finetune dataset')
parser.add_argument('--num_obj_classes', type=int, default=40, help='number of object classes')
parser.add_argument('--num_part_classes', type=int, default=50, help='number of object part classes')
parser.add_argument('--output_seq_length', type=int, default=1, help='output sequence length')

# downstream task: Segmentation settings
parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                    choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
parser.add_argument('--svm_coff', type=float, default=0.1, help='linear_svm coefficient')

# few-shot learning
parser.add_argument('--n_runs', type=int, default=10,
                        help='Num of few-shot runs')
parser.add_argument('--k_way', type=int, default=5,
                        help='Num of classes in few-shot')
parser.add_argument('--n_shot', type=int, default=10,
                        help='Num of samples in one class')
parser.add_argument('--n_query', type=int, default=20,
                        help='Num of query samples in one class')

# visualization
parser.add_argument('--category', type=str, default='Airplane', help='select category')
parser.add_argument('--ballradius', type=int, default=10, help='ballradius')

# wandb settings
parser.add_argument('--wb_url', type=str, default="http://localhost:28282", help='wandb server url')
parser.add_argument('--wb_key', type=str, default="", help='find your login key from wandb web client')

args = parser.parse_args()
