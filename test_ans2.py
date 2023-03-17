import argparse
import os
import torch.optim as optim
import torch.utils.data as util_data
import itertools
import network
import pre_process as prep
from util3 import *
from data_list1 import ImageList
from Graph import *
import vision_transformer as vits

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}


def main(config,num):
    ## set loss criterion
    use_gpu = torch.cuda.is_available()

    ## prepare data
    dsets = {}
    dset_loaders = {}

    dsets['test'] = ImageList(crop_size=config.crop_size, path=config.test_path_prefix, phase='test',
                              transform=prep.image_test(crop_size=config.crop_size),
                              target_transform=prep.land_transform(img_size=config.crop_size,
                                                                   flip_reflect=np.loadtxt(
                                                                       config.flip_reflect))
                              )

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers, drop_last=True)

    ## set network modules
    region_learning = network.network_dict[config.region_learning](input_dim=3, unit_dim=config.unit_dim)
    align_net = network.network_dict[config.align_net](crop_size=config.crop_size, map_size=config.map_size,
                                                       au_num=config.au_num, land_num=config.land_num,
                                                       input_dim=config.unit_dim * 8, fill_coeff=config.fill_coeff)
    local_attention_refine = network.network_dict[config.local_attention_refine](au_num=config.au_num,
                                                                                 unit_dim=config.unit_dim)
    local_au_net = network.network_dict[config.local_au_net](au_num=config.au_num, input_dim=config.unit_dim * 8,
                                                             unit_dim=config.unit_dim)
    global_au_feat = network.network_dict[config.global_au_feat](input_dim=config.unit_dim * 8,
                                                                 unit_dim=config.unit_dim)
    au_net = network.network_dict[config.au_net](au_num=config.au_num, input_dim=3264, unit_dim=config.unit_dim)

    # gcn = network.network_dict['GCN'](nfeat=1440, nhid=config.gcn_hidden, nclass=config.au_num, dropout=config.dropout)
    transformer = vits.vit_small(
        patch_size=16,
        drop_path_rate=0.1,  # stochastic depth
        in_chans=30,
        batch_size=config.train_batch_size,
    )

    if config.pretrain:
        print('resuming model from epoch %d' % (config.pretrain_epoch))
        region_learning.load_state_dict(torch.load(
           '/home/songjuan/transformer-JAA/data/snapshots/JAA/region_learning_' + str(config.pretrain_epoch) + '.pth'))
        align_net.load_state_dict(torch.load(
           '/home/songjuan/transformer-JAA/data/snapshots/JAA/align_net_' + str(config.pretrain_epoch) + '.pth'))
        local_attention_refine.load_state_dict(torch.load(
           '/home/songjuan/transformer-JAA/data/snapshots/JAA/local_attention_refine_' + str(config.pretrain_epoch) + '.pth'))
        local_au_net.load_state_dict(torch.load(
           '/home/songjuan/transformer-JAA/data/snapshots/JAA/local_au_net_' + str(config.pretrain_epoch) + '.pth'))
        global_au_feat.load_state_dict(torch.load(
           '/home/songjuan/transformer-JAA/data/snapshots/JAA/global_au_feat_' + str(config.pretrain_epoch) + '.pth'))

    if config.start_epoch > 0:
        au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/au_net_' + str(num) + '.pth'))
        transformer.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/transformer_' + str(num) + '.pth'))

    if use_gpu:
        region_learning = region_learning.cuda()
        align_net = align_net.cuda()
        local_attention_refine = local_attention_refine.cuda()
        local_au_net = local_au_net.cuda()
        global_au_feat = global_au_feat.cuda()
        au_net = au_net.cuda()
        transformer = transformer.cuda()

    au_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, au_net.parameters()), 'lr': 1}]
    # gcn_parameter_list = [
    # {'params': filter(lambda p: p.requires_grad, gcn.parameters()), 'lr': 1}]
    transformer_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, transformer.parameters()), 'lr': 1}]


    optimizer = optim_dict[config.optimizer_type](itertools.chain(au_net_parameter_list,
                                                                  transformer_parameter_list),
                                                  lr=1.0, momentum=config.momentum, weight_decay=config.weight_decay,
                                                  nesterov=config.use_nesterov)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])


    if not os.path.exists(config.write_path_prefix + config.run_name):
        os.makedirs(config.write_path_prefix + config.run_name)
    if not os.path.exists(config.write_res_prefix + config.run_name):
        os.makedirs(config.write_res_prefix + config.run_name)

    res_file = open(
        config.write_res_prefix + config.run_name + '/AU_pred_' + str(config.start_epoch) + '.txt', 'w')

    region_learning.train(False)
    align_net.train(False)
    local_attention_refine.train(False)
    local_au_net.train(False)
    global_au_feat.train(False)
    au_net.train(False)
    transformer.train(False)
    local_f1score_arr, local_acc_arr, f1score_arr, acc_arr= AU_detection_evalv2(
        dset_loaders['test'], region_learning, align_net, local_attention_refine,
        local_au_net, global_au_feat, au_net,transformer, use_gpu=use_gpu)
    print('epoch =%d, local f1 score mean=%f, local accuracy mean=%f, '
          'f1 score mean=%f, accuracy mean=%f' % (num, local_f1score_arr.mean(),local_acc_arr.mean(),f1score_arr.mean(),acc_arr.mean()))
    print('%d\t%f\t%f\t%f\t%f' % (0, local_f1score_arr.mean(),local_acc_arr.mean(), f1score_arr.mean(),acc_arr.mean()), file=res_file)

    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")

    parser.add_argument('--pretrain', type=bool, default=True, help='')
    parser.add_argument('--pretrain_epoch', type=int, default=4, help='pretrained epoch')

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=112, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=28, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--train_batch_size', type=int, default=32, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=1, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=1, help='number of total epochs')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--gcn_hidden', type=int, default=30, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

    parser.add_argument('--region_learning', type=str, default='HMRegionLearning')
    parser.add_argument('--align_net', type=str, default='AlignNet')
    parser.add_argument('--local_attention_refine', type=str, default='LocalAttentionRefine')
    parser.add_argument('--local_au_net', type=str, default='LocalAUNetv2')
    parser.add_argument('--global_au_feat', type=str, default='HLFeatExtractor')
    parser.add_argument('--au_net', type=str, default='AUNet')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--fill_coeff', type=float, default=0.56)
    parser.add_argument('--run_name', type=str, default='JAA_Transformer')

    # Training configuration.
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')
    parser.add_argument('--display', type=int, default=100, help='iteration gaps for displaying')

    parser.add_argument('--optimizer_type', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD optimizer')
    parser.add_argument('--use_nesterov', type=str2bool, default=True)
    parser.add_argument('--lr_type', type=str, default='step')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.3, help='decay factor')
    parser.add_argument('--stepsize', type=int, default=2, help='epoch for decaying lr')

    # Directories.
    #parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    #parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    #parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    #parser.add_argument('--train_path_prefix', type=str,
                 #       default="/home/zhangchenggong/Competition/GCN-pretrained-JAA/data/list/train")
    #parser.add_argument('--test_path_prefix', type=str,
                     #   default="/home/yanzhengxiang/Alice/TransJAA_data/data3/AU_val")
    parser.add_argument('--write_path_prefix', type=str, default='/home/wenjiayuan/TransJAA_data/data_new/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='/home/wenjiayuan/TransJAA_data/data_new/res/')
    parser.add_argument('--flip_reflect', type=str, default='/home/wenjiayuan/TransJAA_data/data_new/reflect_49.txt')
    parser.add_argument('--test_path_prefix', type=str, default="/home/wenjiayuan/TransJAA_data/data_new/val")

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    #for i in range(3,1):
    main(config,10)