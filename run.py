from __future__ import print_function, absolute_import
from reid.bottom_up import *
from reid import datasets
from reid import models
import numpy as np
import argparse
import os, sys, time
from reid.utils.logging import Logger
import os.path as osp
from torch.backends import cudnn


def resume(args):
    import re
    pattern=re.compile(r'step_(\d+)\.ckpt')
    start_step = -1
    ckpt_file = ""

    # find start step
    files = os.listdir(args.logs_dir)
    files.sort()
    for filename in files:
        try:
            iter_ = int(pattern.search(filename).groups()[0])
            if iter_ > start_step:
                start_step = iter_
                ckpt_file = osp.join(args.logs_dir, filename)
        except:
            continue

    # if need resume
    if start_step >= 0:
        print("continued from iter step", start_step)

    return start_step, ckpt_file


def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    import warnings

    warnings.filterwarnings("ignore")
    save_path = args.logs_dir
    sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ str(args.merge_percent)+ time.strftime(".%m.%d_%H.%M.%S") + '.txt'))
    resume_step, ckpt_file = -1, ''
    if args.resume:
        resume_step, ckpt_file = resume(args)
    # get all unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    new_train_data, cluster_id_labels = change_to_unlabel(dataset_all)

    num_train_ids = len(np.unique(np.array(cluster_id_labels)))
    nums_to_merge = int(num_train_ids * args.merge_percent)

    BuMain = Bottom_up(model_name=args.arch, batch_size=args.batch_size, 
            num_classes=num_train_ids,
            dataset=dataset_all,
            u_data=new_train_data, save_path=args.logs_dir, max_frames=args.max_frames,
            embeding_fea_size=args.fea, initial_steps=args.steps,load_path=args.load_path)

    print("==========\nArgs:{}\n==========".format(args))
    for step in range(int(1/args.merge_percent)-1):

        if step < resume_step:
            continue
        print('step: ', step)
        BuMain.train(new_train_data, step, loss=args.loss)if step != resume_step else BuMain.resume(ckpt_file, step)

        BuMain.evaluate(dataset_all.query, dataset_all.gallery, step)

        # get new train data for the next iteration
        print('----------------------------------------bottom-up clustering------------------------------------------------')
        cluster_id_labels, new_train_data = BuMain.get_new_train_data(cluster_id_labels, nums_to_merge, size_penalty=args.size_penalty)
        print('\n\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bottom-up clustering')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-f', '--fea', type=int, default=2048)
    parser.add_argument('-a', '--arch', type=str, default='avg_pool',choices=models.names())
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'logs'))
    parser.add_argument('--max_frames', type=int, default=250)
    parser.add_argument('--loss', type=str, default='ExLoss')
    parser.add_argument('-m', '--momentum', type=float, default=0.5)
    parser.add_argument('-s', '--step_size', type=int, default=55)
    parser.add_argument('--size_penalty',type=float, default=0.005)
    parser.add_argument('-mp', '--merge_percent',type=float, default=0.05)
    parser.add_argument('--resume', type=str, default=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--load_path', type=str, metavar='PATH',
                        default=os.path.join(working_dir, 'examples'))
    main(parser.parse_args())

