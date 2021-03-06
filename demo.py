import torch
import pickle
import os.path as osp


def evaluate(features, load_path):

    with open(load_path, "wb") as fp:
        pickle.dump({"features": features}, fp)


working_dir = osp.dirname(osp.abspath(__file__))
a=torch.rand(2,3)
evaluate(a, load_path = osp.join(working_dir, 'examples/features_mars_step0.pickle'))
b=1
