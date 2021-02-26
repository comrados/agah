import torch
from PIL import Image
import numpy as np
import visdom
import time
import pickle
from tqdm import tqdm


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current query sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from current query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    #mask = (P > 0).float().sum(dim=0)
    #mask = mask + (mask == 0).float() * 0.001
    #P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    #mask = (R > 0).float().sum(dim=0)
    #mask = mask + (mask == 0).float() * 0.001
    #R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P, R


def p_top_k(qB, rB, query_label, retrieval_label, K, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'AP@K ' + tqdm_label

    num_query = qB.shape[0]
    PK = torch.zeros(len(K)).to(qB.device)

    for i in tqdm(range(num_query), desc=tqdm_label):
        # ground_truth[j] == 1 if same class (if at least 1 same label), otherwise 0, ground truth
        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        hamm_dist = calc_hamming_dist(qB[i, :], rB).squeeze()

        # for each k in K
        for j, k in enumerate(K):
            k = min(k, retrieval_label.shape[0])
            _, sorted_indexes = torch.sort(hamm_dist)
            retrieved_indexes = sorted_indexes[:k]
            retrieved_samples = ground_truth[retrieved_indexes]
            PK[j] += retrieved_samples.sum() / k

    PK = PK / num_query

    """
    import matplotlib.pyplot as plt
    plt.semilogx(K, PK)
    plt.savefig('/home/george/Downloads/_fig.png')
    """

    return PK


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_labels = torch.Tensor([[0, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [1, 0, 0, 1],
                                 [0, 1, 0, 1]])
    retrieval_labels = torch.Tensor([[1, 0, 0, 1],
                                     [1, 1, 0, 0],
                                     [0, 1, 1, 0],
                                     [0, 0, 1, 0],
                                     [1, 0, 0, 0],
                                     [0, 0, 1, 0]])

    # query_labels = torch.Tensor([[0, 1, 0, 0],
    #                              [1, 0, 0, 0],
    #                              [1, 0, 0, 0],
    #                              [0, 1, 0, 0]])
    # retrieval_labels = torch.Tensor([[1, 0, 0, 0],
    #                                  [0, 1, 0, 0],
    #                                  [0, 0, 1, 0],
    #                                  [0, 0, 1, 0],
    #                                  [1, 0, 0, 0],
    #                                  [0, 0, 1, 0]])

    trn_bainary = torch.Tensor(
        [
            [1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1],
            [-1, 1, 1, -1, 1]
        ]
    )
    tst_binary = torch.Tensor(
        [
            [1, 1, 1, -1, -1],
            [1, 1, 1, 1, 1]
        ]
    )
    trn_label = torch.Tensor(
        [[0, 1], [0, 1], [1, 0], [1, 0]]
    )
    tst_label = torch.Tensor(
        [[1, 0], [0, 0]]
    )

    map = calc_map_k(qB, rB, query_labels, retrieval_labels)
    # map = calc_map(tst_binary, trn_bainary, tst_label, trn_label)
    print(map)
    # a = torch.randint(0, 256, (224, 224, 3))
    # image_from_numpy(a)
