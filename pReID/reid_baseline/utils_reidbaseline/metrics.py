import torch
import numpy as np
import torch.nn.functional as F

from .reranking import re_ranking

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


# def euclidean_distance(x, y):
# # def euclidean_distance(_x, _y):
#     # x = normalize(_x)
#     # y = normalize(_y)
#     m, n = x.size(0), y.size(0)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     # breakpoint()
#     dist = xx + yy
#     # dist.addmm_(1, -2, x, y.t())
#     dist.addmm(1, -2, x, y.t())
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     dist = dist.cpu().numpy()
#     # breakpoint()
#     # dist = normalize (dist)
#     return dist 

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    # breakpoint()       
    return dist_mat.cpu().numpy()


# def cosine_similarity(qf, gf):
#     epsilon    = 0.00001
#     dist_mat   = qf.mm(gf.t())
#     qf_norm    = torch.norm(qf, p = 2, dim = 1, keepdim = True)  # mx1
#     gf_norm    = torch.norm(gf, p = 2, dim = 1, keepdim = True)  # nx1
#     qg_normdot = qf_norm.mm(gf_norm.t())

#     dist_mat   = dist_mat.mul(1 / qg_normdot).cpu().numpy()
#     dist_mat   = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
#     dist_mat   = np.arccos(dist_mat)
#     return dist_mat

def cosine_similarity(qf, gf):
    dist = F.cosine_similarity(qf,gf,eps=1e-6)
    dist = dist.cpu().numpy()
    dist = dist.reshape((1,-1))
    return dist


def cosine_distance(x1, x2, eps=1e-8):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    # breakpoint()
    tmp = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    tmp = tmp.cpu().numpy()
    tmp = tmp.reshape((1,-1))
    return tmp

# def cosine_distance(x1, x2=None, eps=1e-8):
#     x2 = x1 if x2 is None else x2
#     w1 = x1.norm(p=2, dim=1, keepdim=True)
#     w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
#     # breakpoint()
#     tmp = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
#     tmp = tmp.cpu().numpy()
#     tmp = 1 - tmp.reshape((1,-1))
#     return tmp



def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc          = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP      = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP     = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP():
    def __init__(self, num_query, max_rank=50, feat_norm=True, method='euclidean', reranking=False):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank  = max_rank
        self.feat_norm = feat_norm
        self.method    = method
        self.reranking = reranking
        self.reset()

    def reset(self):
        self.feats  = []
        self.pids   = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf       = feats[:self.num_query]
        q_pids   = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf       = feats[self.num_query:]
        g_pids   = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=30, k2=10, lambda_value=0.2)

        else:
            if self.method == 'euclidean':
                print('=> Computing DistMat with euclidean distance')
                distmat = euclidean_distance(qf, gf)
            elif self.method == 'cosine':
                print('=> Computing DistMat with cosine similarity')
                distmat = cosine_similarity(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
