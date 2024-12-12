import numpy as np
import torch
import torch.nn as nn
from utils.helper import default_device
import torch.nn.functional as F
from geoopt import Lorentz
from geoopt import ManifoldParameter
import models.encoders as encoders
import geoopt as gt
import geoopt.manifolds.stereographic.math as pmath
from utils.train_utils import build_tree
from config import parser

eps = 1e-15
MIN_NORM = 1e-15
dropout = 0.5
site=[]
args = parser.parse_args()
cuda_device = torch.device('cuda:0')
path = './data/'
dataset = args.dataset
implication = torch.load(path + dataset + '/implication.pt', map_location='cuda:0')

class MobiusLinear(nn.Linear):
    def __init__(self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball_ = gt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = gt.ManifoldParameter(self.bias, manifold=self.ball_)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() * 1e-3, k=self.ball_.k))
        with torch.no_grad():
            fin, fout = self.weight.size()
            k = (6 / (fin + fout)) ** 0.5  # xavier uniform
            self.weight.uniform_(-k, k)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.ball_.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += ", hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


def mobius_linear(input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, k=-1.0,):
    if hyperbolic_input:
        weight = F.dropout(weight, dropout)
        output = pmath.mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, k=k)
        output = pmath.mobius_add(output, bias, k=k)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, k=k)
    output = pmath.project(output, k=k)
    return output


class VorRec(nn.Module):

    def __init__(self, users_items, args, feature_num, hidden_size, embed_dim, num_tag, **kwargs):
        super(VorRec, self).__init__(**kwargs)
        self.c = torch.tensor([args.c]).to(default_device())
        self.manifold = Lorentz(args.c)
        self.encoder = getattr(encoders, "HG")(args.c, args)

        self.num_users, self.num_items = users_items
        self.weight_decay = args.weight_decay
        self.margin1 = args.margin1
        self.margin2 = args.margin2
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.tau = args.tau
        self.tag_weight = nn.Parameter(torch.zeros(num_tag), requires_grad=True)
        self.entropy = None
        self.args = args

        self.ball_ = gt.PoincareBall(c=1.0)
        points = torch.randn(num_tag, embed_dim)*1e-5
        points = pmath.expmap0(points.to(cuda_device), k=self.ball_.k)
        self.emb_tag = gt.ManifoldParameter(points, manifold=self.ball_)

        self.encoder_HMI = nn.Sequential(
            MobiusLinear(feature_num, embed_dim, bias=True, nonlin=None),
        )

        self.vtg = nn.Embedding(num_embeddings=self.num_items, embedding_dim=args.embedding_dim).to(default_device())
        self.vtg.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.vtg.weight = nn.Parameter(self.manifold.expmap0(self.vtg.state_dict()['weight']))
        self.vtg.weight = ManifoldParameter(self.vtg.weight, self.manifold, requires_grad=True)

        self.utg = nn.Embedding(num_embeddings=self.num_users, embedding_dim=args.embedding_dim).to(default_device())
        self.utg.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.utg.weight = nn.Parameter(self.manifold.expmap0(self.utg.state_dict()['weight']))
        self.utg.weight = ManifoldParameter(self.utg.weight, self.manifold, requires_grad=True)


    def encode(self, adj):
        adj = adj.to(default_device())
        emb_utg = self.manifold.projx(self.utg.weight)
        emb_vtg = self.manifold.projx(self.vtg.weight)
        x2 = torch.cat([emb_utg, emb_vtg], dim=0)
        h_tg = self.encoder.encode(x2, adj)
        return h_tg


    def decode(self, h_all, idx):
        h = h_all
        emb_utg = h[idx[:, 0].long()]
        emb_vtg = h[idx[:, 1].long()]
        assert not torch.isnan(emb_utg).any()
        assert not torch.isinf(emb_vtg).any()
        sqdist = self.manifold.dist2(emb_utg, emb_vtg, keepdim=True)
        return sqdist


    def calc_alpha(self, emb_u, user_ids):
        origin = self.manifold.origin(args.embedding_dim)
        tmp = self.manifold.dist2(emb_u, origin, keepdim=True)
        alpha = torch.sqrt(tmp)
        return alpha


    def compute_loss(self, embeddings, triples, tag_labels):
        assert not torch.isnan(triples).any()
        triples = triples.to(default_device())
        train_edges = triples[:, [0, 1]]
        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)
        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)
        loss = pos_scores - neg_scores + self.margin2
        emb_u = embeddings[train_edges[:, 0].long()]
        alpha = self.calc_alpha(emb_u, train_edges[:, 0].long())
        loss = loss * alpha
        loss[loss < 0] = 0
        loss = torch.sum(loss)

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

        if self.lambda1 > 0:
            # Voronoi diagram
            logits = self.Ex_loss(self.vtg.weight)
            rows, cols = np.where(tag_labels > 0)
            target = logits[rows, cols].unsqueeze(-1)
            all_dist = logits[rows]
            classification_loss = (torch.sum(F.relu(torch.log(target) - torch.log(all_dist) + self.margin1), dim=-1)).sum()
            loss = loss + self.lambda1 * classification_loss

            # record entropy
            with torch.no_grad():
                tmp = torch.softmax(-logits, dim=-1)
                self.entropy = -torch.sum(tmp*torch.log(tmp+eps), dim=-1).mean()

            # contrastive learning
            q = self.emb_tag[implication[:, 0].long()]
            pos = self.emb_tag[implication[:, 1].long()]
            dist_pos = self.ball_.dist2(q, pos, keepdim=True)

            q1 = q.unsqueeze(1).expand(-1, self.emb_tag.shape[0], -1)
            neg = self.emb_tag.expand_as(q1)
            dist_neg = self.ball_.dist2(q1, neg)

            cl_logits = torch.cat([dist_pos, dist_neg], dim=1) / self.tau
            cl_logits = -cl_logits[cl_logits > 1e-9].reshape(q1.shape[0], q1.shape[1])
            cri_cl = nn.CrossEntropyLoss(reduction='sum')
            labels = torch.zeros(cl_logits.shape[0], dtype=torch.long).to(cuda_device)
            cl_loss = cri_cl(cl_logits, labels)

            loss = loss + self.lambda2 * cl_loss

        return loss


    def Ex_loss(self, X):
        encoded = self.ball_.projx(X)
        encoded = self.encoder_HMI(encoded)
        self.ball_.assert_check_point_on_manifold(encoded)
        dist = torch.cdist(self.ball_.logmap0(encoded), self.ball_.logmap0(self.emb_tag))
        log_probability = dist / torch.exp(self.tag_weight)
        return log_probability


    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_utg = h[:, :][i].repeat(num_items).view(num_items, -1)
            emb_vtg = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.dist2(emb_utg, emb_vtg)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix
