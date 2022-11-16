import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List
from tqdm import tqdm
from collections import defaultdict
from operator import itemgetter


class IntraAgg(nn.Module):
    """
    在某一关系下进行message aggregate
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        # features: torch.Tensor, # 怎么可能在你刚初始化的时候就把features传进来呢……你又未必是第一层
        # rho: float,
        # avg_half_pos_neigh : int, # 用于决定oversample时选多少个同类节点
        # train_pos_mask: list,
        device: torch.device
    ) -> None:
        """
        原文太无赖，居然在pclayer外面就把intraAgg声明好，数据也传进去了 \\
        :param feature_dim: 原数据每点的特征维数
        :param output_dim: 本层的嵌入维度，也就是输出维度
        """
        super(IntraAgg, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        # self.features = features # (|N|,feat_dim)
        # self.rho = rho # 用于距离函数判断的？
        self.device = device
        # self.train_pos_mask = train_pos_mask # 这是个列表啊
        # TODO 为什么这个线性层维度设置怪怪的
        self.proj = nn.Linear(2*feature_dim, output_dim)

        # 在train阶段，这些rho不会被用，但会被更新
        self.rho_neg = 0.5
        self.rho_pos = 0.5

        # self.avg_half_pos_neigh = avg_half_pos_neigh

    def forward(
        self,
        features: torch.Tensor,
        batch_center_mask: list,
        batch_center_labels: list,
        train_pos_mask: list,
        rx_list: List[list],
        batch_center_logits: torch.Tensor,  # (|B|,2)
        batch_all_logits: torch.Tensor,  # (|BatchAll|,2)
        train_pos_logits: torch.Tensor,  # (|Pos|,2)
        trainIdx2OrderIdx: dict,
        orderIdx2trainIdx: dict,
        avg_half_pos_neigh: int,
        train_flag: bool = True,  # 如果是test，没法靠标签信息来choose
        rho_neg: float = 0.4
    ):
        """
        在一层关系内进行message passing
        :param batch_center_mask: 本batch内的中心点
        :param batch_center_labels: 本batch中心点的label
        :param rx_list: 第x关系的所有中心点的邻居情况 列表套列表
        :param batch_center_logits:
        :param batch_all_logits:
        :param train_pos_logits:
        :param trainIdx2orderIdx: 从真正的node id投射到rxlist中索引的词典
        """

        self.avg_half_pos_neigh = avg_half_pos_neigh
        self.train_pos_mask = train_pos_mask

        # 考察为test/valid的情况
        if not train_flag:
            with torch.no_grad():
                # 首先，肯定要对邻居进行undersample
                # A(v,u)>0 且 D(v,u) < rho-
                out_feats = []
                for idx, one_center_logits in enumerate(batch_center_logits):
                    # 一定要考虑rxlist里面只有一个元素时的维数情况！
                    certain_neighbor_logits = batch_all_logits[np.array(itemgetter(
                        *(rx_list[idx]))(trainIdx2OrderIdx))].reshape(-1, 2)
                    # 计算distance
                    distance = torch.abs(certain_neighbor_logits -
                                         one_center_logits)[:, 0]
                    howManyNeighbors = distance.shape[0]
                    sampledNeighbor = (distance.argsort()[
                        0:int(howManyNeighbors / 2) + 1]).tolist()

                    # undersample之后的orderIdx在undersampledNeighbor里
                    # 进行aggregate！
                    neighbor_feats = features[np.array(itemgetter(
                        *sampledNeighbor)(orderIdx2trainIdx))].reshape(-1, self.feature_dim)
                    agg_feats = torch.mean(neighbor_feats, axis=0)

                    # 把和中心节点的feat进行contact
                    # 注意有一个reshape！
                    contacted_feat = torch.cat(
                        [features[orderIdx2trainIdx[idx]], agg_feats], axis=0).reshape(1, -1)
                    # shape: (1,2*h_{l-1})

                    # 进行线性映射
                    out_feats.append(F.relu(self.proj(contacted_feat)))

                rx_out_feats = torch.cat(out_feats, axis=0)
                return rx_out_feats

        # 此时只是train

        # 首先，肯定要对邻居进行undersample
        # A(v,u)>0 且 D(v,u) < rho-
        rx_list_undersampled = []
        out_feats = []
        for idx, one_center_logits in enumerate(batch_center_logits):
            # 先把这个中心点的邻居点的logits提取出来
            # certain_neighbor_logits = batch_all_logits[rx_list[idx]]
            # print(f"idx={idx}, len(rx_list[idx])={len(rx_list[idx])}")
            # if len(rx_list[idx]) == 1:
            #     certain_neighbor_logits = batch_all_logits[[itemgetter(
            #         *(rx_list[idx]))(trainIdx2OrderIdx)]]

            # 一定要考虑rxlist里面只有一个元素时的维数情况！
            certain_neighbor_logits = batch_all_logits[np.array(itemgetter(
                *(rx_list[idx]))(trainIdx2OrderIdx))].reshape(-1, 2)
            # 计算distance
            distance = torch.abs(certain_neighbor_logits -
                                 one_center_logits)[:, 0]
            howManyNeighbors = distance.shape[0]
            sampledNeighbor = (distance.argsort()[
                               0:int(howManyNeighbors / 2) + 1]).tolist()
            # 对rho-进行更新 这个更新有什么意义吗？
            # self.rho_neg = distance(distance.argsort()[int(howManyNeighbors / 2)])
            # rx_list_undersampled.append(nearest50Idx)
            # 这就是我们降采样之后的邻居样本，这里是orderIdx

            # label=1的时候是小样本！
            choosedSameClassNode = []
            if batch_center_labels[idx] == 1:
                # TODO 这里的维度肯定有点问题
                distance2 = torch.abs(
                    train_pos_logits - one_center_logits)[:, 0]  # 这个时候已经flatten了
                choosedSameClassNode = (distance2.argsort()[
                    0:self.avg_half_pos_neigh + 1]).tolist()

            # undersample之后的orderIdx在undersampledNeighbor里
            # oversample之后的orderIdx在choosedSameClassNode里

            # 进行aggregate！
            neighbor_feats = features[np.array(itemgetter(
                *sampledNeighbor)(orderIdx2trainIdx))].reshape(-1, self.feature_dim)
            if not choosedSameClassNode == []:
                minor_feats = features[np.array(self.train_pos_mask)[
                    choosedSameClassNode]].reshape(-1, self.feature_dim)
                neighbor_feats = torch.cat(
                    [neighbor_feats, minor_feats], axis=0)

            agg_feats = torch.mean(neighbor_feats, axis=0)

            # 把和中心节点的feat进行contact
            # 注意有一个reshape！
            contacted_feat = torch.cat(
                [features[orderIdx2trainIdx[idx]], agg_feats], axis=0).reshape(1, -1)
            # shape: (1,2*h_{l-1})

            # 进行线性映射
            out_feats.append(F.relu(self.proj(contacted_feat)))

        rx_out_feats = torch.cat(out_feats, axis=0)
        return rx_out_feats

    def NeighborhoodSamplerForTraining(
        self,
        batch_center_logits: torch.Tensor,
        batch_center_labels: torch.Tensor,
        batch_all_logits: torch.Tensor,
        train_pos_logits: torch.Tensor,
        rx_list: List[list]
    ):
        """
        这里是training阶段，我们将会根据邻居的情况来adaptively决定rho！
        """
        # 首先，肯定要对邻居进行undersample
        # A(v,u)>0 且 D(v,u) < rho-
        rx_list_undersampled = []
        for idx, one_center_logits in enumerate(batch_center_logits):
            # 先把这个中心点的邻居点的logits提取出来
            certain_neighbor_logits = batch_all_logits[rx_list[idx]]
            # 进行相减！
            distance = torch.abs(certain_neighbor_logits -
                                 one_center_logits)[:, 0]
            howManyNeighbors = distance.shape[0]
            undersampledNeighbor = (
                distance.argsort()[0:int(howManyNeighbors / 2) + 1]).tolist()
            # 对rho-进行更新 这个更新有什么意义吗？
            self.rho_neg = distance(
                distance.argsort()[int(howManyNeighbors / 2)])
            # rx_list_undersampled.append(nearest50Idx)
            # 这就是我们降采样之后的邻居样本，这里是orderIdx

            # label=1的时候是小样本！
            if batch_center_labels[idx] == 1:
                # TODO 这里的维度肯定有点问题
                distance2 = torch.abs(
                    train_pos_logits - one_center_logits)[:, 0]  # 这个时候已经flatten了
                choosedSameClassNode = distance2.argsort()[
                    0:self.avg_half_pos_neigh + 1]

            # undersample之后的orderIdx在undersampledNeighbor里
            # oversample之后的orderIdx在choosedSameClassNode里

    def NeighborhoodSamplerForTest(
        self,
    ):
        pass


class InterAgg(nn.Module):
    """ 
    对三层关系进行message aggregate
    """

    def __init__(
        self,
        # features: torch.Tensor,
        feature_dim: int,
        output_dim: int,
        # adj_lists: defaultdict,
        # train_pos_mask: list,
        device: torch.device,
        num_classes: int = 2,
        num_relations: int = 3
    ) -> None:
        super(InterAgg, self).__init__()
        # self.features = features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        # self.adj_lists = adj_lists # 3个关系的defaultdict
        # self.train_pos_mask = train_pos_mask
        self.device = device
        self.num_classes = num_classes

        # 三个关系的embedding要糅合成一个，需要进行一个线性层转换
        self.proj = nn.Linear(
            in_features=num_relations*self.output_dim + self.feature_dim,
            # 入维度是三个realtion的embedding和原特征contact在一起的
            out_features=output_dim
        )

        # 可是距离函数不是每个关系的都不一样吗？？
        self.label_linear = nn.Linear(self.feature_dim, self.num_classes)
        # 这个距离函数甚至只输出logits，没有进行sigmoid

        # 准备intraAgg层
        self.intra1 = IntraAgg(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            # avg_half_pos_neigh=self.avg_half_neigh_size[0],
            # train_pos_mask=self.train_pos_mask,
            device=self.device
        )
        self.intra2 = IntraAgg(
            self.feature_dim, self.output_dim, self.device
        )
        self.intra3 = IntraAgg(
            self.feature_dim, self.output_dim, self.device
        )

    def forward(
        self,
        features: torch.Tensor,
        batch_center_mask,
        batch_center_label,
        train_pos_mask,
        adj_lists,
        train_flag=True
    ):
        """ 
        :param batch_mask: 本批次中要训练的点
        :param batch_label: 本批次中训练点的label
        """
        self.features = features
        self.adj_lists = adj_lists
        self.train_pos_mask = train_pos_mask

        if train_flag:
            # 计算minority class的average neighborhood size
            avg_half_neigh_size = []
            for relationIdx in range(len(self.adj_lists)):
                total = 0
                for trainIdx in self.train_pos_mask:
                    total += len(self.adj_lists[relationIdx][trainIdx])
                avg_half_neigh_size.append(
                    int(total / len(self.train_pos_mask)))
            self.avg_half_neigh_size = avg_half_neigh_size
        else:
            self.avg_half_neigh_size = [0, 0, 0]  # 意思一下

        # batch_mask是什么，是本batch中所要考察的中心点
        # 我们后续要用到的信息包括本batch的中心点及它们的1-hop neighbor
        # 所以搞一个batch_all_mask，就包括了上述这些需要的点
        to_neighs = []  # to_neighs里面最终将会是三个列表，每个列表是某一个关系的adjlist转成列表
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)])
                             for node in batch_center_mask])
        # to_neighs be like: [[{某点的所有邻居},{},...,{}],[],[]]
        batch_all_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
                                    set.union(*to_neighs[2], set(batch_center_mask)))
        # batch_all_nodes be like: {0,1,3,4,5,...}
        batch_all_mask = list(batch_all_nodes)
        # batch_all_mask be like: [0,1,3,4,5,...]
        # batch_all_mask 内承载了本batch训练所需的所有点的index -> TODO 这个index是针对谁来说的？

        # 提取出本batch all点的feature
        # batch_features = self.features[batch_all_mask]
        # 我去，features是Embedding层……
        batch_all_features = self.features[torch.LongTensor(
            batch_all_mask).to(self.device)]
        # -> shape (|BatchAll|,feat_dim)
        # postive nodes features
        if train_flag:
            pos_features = self.features[torch.LongTensor(
                self.train_pos_mask).to(self.device)]
        # -> shape (|Pos|,feat_dim)

        # 出于加快访问速度的考虑（应该是），defaultdict涉及到查找过程——这太慢啦！
        # 但是nodes的trainIdx和在batch_all_mask中的orderIdx需要进行相互转化，对吧
        trainIdx2orderIdx = {trainIdx: orderIdx for trainIdx, orderIdx in zip(
            batch_all_mask, range(len(batch_all_nodes)))}
        orderIdx2trainIdx = (lambda d: dict(
            zip(d.values(), d.keys())))(trainIdx2orderIdx)

        # TODO 可是score不是在每个intra层里单独算的吗？？？
        # 先把batch all的logits都算完
        batch_all_logits = self.label_linear(batch_all_features)
        # 注意到，pos mask中的点很明显可能不在batch all中
        if train_flag:
            pos_logits = self.label_linear(pos_features)
        else:
            pos_logits = None

        # 提取一些特定点的logits
        # 提取本batch center点的logits
        batch_center_logits = batch_all_logits[list(itemgetter(
            *batch_center_mask)(trainIdx2orderIdx))]

        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]
        # rx_list: [[此关系下某个点的所有邻居],[],[]...,[]]

        r1_embeds = self.intra1.forward(
            self.features,
            batch_center_mask,
            batch_center_label,
            self.train_pos_mask,  # test的时候就是None
            r1_list,
            batch_center_logits,
            batch_all_logits,
            pos_logits,
            trainIdx2orderIdx,
            orderIdx2trainIdx,
            self.avg_half_neigh_size[0],
            train_flag=train_flag
        )
        r2_embeds = self.intra1.forward(
            self.features,
            batch_center_mask,
            batch_center_label,
            self.train_pos_mask,
            r2_list,
            batch_center_logits,
            batch_all_logits,
            pos_logits,
            trainIdx2orderIdx,
            orderIdx2trainIdx,
            self.avg_half_neigh_size[1],
            train_flag=train_flag
        )
        r3_embeds = self.intra1.forward(
            self.features,
            batch_center_mask,
            batch_center_label,
            self.train_pos_mask,
            r3_list,
            batch_center_logits,
            batch_all_logits,
            pos_logits,
            trainIdx2orderIdx,
            orderIdx2trainIdx,
            self.avg_half_neigh_size[2],
            train_flag=train_flag
        )
        # rx_embeds -> (|B|,out_dim)
        all_relation_and_self_embeds = torch.cat(
            [self.features[batch_center_mask], r1_embeds, r2_embeds, r3_embeds], dim=1)
        proj_all_embeds = F.relu(self.proj(all_relation_and_self_embeds))

        return proj_all_embeds, batch_center_logits
