from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch_geometric.data import HeteroData


from typing import List
from tqdm import tqdm
from collections import defaultdict
from operator import itemgetter

from myutils import LabelBalancedSampler
from agg import InterAgg


class PCGNN(nn.Module):
    """ 
    一层PC-GNN用以message passing -> 核心特点是，居然需要label=。= \\
    论文源代码居然直接把一层当整个模型了……好🐕啊
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # adj_lists: List[defaultdict],
        # train_pos_mask: list,
        device: torch.device,
        normalize=True,
        num_classes: int = 2,
        bias=False,
    ):
        """
        一层Pick&Choose 的 message passager \\
        难道inter层不应该在这里进行声明吗…… \\
        :param in_channels: 输入的特征维数
        :param out_channels: 输出的特征维数
        :param num_classes: 最后需要做节点分类的类数
        """
        super(PCGNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.interAgg = InterAgg(
            feature_dim=in_channels,
            output_dim=out_channels,
            # adj_lists=adj_lists,
            # train_pos_mask=train_pos_mask,
            device=device
        )

        self.reset_parameters()

    def reset_parameters(self):
        # self.lin_l.reset_parameters()
        # self.lin_r.reset_parameters()
        pass

    def forward(self, features, labels, batch_mask, train_pos_mask, adj_lists, train_flag=True):
        """
        :param features: (|N|, input_channels)
        :param labels: (|N|,)
        :param batch_mask: (|B|,)在此次过程中需要考察的中心点的mask
        :param train_pos_mask:
        :param adj_lists:
        :return output_embeds: (|B|,out_dim)
        :return label_scores: (|B|,2)
        """

        embeds, logits = self.interAgg(
            features=features,
            batch_center_mask=batch_mask,
            batch_center_label=labels[batch_mask],
            train_pos_mask=train_pos_mask,
            adj_lists=adj_lists,
            train_flag=train_flag
        )

        return embeds, logits


class GNNStack(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.5,
        heads: int = 1,
        model_type: str = 'PCGNN',
        emb: bool = False
    ) -> None:
        super(GNNStack, self).__init__()

        conv_model = self.build_conv_model(model_type)
        # self.convs = nn.ModuleList()
        # self.convs.append(conv_model(
        #     in_channels=input_dim,
        #     out_channels=hidden_dim
        # ))
        assert (num_layers >= 1), 'Number of layers is not >=1'
        if num_layers == 1:
            self.convs = conv_model(
                in_channels=input_dim,
                out_channels=output_dim,
                device=device
            )
        else:
            # for l in range(num_layers):
            #     if l == 0:
            #         self.convs.append(conv_model(input_dim,hidden_dim))
            #     elif l == num_layers-1:
            #         self.convs.append(conv_model(heads*hidden_dim,output_dim))
            #     else:
            #         self.convs.append(conv_model(heads*hidden_dim, hidden_dim))
            raise NotImplementedError

        # # post-message-passing
        # self.post_mp = nn.Sequential(
        #     nn.Linear(heads * hidden_dim, hidden_dim), nn.Dropout(self.dropout),
        #     nn.Linear(hidden_dim, output_dim))
        # # 嗯……其实我一共设了num_layers+1层

        # PCGNN最后还有一个线性层用于node分类！
        self.final_proj = nn.Linear(output_dim, num_classes)

        self.dropout = dropout
        self.num_layers = num_layers

        self.emb = emb
        self.criterion = nn.CrossEntropyLoss()

    def build_conv_model(self, model_type):
        if model_type == 'PCGNN':
            return PCGNN
        else:
            raise NotImplementedError

    def forward(self, features, labels, batch_mask, train_pos_mask, adj_lists, train_flag=True):
        """ 
        :param features: 所有点的features
        :param labels: 所有点的label
        :param batch_mask: 本批次点的mask 或者是之后test/valid的时候的mask
        :param train_pos_mask:
        :param adj_lists:
        """
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            embeds, logits = self.convs(
                features, labels, batch_mask, train_pos_mask, adj_lists, train_flag)
            # x = F.relu(x)
            # x = F.dropout(x, p=self.dropout,training=self.training)

        # x = self.post_mp(x)
        if self.emb == True:
            return embeds

        return embeds, logits

    def loss(self, features, labels, batch_mask, train_pos_mask, adj_lists):
        """ 
        PCGNN的loss包括两个:loss_{gnn}和loss_{dist}
        """
        embeds, logits = self.forward(
            features=features,
            labels=labels,
            batch_mask=batch_mask,
            train_pos_mask=train_pos_mask,
            adj_lists=adj_lists
        )

        # PCGNN有两个loss
        # loss_{gnn}
        gnn_pred = self.final_proj(embeds)
        gnn_loss = self.criterion(gnn_pred, labels[batch_mask].squeeze())

        # loss_{dist}
        dist_loss = self.criterion(logits, labels[batch_mask].squeeze())

        return gnn_loss + dist_loss

    def test(self, features, labels, batch_mask, adj_lists, thres=0.5):
        with torch.no_grad():
            self.eval()
            embeds, logits = self.forward(
                features, labels, batch_mask, None, adj_lists, train_flag=False)
            gnn_prob = torch.sigmoid(self.final_proj(embeds))  # -> (|B|,2)
            label_prob1 = torch.sigmoid(logits)

            gnn_prob_arr = gnn_prob.data.cpu().numpy()[:, 1]

            gnn_pred = np.zeros_like(gnn_prob_arr, dtype=np.int32)
            gnn_pred[gnn_prob_arr >= thres] = 1
            gnn_pred[gnn_prob_arr < thres] = 0

            f1_label1 = f1_score(labels[batch_mask],
                                 label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
            acc_label1 = accuracy_score(labels[batch_mask],
                                        label_prob1.data.cpu().numpy().argmax(axis=1))
            recall_label1 = recall_score(
                labels[batch_mask], label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")

            gnn_pred_list = gnn_pred.tolist()
            gnn_prob_list = gnn_prob_arr.tolist()
            label_list1 = label_prob1.data.cpu().numpy()[:, 1].tolist()

            auc_gnn = roc_auc_score(
                labels[batch_mask], np.array(gnn_prob_list))
            ap_gnn = average_precision_score(
                labels[batch_mask], np.array(gnn_prob_list))
            auc_label1 = roc_auc_score(
                labels[batch_mask], np.array(label_list1))
            ap_label1 = average_precision_score(
                labels[batch_mask], np.array(label_list1))

            f1_binary_1_gnn = f1_score(labels[batch_mask], np.array(
                gnn_pred_list), pos_label=1, average='binary')
            f1_binary_0_gnn = f1_score(labels[batch_mask], np.array(
                gnn_pred_list), pos_label=0, average='binary')
            f1_micro_gnn = f1_score(labels[batch_mask], np.array(
                gnn_pred_list), average='micro')
            f1_macro_gnn = f1_score(labels[batch_mask], np.array(
                gnn_pred_list), average='macro')
            conf_gnn = confusion_matrix(
                labels[batch_mask], np.array(gnn_pred_list))
            tn, fp, fn, tp = conf_gnn.ravel()

            print(f"   GNN F1-binary-1: {f1_binary_1_gnn:.4f}\tF1-binary-0: {f1_binary_0_gnn:.4f}" +
                  f"\tF1-macro: {f1_macro_gnn:.4f}\tAUC: {auc_gnn:.4f}")
            print(f"   GNN TP: {tp}\tTN: {tn}\tFN: {fn}\tFP: {fp}")
            print(f"Label1 F1: {f1_label1 / len(batch_mask):.4f}\tAccuracy: {acc_label1 / len(batch_mask):.4f}" +
                  f"\tRecall: {recall_label1 / len(batch_mask):.4f}\tAUC: {auc_label1:.4f}\tAP: {ap_label1:.4f}")

            return f1_macro_gnn, f1_binary_1_gnn, f1_binary_0_gnn, auc_gnn


class ModelHandler():

    def __init__(
        self,
        data: HeteroData,
        data_name: str = 'Amazon',
        random_seed: int = 42,
        use_cuda: bool = True,
        opt: str = 'adam',
        weight_decay: float = 1e-2,
        lr: float = 0.005,
        dropout: float = 0.5,
        num_layers: int = 2,
        num_epochs: int = 51,
        batch_size: int = 256,
        valid_epochs: int = 5
    ) -> None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # preprare data
        self.data = data
        # 关于此数据集有个很重要的事，即，node数据都是ndarray格式，relation的数据都是tensor！
        # TODO normlize feature??

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.opt, self.weight_decay, self.lr = opt, weight_decay, lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.valid_epochs = valid_epochs

    def train(self):
        feat_data, label_data = self.data['review'].x, self.data['review'].y
        train_mask, valid_mask, test_mask = self.data['review'].train_mask, self.data[
            'review'].valid_mask, self.data['review'].test_mask
        train_pos_mask = self.data['review'].train_pos_mask
        adj_lists = [
            self.data['review', 'r1', 'review'].adj_list[0],
            self.data['review', 'r2', 'review'].adj_list[0],
            self.data['review', 'r3', 'review'].adj_list[0],
        ]

        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(
            feat_data), requires_grad=False).to(self.device)

        feat_data = torch.FloatTensor(feat_data, device=self.device)
        label_data = torch.LongTensor(label_data, device=self.device)

        # 我们只使用PCGNN
        GNN = GNNStack(
            input_dim=feat_data.shape[1],
            hidden_dim=64,
            output_dim=64,
            device=self.device
        )

        # optimizer
        if(self.opt == 'adam'):
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, GNN.parameters(
            )), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("This optimizer is not implemented yet.")

        f1_mac_best, auc_best, ep_best = 0, 0, -1

        for epoch in range(self.num_epochs):
            # Pick阶段，借助labelBalancedSampler进行一个降采样
            train_mask = LabelBalancedSampler(
                train_mask, label_data[train_mask], self.data['homo_adj_list'][0], len(self.data['review'].train_pos_mask)*2)
            # 这里的train mask是经过概率pick过的！

            # 我们在准备数据集时已经准备好了train_pos/neg_mask
            num_batches = int(len(train_mask) / self.batch_size) + 1

            loss = 0.

            # 开始batch训练
            for batch in tqdm(range(num_batches)):
                ind_start = batch*self.batch_size
                ind_end = min((batch+1)*self.batch_size, len(train_mask))
                batch_nodes_mask = train_mask[ind_start:ind_end]
                # batch_label = label_data[batch_nodes_mask]
                # TODO 这里的类型可能存在问题

                optimizer.zero_grad()
                loss = GNN.loss(
                    features=feat_data,
                    labels=label_data,
                    batch_mask=batch_nodes_mask,
                    train_pos_mask=train_pos_mask,
                    adj_lists=adj_lists
                )

                loss.backward()
                optimizer.step()
                loss += loss.item()

            print(f'Epoch: {epoch}, loss: {loss.item() / num_batches}')

            # valid(每3个epoch valid一次)
            if epoch % self.valid_epochs == 0:
                print(f"valid at epoch {epoch} ", end=" ")
                f1_mac_val, f1_1_val, f1_0_val, auc_val = GNN.test(
                    feat_data, label_data, valid_mask, adj_lists)

                if auc_val > auc_best:
                    f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
