import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)
        # self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 1, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))

        # edge = self.W(torch.cat([edge, node], dim=-1))

        return edge


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = RefiningStrategy(gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5)

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape # 16 x 102 x 300
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2) # 16 x 102 x 102 x 50 -> 16 x 50 x 102 x 102 
    
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim) # 16 x 102 x 300 -> 16 x 1 x 102 x 300 -> 16 x 50 x 102 x 300

        weight_prob_softmax += self_loop
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs) # torch.matmul()是tensor乘法 16x102x102x50 X 16x50x102x300 
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim] 16 x 102 x 300
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs # 最终的经过图卷积神经网络的经过特征提取之后的词向量
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous() # 16 x 50 x 102 x 102 -> 16 x 102 x 102 x 50
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim) # 16 x 102 x 300 -> 16 x 1 x 102 x 300 -> 16 x 102 x 102 x 300
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous() # 16 x 102 x 102 x 300 -> 16 x 102 x 102 x 300
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2) # 这个就是节点和边关系进行拼接

        return node_outputs, edge_outputs


class Biaffine(nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features # 300
        self.in2_features = in2_features # 300
        self.out_features = out_features # 10
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0]) # 301
        self.linear_output_size = out_features * (in2_features + int(bias[1])) # 3010
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False) # 301 -> 3010

    def forward(self, input1, input2): # input1和input2分别是300维的词向量a和词向量o
        batch_size, len1, dim1 = input1.size() # 16 x 句子1的token长度 x dim
        batch_size, len2, dim2 = input2.size() # 16 x 句子2的token长度 x dim
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            input1 = torch.cat((input1, ones), dim=2) # dim变成301
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1) # 线性变换301 -> 3010 : 16 x len1 x 3010
        affine = affine.view(batch_size, len1*self.out_features, dim2) # 16 x len1*10 x 301
        input2 = torch.transpose(input2, 1, 2) # 16 x len2 x 301 -> 16 x 301 x len2
        biaffine = torch.bmm(affine, input2) # 矩阵乘法 16 x len1*10 x len2
        biaffine = torch.transpose(biaffine, 1, 2) # 16 x len2 x len1*10
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features) # 16 x len2 x len1 x 10
        return biaffine


class EMCGCN(torch.nn.Module): # 用pytorch建立的EMCGCN神经网络模型框架
    def __init__(self, args): # 对一些模型、参数进行初始化
        super(EMCGCN, self).__init__() # 固定写法，目的是向其父类中传递参数进行初始化其父类
        self.args = args 
        self.bert = BertModel.from_pretrained(args.bert_model_path) # 加载用到的bert模型
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path) # 加载分词器
        self.dropout_output = torch.nn.Dropout(args.emb_dropout) # 

        # 这里并没有生成向量，而是初始化了一个对象，这个对象是干生成向量这个活的
        self.post_emb = torch.nn.Embedding(args.post_size, args.class_num, padding_idx=0) # 相对位置嵌入
        self.deprel_emb = torch.nn.Embedding(args.deprel_size, args.class_num, padding_idx=0) # 依赖关系嵌入
        self.postag_emb  = torch.nn.Embedding(args.postag_size, args.class_num, padding_idx=0) # 词性标注嵌入
        self.synpost_emb = torch.nn.Embedding(args.synpost_size, args.class_num, padding_idx=0) # 基于依赖树的相对位置距离嵌入
        
        self.triplet_biaffine = Biaffine(args, args.gcn_dim, args.gcn_dim, args.class_num, bias=(True, True)) # 初始化Biaffine Attention对象
        self.ap_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim) # MLPa, 768->300前向全链接层，输入bert_feature_dim的向量维度768， 输出args.gcn_dim300维度
        self.op_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim) # MLPo, 768>300

        self.dense = nn.Linear(args.bert_feature_dim, args.gcn_dim) # 压缩线性变换
        self.num_layers = args.num_layers #
        self.gcn_layers = nn.ModuleList() # 图卷积神经网络的层

        self.layernorm = LayerNorm(args.bert_feature_dim) # 层次归一化

        for i in range(self.num_layers): # 1个
            self.gcn_layers.append(
                GraphConvLayer(args.device, args.gcn_dim, 5*args.class_num, args.class_num, args.pooling))

    def forward(self, tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost):
        bert_feature, _ = self.bert(tokens, masks)
        bert_feature = self.dropout_output(bert_feature) 

        batch, seq = masks.shape # 16 x 102
        tensor_masks = masks.unsqueeze(1).expand(batch, seq, seq).unsqueeze(-1) # 16 x 102 -> 16 x 102 x 102 -> 16 x 102 x 102 x 1
        
        # * multi-feature 16 x 102 x 102 x 10也就是说这个时候进行向量化
        word_pair_post_emb = self.post_emb(word_pair_position)
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel)
        word_pair_postag_emb = self.postag_emb(word_pair_pos)
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost)
        
        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature)) # 对应论文中的MLPa
        op_node = F.relu(self.op_fc(bert_feature)) # 对应论文中MLPo
        biaffine_edge = self.triplet_biaffine(ap_node, op_node) # 16 x 102 x 102 x 10 gcn的边关系输入
        gcn_input = F.relu(self.dense(bert_feature)) # 压缩，通过全链接网络变换一下维度 gcn的词向量输入
        gcn_outputs = gcn_input # 上一层的output是下一层的input
 
        weight_prob_list = [biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, word_pair_postag_emb, word_pair_synpost_emb] # 各种R
        
        biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks # 对类别分数进行归一化到0~1之间
        word_pair_post_emb_softmax = F.softmax(word_pair_post_emb, dim=-1) * tensor_masks
        word_pair_deprel_emb_softmax = F.softmax(word_pair_deprel_emb, dim=-1) * tensor_masks
        word_pair_postag_emb_softmax = F.softmax(word_pair_postag_emb, dim=-1) * tensor_masks
        word_pair_synpost_emb_softmax = F.softmax(word_pair_synpost_emb, dim=-1) * tensor_masks

        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq)) # torch.eye()生成对角钱全1， 其余部分为0的二维数组 batchsize = 16 -> 16个102 x 102 对角线为1 的多维矩阵
        # torch.stack()作用是将一个个二维数组进行拼接，形成一个三维矩阵。.unsqueeze()函数作用是升维。 .expend()函数作用是：将张量广播到新形状16 x 5*10 x 102 x 102 。
        # .permute()函数的作用是重新排列，也就是重新建立形状，置换维度。
        # 16 x 102 x 102 -> 增加维度16 x 1 x 102 x 102 -> 16 x 5*10 x 102 x 102 
        # tensor_masks: 16 x 102 x 102 x 1 -> 16 x 1 x 102 x 102 -> 经过contiguous()函数之后内存存储顺序也改变了
        # 最后self_loop是16 x 50 x 102 x 102, 且对多余部分进行了去除，都设为0
        self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(1).expand(batch, 5*self.args.class_num, seq, seq) * tensor_masks.permute(0, 3, 1, 2).contiguous()
        
        weight_prob = torch.cat([biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, \
            word_pair_postag_emb, word_pair_synpost_emb], dim=-1) # 拼接的是5个R 16 x 102 x 102 x 50
        weight_prob_softmax = torch.cat([biaffine_edge_softmax, word_pair_post_emb_softmax, \
            word_pair_deprel_emb_softmax, word_pair_postag_emb_softmax, word_pair_synpost_emb_softmax], dim=-1) # 拼接的是经过softmax的5个R

        for _layer in range(self.num_layers): # 图卷积神经网络
            gcn_outputs, weight_prob = self.gcn_layers[_layer](weight_prob_softmax, weight_prob, gcn_outputs, self_loop)  # [batch, seq, dim]
            weight_prob_list.append(weight_prob)

        return weight_prob_list