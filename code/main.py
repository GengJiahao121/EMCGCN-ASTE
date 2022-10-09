#coding utf-8

import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator, label2id
from model import EMCGCN
import utils

import numpy as np

from prepare_vocab import VocabHelp
from transformers import AdamW

def get_bert_optimizer(model, args):
    # # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ["bert.embeddings", "bert.encoder"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.learning_rate
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer


def train(args): # 训练

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json')) # 从json文件中读取数据  训练集
    random.shuffle(train_sentence_packs) # 打乱句子顺序
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json')) # 验证集

    # load 四个句子特征各自的类别。这四个都是什么意思？应该是四个句子特征各自的标签类别信息
    post_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_post.vocab') # 
    deprel_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_deprel.vocab') # 
    postag_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_postag.vocab') # 
    synpost_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_synpost.vocab') # 
    args.post_size = len(post_vocab) # 81
    args.deprel_size = len(deprel_vocab) # 45
    args.postag_size = len(postag_vocab) # 855
    args.synpost_size = len(synpost_vocab) # 7

    instances_train = load_data_instances(train_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args) # 传入训练数据，构建模型用到的实例
    instances_dev = load_data_instances(dev_sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args) # 传入验证集数据，构建模型用到的实例
    random.shuffle(instances_train) # 打乱训练集形成的实例的顺序
    trainset = DataIterator(instances_train, args) # 这个DataIterator的作用应该是进行分批操作batch
    devset = DataIterator(instances_dev, args) # 同上，只不过是验证集

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = EMCGCN(args).to(args.device) # 初始化模型，装载模型到device
    optimizer = get_bert_optimizer(model, args) # 优化器

    # label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
    weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).float().cuda() # 这个weight是干什么用的？

    best_joint_f1 = 0 
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i)) # 第几次循环数据集epoch
        for j in trange(trainset.batch_count): # 第几个批次batch
            # sentences: 句子, tokens: 句子分词之后的结果, lengths: 词经过toid操作之后的tokens列表长度, masks: 一个max_sequence_lenght的列表，对应句子token长度的位置为1，其他位置为0
            # aspect_tags: 列表，padding部分为-1，其他部分为0， aspect部分是aspect第一个词的第一个token为1， 中间词的第一个token为2
            # tags: 矩阵，十种关系的哪一种，不是都是-1
            # tags_symmetry: 
            _, sentences, tokens, lengths, masks, _, _, aspect_tags, tags, word_pair_position, \
            word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry = trainset.get_batch(j)
            tags_flatten = tags.reshape([-1]) # 将矩阵展平
            tags_symmetry_flatten = tags_symmetry.reshape([-1]) # 将矩阵展平
            if args.relation_constraint: # 真实的矩阵和通过模型计算预测得到的矩阵之间进行求交叉熵进行拟合
                predictions = model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost) # 
                biaffine_pred, post_pred, deprel_pred, postag, synpost, final_pred = predictions[0], predictions[1], predictions[2], predictions[3], predictions[4], predictions[5]
                l_ba = 0.10 * F.cross_entropy(biaffine_pred.reshape([-1, biaffine_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_rpd = 0.01 * F.cross_entropy(post_pred.reshape([-1, post_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_dep = 0.01 * F.cross_entropy(deprel_pred.reshape([-1, deprel_pred.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_psc = 0.01 * F.cross_entropy(postag.reshape([-1, postag.shape[3]]), tags_symmetry_flatten, ignore_index=-1)
                l_tbd = 0.01 * F.cross_entropy(synpost.reshape([-1, synpost.shape[3]]), tags_symmetry_flatten, ignore_index=-1)

                if args.symmetry_decoding: # 意思是仅仅解码对角线的值，对角线的值无非是方面词和意见词，没有情感极性的预测
                    l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_symmetry_flatten, weight=weight, ignore_index=-1)
                else: # 默认是进行元组的抽取
                    l_p = F.cross_entropy(final_pred.reshape([-1, final_pred.shape[3]]), tags_flatten, weight=weight, ignore_index=-1)

                loss = l_ba + l_rpd + l_dep + l_psc + l_tbd + l_p
            else: # 这个就是仅仅只有Biaffine Attention的那个Loss
                preds = model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)[-1]
                preds_flatten = preds.reshape([-1, preds.shape[3]])
                if args.symmetry_decoding:
                    loss = F.cross_entropy(preds_flatten, tags_symmetry_flatten, weight=weight, ignore_index=-1)
                else:
                    loss = F.cross_entropy(preds_flatten, tags_flatten, weight=weight, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args) # 评估函数

        if joint_f1 > best_joint_f1: # 得到最优的f1值
            model_path = args.model_dir + 'bert' + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1)) # 输出


def eval(model, dataset, args, FLAG=False):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_sentences = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(dataset.batch_count):
            sentence_ids, sentences, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, \
            word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry = dataset.get_batch(i)
            preds = model(tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost)[-1]
            preds = F.softmax(preds, dim=-1)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)
            all_sentences.extend(sentences)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\t\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

        if FLAG:
            metric.tagReport()

    model.train()
    return precision, recall, f1


def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + 'bert' + args.task + '.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    post_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_post.vocab')
    deprel_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_deprel.vocab')
    postag_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_postag.vocab')
    synpost_vocab = VocabHelp.load_vocab(args.prefix + args.dataset + '/vocab_synpost.vocab')
    instances = load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args, False)


if __name__ == '__main__':
    # torch.set_printoptions(precision=None, threshold=float("inf"), edgeitems=None, linewidth=None, profile=None)
    parser = argparse.ArgumentParser() # 创建参数解析器对象

    # 给parser添加参数
    parser.add_argument('--prefix', type=str, default="../data/D1/",
                        help='dataset and embedding path prefix') #数据集文件路径前缀
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix') # 模型地址路径前缀
    parser.add_argument('--task', type=str, default="triplet", choices=["triplet"],
                        help='option: pair, triplet') # 任务类型
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test') # 模式：训练 or 测试
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset') # 数据集类型
    parser.add_argument('--max_sequence_len', type=int, default=102,
                        help='max length of a sentence') # 输入的句子的最大长度
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu') # 训练设备：gpu or cpu

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path') # 预训练的模型路径
    
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature') # bert的输入向量的维度

    parser.add_argument('--batch_size', type=int, default=16,
                        help='bathc size') # 但批输入数据量
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epoch number') # 轮次
    parser.add_argument('--class_num', type=int, default=len(label2id),
                        help='label number')  # 标签个数，10种关系
    parser.add_argument('--seed', default=1000, type=int)  # 随机种子
    parser.add_argument('--learning_rate', default=1e-3, type=float)  # 学习率
    parser.add_argument('--bert_lr', default=2e-5, type=float) # bert学习率
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # 优化器
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.") #  

    parser.add_argument('--emb_dropout', type=float, default=0.5) #
    parser.add_argument('--num_layers', type=int, default=1) #
    parser.add_argument('--pooling', default='avg', type=str, help='[max, avg, sum]') # 池化类型
    parser.add_argument('--gcn_dim', type=int, default=300, help='dimension of GCN') # GCN向量的维度
    parser.add_argument('--relation_constraint', default=True, action='store_true') # 
    parser.add_argument('--symmetry_decoding', default=False, action='store_true') #

    args = parser.parse_args() # 解析参数

    if args.seed is not None: # 固定随机数的数值，使每次产生的随机数是一致的，进而保证相同输入下，输出是相同的，因为初始化权重矩阵的随机参数可能有很多种，我们要每次实验都要他一样，才能进行实验
        random.seed(args.seed)
        np.random.seed(args.seed) 
        torch.manual_seed(args.seed) # 为cpu设置种子，产生随机数
        torch.cuda.manual_seed(args.seed) # 为gpu设置种子，产生随机数
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False # 如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的

    if args.task == 'triplet':
        args.class_num = len(label2id) # 任务为三元组： 类别个数设置为10

    if args.mode == 'train': 
        train(args) # 训练
        test(args) # 测试
    else:
        test(args) # 测试
