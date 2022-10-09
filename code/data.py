import math
import torch
import numpy as np
from collections import OrderedDict, defaultdict
from transformers import BertTokenizer


sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}

label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
# label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}

label2id, id2label = OrderedDict(), OrderedDict()
for i, v in enumerate(label):
    label2id[v] = i
    id2label[i] = v

# 返回标签标记的index范围
def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


# 需要处理的单个句子的各种值的初始化
class Instance(object):
    def __init__(self, tokenizer, sentence_pack, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args):
        self.id = sentence_pack['id'] # 句子id
        self.sentence = sentence_pack['sentence'] # 句子信息
        self.tokens = self.sentence.strip().split() #将句子进行分词得到的词列表; Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        self.postag = sentence_pack['postag'] # 词性标注信息，目的是确定词的词性
        self.head = sentence_pack['head'] # ???
        self.deprel = sentence_pack['deprel'] # 词与词之间的关系，依存关系
        self.sen_length = len(self.tokens) # 句子长度
        self.token_range = [] # 每个词的开始地址和结束地址 
        self.bert_tokens = tokenizer.encode(self.sentence) # 将句子序列转换成数字序列，并加入开始CLS和SEP
        #初始化bert的输入的各种用到的序列
        self.length = len(self.bert_tokens) # 加入CLS和SEP之后的长度
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long() # bert的输入序列统一长度max_sequence
        self.aspect_tags = torch.zeros(args.max_sequence_len).long() # 
        self.opinion_tags = torch.zeros(args.max_sequence_len).long() # 
        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long() # 矩阵
        self.tags_symmetry = torch.zeros(args.max_sequence_len, args.max_sequence_len).long() # 
        self.mask = torch.zeros(args.max_sequence_len) # 

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i] # 句子长度未达到最大长度而进行的padding补0操作
        self.mask[:self.length] = 1 #将mask序列全置为1

        # 计算每个token的开始地址和结束地址,因为有的词需要占用两个位置
        token_start = 1 
        for i, w, in enumerate(self.tokens):
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        assert self.length == self.token_range[-1][-1]+2 # 多了CLS, SEP

        self.aspect_tags[self.length:] = -1 # padding部分设置为-1
        self.aspect_tags[0] = -1 # 第一个为cls，不可能为aspect,设置为-1
        self.aspect_tags[self.length-1] = -1 # 最后一个为seq结束符，不可能为aspect, 设置为-1

        self.opinion_tags[self.length:] = -1 # 和aspect进行同样的操作
        self.opinion_tags[0] = -1 
        self.opinion_tags[self.length - 1] = -1

        self.tags[:, :] = -1 # 标签矩阵初始化为-1
        self.tags_symmetry[:, :] = -1 
        for i in range(1, self.length-1): # 句子所在index的范围
            for j in range(i, self.length-1): # 上三角
                self.tags[i][j] = 0

        # 虽然我们从json中读取的数据，但是我们要将其转换成机器识别的形式，所以我们要对各种用到的矩阵，数组进行赋值，初始化
        for triple in sentence_pack['triples']: # 迭代单个句子中的每组三元组
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect) # 方面词所在index范围
            opinion_span = get_spans(opinion) # 意见词所在index范围

            '''set tag for aspect'''
            for l, r in aspect_span: # aspect_span说的是句子中的方面词开始index和结束index
                start = self.token_range[l][0]
                end = self.token_range[r][1]  # 为什么这样弄，因为将词转换为bert id时，有可能一个词用两个id才可以表示，所以index不相互对应
                # 对tags矩阵进行贴标签
                for i in range(start, end+1): # 
                    for j in range(i, end+1):
                        if j == start: # j == start 说明它时方面词词组的第一个词
                            self.tags[i][j] = label2id['B-A']
                        elif j == i: # j != start && j == i 说明它是属于方面词的内部
                            self.tags[i][j] = label2id['I-A'] # i == j 要不是B-A, 要不是I-A
                        else:
                            self.tags[i][j] = label2id['A'] # 同属于一个方面词, i != j就是同属于一个方面词组
                # 给aspect_tags, tags贴标签，是aspect的设为1或2， 不是的设为-1，若一个词由多个token表示，则设该词的第一个token为1，其他的token为-1
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2 # i == l 代表方面词的第一个词
                    al, ar = self.token_range[i] # 方面词中词i的开始位置和结束位置
                    self.aspect_tags[al] = set_tag # 
                    self.aspect_tags[al+1:ar+1] = -1 # 除方面词第一个token之外的其他方面词tokens设置为-1
                    '''mask positions of sub words'''
                    self.tags[al+1:ar+1, :] = -1 # 行和列全都设置为-1
                    self.tags[:, al+1:ar+1] = -1

            '''set tag for opinion'''
            for l, r in opinion_span:
                start = self.token_range[l][0] # 开始那个词对应的tokens的开始的token
                end = self.token_range[r][1] # 到结束那个词对应的tokens的结束的token
                # 对tags矩阵进行贴标签
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        if j == start: # i = j时，判断token属于什么类别， i != j时，判断他们之间的关系
                            self.tags[i][j] = label2id['B-O'] # B-O, I-O都是位于对角线的
                        elif j == i:
                            self.tags[i][j] = label2id['I-O']
                        else:
                            self.tags[i][j] = label2id['O'] # 同属于一个方面词组对应的tokens里面
                
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2 # 方面词组的第一个词为1，其他的词为2 
                    pl, pr = self.token_range[i] # 获取词对应的token的开始index和结束index
                    self.opinion_tags[pl] = set_tag # 方面词组的第一个词的第一个token设为1，第一个词的其他token设为-1； 方面词组的除了第一个方面词的词对应的token的第一个token设为2， 其他token设为-1
                    self.opinion_tags[pl+1:pr+1] = -1 
                    self.tags[pl+1:pr+1, :] = -1 # tags矩阵的上述说的位置也是同样的规则进行初始化
                    self.tags[:, pl+1:pr+1] = -1
            # 一个方面词一个意见的这样的关系，在tags矩阵中对应的位置进行赋值
            for al, ar in aspect_span: # 方面词组在句子中的跨度，如果有多个方面词组，就迭代多个
                for pl, pr in opinion_span: # 意见词组在句子中的跨度
                    for i in range(al, ar+1): # 方面词在句子中的跨度的迭代
                        for j in range(pl, pr+1): # 意见词在句子中的跨度的迭代
                            sal, sar = self.token_range[i] # 方面词对应的token的开始index和结束index
                            spl, spr = self.token_range[j] # 意见词对应的token的开始index和结束index
                            self.tags[sal:sar+1, spl:spr+1] = -1 # 一个是aspect,一个时opinion这样的关系的位置的值设为-1 ，先全设为-1，下面对有关系的地方再进行赋值
                            if args.task == 'pair': # 任务类型为输出pair, 不是triplet
                                if i > j: # 方面词在意见词后面
                                    self.tags[spl][sal] = 7 # 设为7，7对应的label为negative,这个是随便设的，不为-1就说明他们两个具备a-o关系，不需要识别情感
                                else: # 方面词在意见词前面
                                    self.tags[sal][spl] = 7
                            elif args.task == 'triplet':
                                if i > j:# 这个大小关系的目的是因为我仅仅用到上三角，所以遇到这个标签在下三角的我把它调到上三角
                                    self.tags[spl][sal] = label2id[triple['sentiment']] # 
                                else:
                                    self.tags[sal][spl] = label2id[triple['sentiment']]
        # 将tags_symmetry赋值成和tags矩阵一样内容
        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags_symmetry[i][j] = self.tags[i][j]
                self.tags_symmetry[j][i] = self.tags_symmetry[i][j]
        
        '''1. generate position index of the word pair''' # 词与词之间在句子中的相对位置距离
        self.word_pair_position = torch.zeros(args.max_sequence_len, args.max_sequence_len).long() # 初始化
        for i in range(len(self.tokens)): # 句子中词的个数的迭代
            start, end = self.token_range[i][0], self.token_range[i][1] # 每个词对应的token的开始index和结束index
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_position[row][col] = post_vocab.stoi.get(abs(row - col), post_vocab.unk_index)
        
        """2. generate deprel index of the word pair""" # 依赖关系
        self.word_pair_deprel = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = self.token_range[self.head[i] - 1] if self.head[i] != 0 else (0, 0)
                for k in range(s, e + 1):
                    self.word_pair_deprel[j][k] = deprel_vocab.stoi.get(self.deprel[i])
                    self.word_pair_deprel[k][j] = deprel_vocab.stoi.get(self.deprel[i])
                    self.word_pair_deprel[j][j] = deprel_vocab.stoi.get('self')
        
        """3. generate POS tag index of the word pair""" # 词性标注
        self.word_pair_pos = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_pos[row][col] = postag_vocab.stoi.get(tuple(sorted([self.postag[i], self.postag[j]])))
                        
        """4. generate synpost index of the word pair""" # 基于句法的相对位置
        self.word_pair_synpost = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        tmp = [[0]*len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            j = self.head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)
            
        word_level_degree = [[4]*len(self.tokens) for _ in range(len(self.tokens))]

        for i in range(len(self.tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)
        
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_synpost[row][col] = synpost_vocab.stoi.get(word_level_degree[i][j], synpost_vocab.unk_index)

# 加载实例，实例中包括了四种类型的句子特征的矩阵 + biaffine attention用到的相关数据 
def load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args): 
    instances = list()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    for sentence_pack in sentence_packs:
        instances.append(Instance(tokenizer, sentence_pack, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        tags_symmetry = []
        word_pair_position = []
        word_pair_deprel = []
        word_pair_pos = []
        word_pair_synpost = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            bert_tokens.append(self.instances[i].bert_tokens_padding)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags)
            opinion_tags.append(self.instances[i].opinion_tags)
            tags.append(self.instances[i].tags)
            tags_symmetry.append(self.instances[i].tags_symmetry)

            word_pair_position.append(self.instances[i].word_pair_position)
            word_pair_deprel.append(self.instances[i].word_pair_deprel)
            word_pair_pos.append(self.instances[i].word_pair_pos)
            word_pair_synpost.append(self.instances[i].word_pair_synpost)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)
        tags_symmetry = torch.stack(tags_symmetry).to(self.args.device)

        word_pair_position = torch.stack(word_pair_position).to(self.args.device)
        word_pair_deprel = torch.stack(word_pair_deprel).to(self.args.device)
        word_pair_pos = torch.stack(word_pair_pos).to(self.args.device)
        word_pair_synpost = torch.stack(word_pair_synpost).to(self.args.device)

        return sentence_ids, sentences, bert_tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, \
            word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry
