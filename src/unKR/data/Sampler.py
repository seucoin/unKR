import torch

from .DataPreprocess import *
from torch.autograd import Variable

def generate_distribution(conf, sigma, size):

    gauss = stats.norm(conf, sigma)
    pdf = gauss.pdf(np.linspace(0, 1, size))
    label_dis = pdf/np.sum(pdf)
    return label_dis


def generate_one_hot_distribution(conf, size):
    index = int(conf * (size - 1))

    label_dis = np.zeros(size)

    label_dis[index] = 1.0

    return label_dis

class UKGEUniSampler(UKGEBaseSampler):
    """Random negative sampling
    Filtering out positive samples and selecting some samples randomly as negative samples.

    Attributes:
        cross_sampling_flag: The flag of cross sampling head and tail negative samples.
    """

    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0
        if self.args.model_name == 'ssCDL':
            self.sigma = args.sigma
            self.size = args.size

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        torch.set_printoptions(precision=8)
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t, _ in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r - 1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            # print(data)
            for h, r, t, _ in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r - 1)]
                    subsampling_weight.append(weight)
        small_data = []
        neg_ent_sample_small = []
        for h, r, t, s in data:
            if s <= 0.5:
                small_data.append((h, r, t, s))
        for h, r, t, _ in small_data:
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_ent_sample_small.append(neg_tail)
            # if self.args.use_weight:
            #     weight = self.count[(h, r)] + self.count[(t, -r - 1)]
            #     subsampling_weight.append(weight)
        batch_data["ori_data"] = data # storage the original data, it will be used in PASSLEAFLitModel.
        # batch_data['small_data'] = []
        batch_data['small_data'] = small_data
        batch_data["positive_sample"] = torch.tensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        batch_data['true_head'] = self.rt2h_train
        batch_data['true_tail'] = self.hr2t_train
        batch_data["small_data_positive_sample"] = torch.tensor(np.array(small_data))
        batch_data["small_data_negative_sample"] = torch.tensor(np.array(neg_ent_sample_small))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1 / torch.tensor(subsampling_weight))

        # When train student_model in UPGAT, get pseudo_sample and adj_matrix as below.
        if self.args.model_name == 'UPGAT':
            train_data = torch.tensor([item[:3] for item in self.train_triples])
            if not self.args.teacher_model:
                pseudo_data = torch.tensor([item[:3] for item in self.pseudo_triples])
                train_data = torch.cat((train_data, pseudo_data), dim=0)
                try:
                    batch_data["pseudo_sample"] = torch.stack(next(self.pseudo_dataiter), dim=1)
                except StopIteration:
                    pseudo_dataloader = DataLoader(self.pseudo_triples, batch_size=self.args.pseudo_bs,
                                                   num_workers=self.args.num_workers, pin_memory=True, shuffle=True,
                                                   drop_last=True)
                    self.pseudo_dataiter = iter(pseudo_dataloader)
                    batch_data["pseudo_sample"] = torch.stack(next(self.pseudo_dataiter),dim=1)
            head, rela, tail = train_data.t()
            self.adj_matrix = (torch.stack((head, tail)), rela)
            batch_data["adj_matrix"] = self.adj_matrix

        if self.args.model_name == 'ssCDL':
            data_ldl = batch_data["positive_sample"].clone()
            conf = []
            onehot = []
            # transform confidence into confidence distribution
            for triples in data_ldl:
                ldl = generate_distribution(triples[3].item(),self.sigma,self.size)
                # print(ldl)
                tmp = torch.from_numpy(ldl)
                # print(tmp.size())
                conf.append(tmp)
            batch_data["conf_ldl"] = conf
            # for triples in data_ldl:
            for triples in data_ldl:
                ldl = generate_one_hot_distribution(triples[3].item(),self.size)
                # print(ldl)
                tmp = torch.from_numpy(ldl)
                # print(tmp.size())
                onehot.append(tmp)
            batch_data["onehot"] = onehot

            data_ldl = batch_data["small_data_positive_sample"].clone()
            conf_small = []
            onehot_small = []
            for triples in data_ldl:
                ldl = generate_distribution(triples[3].item(), self.sigma, self.size)
                # print(ldl)
                tmp = torch.from_numpy(ldl)
                # print(tmp.size())
                conf_small.append(tmp)

            batch_data["small_data_conf_ldl"] = conf_small
            # for triples in data_ldl:
            for triples in data_ldl:
                ldl = generate_one_hot_distribution(triples[3].item(),self.size)
                # print(ldl)
                tmp = torch.from_numpy(ldl)
                # print(tmp.size())
                onehot_small.append(tmp)
            batch_data["small_data_onehot"] = onehot_small

            train_triples_num = len(self.train_triples)

            interval_ratios = {key: count / train_triples_num for key, count in self.interval_counts.items()}
            weights_tensor = torch.zeros(len(data))

            for idx, (_, _, _, s) in enumerate(data):
                interval_index = int(s * 10)
                interval_key = f"{interval_index / 10:.1f}-{(interval_index + 1) / 10:.1f}"  # 更新区间的表示方式

                weight = interval_ratios[interval_key]
                weights_tensor[idx] = weight
            batch_data['weights_tensor'] = weights_tensor

        return batch_data

    def uni_sampling(self, data):
        batch_data = {}
        neg_head_list = []
        neg_tail_list = []
        for h, r, t, _ in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_head_list.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_tail_list.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_head'] = torch.LongTensor(np.arrary(neg_head_list))
        batch_data['negative_tail'] = torch.LongTensor(np.arrary(neg_tail_list))
        return batch_data

    def get_sampling_keys(self):
        if self.args.model_name == 'UPGAT':
            if self.args.teacher_model:
                return ['positive_sample', 'negative_sample', 'mode', 'adj_matrix']
            else:
                return ['positive_sample', 'negative_sample', 'mode', 'adj_matrix', 'pseudo_sample']
        else:
            return ['positive_sample', 'negative_sample', 'mode']


class UKGEPSLSampler(UKGEBaseSampler):
    """Random negative sampling Filtering out positive samples and selecting some samples randomly as negative samples.
    UKGEPSLSampler is for UKGE_PSL

    Attributes:
        cross_sampling_flag: The flag of cross sampling head and tail negative samples.
    """

    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """

        batch_data = {}
        neg_ent_sample = []  # negative samples
        PSL_sample = []  # PSL samples
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t, _ in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r - 1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t, _ in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r - 1)]
                    subsampling_weight.append(weight)

        PSL_ent_num = int(self.RatioOfPSL * len(data)) # calculate the number of PSL samples in each batch, storage the PSL samples in batch_data['PSL_sample']

        if len(self.PSL_triples) >= PSL_ent_num + 1:
            for i in range(PSL_ent_num + 1):
                PSL_sample.append(self.PSL_triples[i])
            self.PSL_triples = self.PSL_triples[PSL_ent_num + 1:]
        else:
            for i in range(len(self.PSL_triples)):
                PSL_sample.append(self.PSL_triples[i])
        batch_data["positive_sample"] = torch.tensor(np.array(data))
        batch_data["negative_sample"] = torch.tensor(np.array(neg_ent_sample))
        batch_data['PSL_sample'] = torch.tensor(np.array(PSL_sample))

        batch_data['true_head'] = self.h2rt_train
        batch_data['true_tail'] = self.hr2t_train
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1 / torch.tensor(subsampling_weight))

        return batch_data

    def uni_sampling(self, data):
        batch_data = {}
        neg_head_list = []
        neg_tail_list = []
        for h, r, t, _ in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_head_list.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_tail_list.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_head'] = torch.LongTensor(np.arrary(neg_head_list))
        batch_data['negative_tail'] = torch.LongTensor(np.arrary(neg_tail_list))
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']

class UKGETestSampler(object):
    """Sampling triples and recording positive triples for testing.
    We offer two test sample methods: one is to use the same method as neuralkg to process the test set,
    and the other is to only use high confidence samples from the test set for testing

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        hr2t_all_high_score: Record the tail corresponding to the same head and relation (only for high score samples in test).
        rt2h_all_high_score: Record the head corresponding to the same tail and relation (only for high score samples in test).
        num_ent: The count of entities.
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.hr2t_all_high_score = ddict(set)
        self.rt2h_all_high_score = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.
        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
            self.hr2t_all_high_score: The set of hr2t (only for high score samples in test).
            self.rt2h_all_high_score: The set of rt2h (only for high score samples in test).
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t, s in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
            if s >= self.sampler.args.confidence_filter:
                self.hr2t_all_high_score[(h, r)].add(t)
                self.rt2h_all_high_score[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for h, r in self.hr2t_all_high_score:
            self.hr2t_all_high_score[(h, r)] = torch.tensor(list(self.hr2t_all_high_score[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))
        for r, t in self.rt2h_all_high_score:
            self.rt2h_all_high_score[(r, t)] = torch.tensor(list(self.rt2h_all_high_score[(r, t)]))

    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        head_label_filter = torch.zeros(len(data), self.num_ent)
        tail_label_filter = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail, score = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
            if self.rt2h_all_high_score[(rel, tail)] != set():
                head_label_filter[idx][self.rt2h_all_high_score[(rel, tail)]] = 1.0
            if self.hr2t_all_high_score[(head, rel)] != set():
                tail_label_filter[idx][self.hr2t_all_high_score[(head, rel)]] = 1.0

        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        batch_data["head_label_filter"] = head_label_filter
        batch_data["tail_label_filter"] = tail_label_filter
        return batch_data

    def construct_hr_map(self, data):
        hr_map = {}
        for h, r, t, w in data:
            w = float(w)

            if hr_map.get(h) == None:
                hr_map[h] = {}
            if hr_map[h].get(r) == None:
                hr_map[h][r] = {t: w}
            else:
                hr_map[h][r][t] = w

        return hr_map

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label"]
        # return ["positive_sample"]


class GMUCSampler(GMUCBaseSampler):
    """GMUC sampling
        Process task-based data.
    """

    def __init__(self, args):
        super().__init__(args)

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.

        Args:
            data: A task/relation and all its triples.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        query, train_and_test1 = data[0].keys(), data[0].values()
        query = str(list(query)).lstrip('[').rstrip(']').replace("'", "")
        #  降维度
        train_and_test1 = list(train_and_test1)
        task_triples = [item for sublist in train_and_test1 for item in sublist]

        if self.args.type_constrain:
            candidates = self.rel2candidates[query]
        else:
            candidates = list(self.ent2id.keys())
            random.shuffle(candidates)
            candidates = candidates[:1000]

        random.shuffle(task_triples)

        support_triples = task_triples[:self.args.few]
        support_pairs = [[self.symbol2id[triple[0]], self.symbol2id[triple[2]], float(triple[3])] for triple in
                         support_triples]  # (h, t, s)
        support_left = [self.ent2id[triple[0]] for triple in support_triples]
        support_right = [self.ent2id[triple[2]] for triple in support_triples]

        all_test_triples = task_triples[self.args.few:]

        if len(all_test_triples) < self.args.train_bs:
            query_triples = [random.choice(all_test_triples) for _ in range(self.args.train_bs)]
        else:
            query_triples = random.sample(all_test_triples, self.args.train_bs)

        query_pairs = [[self.symbol2id[triple[0]], self.symbol2id[triple[2]], float(triple[3])] for triple in
                       query_triples]  # (h, t, s)
        query_left = [self.ent2id[triple[0]] for triple in query_triples]
        query_right = [self.ent2id[triple[2]] for triple in query_triples]
        query_confidence = [float(triple[3]) for triple in query_triples]

        false_pairs, false_left, false_right = self.generate_false(query_triples, candidates)

        support_meta = self.get_meta(support_left, support_right)
        query_meta = self.get_meta(query_left, query_right)
        false_meta = self.get_meta(false_left, false_right)

        # symbolid-ic value pairs for loss
        symbolid_ic = []
        if self.args.has_ont:
            if self.args.rel_uc == 1:
                rel2ic = self.rel_uc1
            else:
                rel2ic = self.rel_uc2

            symbols = set()
            for triple in task_triples:
                e1 = triple[0]
                e2 = triple[2]
                if e1 not in symbols:
                    symbolid_ic.append([self.symbol2id[e1], float(self.ent2ic[e1])])
                    symbols.add(e1)
                if e2 not in symbols:
                    symbolid_ic.append([self.symbol2id[e2], float(self.ent2ic[e2])])
                    symbols.add(e2)
            symbolid_ic.append([self.symbol2id[query], float(rel2ic[query])])

        if self.args.if_GPU:
            support = Variable(torch.Tensor(support_pairs)).cuda()
            query = Variable(torch.Tensor(query_pairs)).cuda()
            false = Variable(torch.Tensor(false_pairs)).cuda()
            symbolid_ic = Variable(torch.Tensor(symbolid_ic)).cuda()
        else:
            support = Variable(torch.Tensor(support_pairs))
            query = Variable(torch.LongTensor(query_pairs))
            false = Variable(torch.LongTensor(false_pairs))
            symbolid_ic = Variable(torch.Tensor(symbolid_ic))

        batch_data["support"] = support
        batch_data["query"] = query
        batch_data["false"] = false
        batch_data["query_confidence"] = query_confidence
        batch_data["support_meta"] = support_meta
        batch_data["query_meta"] = query_meta
        batch_data["false_meta"] = false_meta
        batch_data["symbolid_ic"] = symbolid_ic

        return batch_data

    def get_meta(self, left, right):
        """get meta data
        """
        if self.args.if_GPU:
            left_connections = Variable(
                torch.Tensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
            left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
            right_connections = Variable(
                torch.Tensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
            right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        else:
            left_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0)))
            left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left]))
            right_connections = Variable(torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0)))
            right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right]))

        return left_connections, left_degrees, right_connections, right_degrees

    def get_sampling_keys(self):
        return ['support', 'query', 'false', 'query_confidence', 'support_meta', 'query_meta', 'false_meta']


class GMUCTestSampler(object):
    """Sampling triples and recording positive triples for testing.

    Attributes:
        sampler: The function of training sampler.
        num_ent: The number of entities.
        rel2candidates: Record the entities corresponding to the same relation, type: list.
        symbol2id: Encoding the entity and relation in triples, type: dict.
        ent2id: Encoding the entity in triples, type: dict.
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.args = self.sampler.args
        self.num_ent = sampler.args.num_ent
        self.rel2candidates = sampler.rel2candidates
        # self.test_with_neg_tasks = sampler.test_with_neg_tasks
        self.symbol2id = sampler.symbol2id
        self.ent2id = sampler.ent2id

    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        query = str(list(data[0].keys())).replace("'", "").replace("[", "").replace("]", "")
        triple = list(data[0].values())
        triples = [item for sublist in triple for item in sublist]

        batch_data = {}
        batch_data["query"] = query
        batch_data["triples"] = triples

        return batch_data

    def get_sampling_keys(self):
        return ["query", "triples"]


class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]
