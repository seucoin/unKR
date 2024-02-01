import copy
import json
import math
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os
from collections import defaultdict as ddict
from tqdm import tqdm
import random


class UKGData(object):
    """Data preprocessing of ukg data.

    Attributes:
        args: Some pre-set parameters, such as dataset path, etc. 
        ent2id: Encoding the entity in triples, type: dict.
        rel2id: Encoding the relation in triples, type: dict.
        id2ent: Decoding the entity in triples, type: dict.
        id2rel: Decoding the realtion in triples, type: dict.
        train_triples: Record the triples for training, type: list.
        valid_triples: Record the triples for validation, type: list.
        test_triples: Record the triples for testing, type: list.
        PSL_triples: Record the triples for softlogic, type: list. (will be used in UKGE_PSL)
        pseudo_triples: Record the triples for pseudo, type: list. (will be used in UPGAT)
        all_true_triples: Record all triples including train,valid and test, type: list.
        hr2t_train: Record the tail corresponding to the same head and relation, type: defaultdict(class:set).
        rt2h_train: Record the head corresponding to the same tail and relation, type: defaultdict(class:set).
        h2rt_train: Record the tail, relation corresponding to the same head, type: defaultdict(class:set).
        t2rh_train: Record the head, realtion corresponding to the same tail, type: defaultdict(class:set).
        hr2t_total: Record the tail corresponding to the same head and relation in whole dataset(train + val + test), type: defaultdict(class:set).
        rt2h_total: Record the head corresponding to the same tail and relation in whole dataset(train + val + test), type: defaultdict(class:set).
        RatioOfPSL: Record the ratio of the number of PSL samples to the number of training samples. (will be used in UKGE_PSL)
        pseudo_dataiter: Record the data in dataloader. (will be used in UPGAT)
    """

    def __init__(self, args):
        self.args = args

        # basic part
        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = {}
        self.id2rel = {}
        # store the ID of the sample
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        self.PSL_triples = []
        self.all_true_triples = set()
        # for PSL
        self.RatioOfPSL = 0
        # for UPGAT
        self.pseudo_triples = []
        self.pseudo_dataiter = None
        # for calculating nDCG
        # self.hr_map = {}
        self.hr2t_train = ddict(set)
        self.rt2h_train = ddict(set)
        self.h2rt_train = ddict(set)
        self.t2rh_train = ddict(set)
        self.hr2t_total = ddict(set)
        self.rt2h_total = ddict(set)

        self.get_id()
        # self.get_ndcg_test_data()
        if args.use_weight:
            self.count = self.count_frequency(self.train_triples)

    def get_id(self):
        """Get entity/relation id, and entity/relation number.

        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.id2ent: id to Entity.
            self.id2rel: id to Relation.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """
        last_e = -1
        last_r = -1

        with open(os.path.join(self.args.data_path, "train.tsv"), encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip('\n').split('\t')
                h, r, t, w = line[0], line[1], line[2], float(line[3])

                if self.ent2id.get(h) == None:
                    last_e += 1
                    self.ent2id[h] = last_e
                    self.id2ent[last_e] = h

                if self.ent2id.get(t) == None:
                    last_e += 1
                    self.ent2id[t] = last_e
                    self.id2ent[last_e] = t

                if self.rel2id.get(r) == None:
                    last_r += 1
                    self.rel2id[r] = last_r
                    self.id2rel[last_r] = r

                h = self.ent2id[h]
                r = self.rel2id[r]
                t = self.ent2id[t]

                self.train_triples.append((h, r, t, w))

        with open(os.path.join(self.args.data_path, "val.tsv"), encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip('\n').split('\t')
                h, r, t, w = line[0], line[1], line[2], float(line[3])

                if self.ent2id.get(h) == None:
                    last_e += 1
                    self.ent2id[h] = last_e
                    self.id2ent[last_e] = h

                if self.ent2id.get(t) == None:
                    last_e += 1
                    self.ent2id[t] = last_e
                    self.id2ent[last_e] = t

                if self.rel2id.get(r) == None:
                    last_r += 1
                    self.rel2id[r] = last_r
                    self.id2rel[last_r] = r

                h = self.ent2id[h]
                r = self.rel2id[r]
                t = self.ent2id[t]

                self.valid_triples.append((h, r, t, w))

        with open(os.path.join(self.args.data_path, "test.tsv"), encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip('\n').split('\t')
                h, r, t, w = line[0], line[1], line[2], float(line[3])

                if self.ent2id.get(h) == None:
                    last_e += 1
                    self.ent2id[h] = last_e
                    self.id2ent[last_e] = h

                if self.ent2id.get(t) == None:
                    last_e += 1
                    self.ent2id[t] = last_e
                    self.id2ent[last_e] = t

                if self.rel2id.get(r) == None:
                    last_r += 1
                    self.rel2id[r] = last_r
                    self.id2rel[last_r] = r

                h = self.ent2id[h]
                r = self.rel2id[r]
                t = self.ent2id[t]

                self.test_triples.append((h, r, t, w))

        softlogic_temp_path = os.path.join(self.args.data_path, "softlogic.tsv")
        if os.path.exists(softlogic_temp_path):
            with open(os.path.join(self.args.data_path, "softlogic.tsv"), encoding='utf-8') as fin:
                for line in fin:
                    line = line.rstrip('\n').split('\t')
                    h, r, t, w = line[0], line[1], line[2], float(line[3])

                    if self.ent2id.get(h) == None:
                        last_e += 1
                        self.ent2id[h] = last_e
                        self.id2ent[last_e] = h

                    if self.ent2id.get(t) == None:
                        last_e += 1
                        self.ent2id[t] = last_e
                        self.id2ent[last_e] = t

                    if self.rel2id.get(r) == None:
                        last_r += 1
                        self.rel2id[r] = last_r
                        self.id2rel = r

                    h = self.ent2id[h]
                    r = self.rel2id[r]
                    t = self.ent2id[t]
                    self.PSL_triples.append((h, r, t, w))

        self.RatioOfPSL = len(self.PSL_triples) / len(self.train_triples) # calculate the ratio of the number of PSL samples to the number of training samples.

        # When train student_model in UPGAT, get pseudo_triples as below.
        if self.args.model_name == 'UPGAT' and not self.args.teacher_model:
            with open(os.path.join(self.args.data_path, "pseudo.tsv"), encoding='utf-8') as fin:
                for line in fin:
                    line = line.rstrip('\n').split('\t')
                    h, r, t, w = int(line[0]), int(line[1]), int(line[2]), float(line[3])
                    self.pseudo_triples.append((h, r, t, w))
            self.args.pseudo_bs = len(self.pseudo_triples) // (len(self.train_triples)//self.args.train_bs)
            pseudo_dataloader = DataLoader(self.pseudo_triples, batch_size=self.args.pseudo_bs, num_workers=self.args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
            self.pseudo_dataiter = iter(pseudo_dataloader)
            self.all_true_triples = set(
                self.train_triples + self.valid_triples + self.test_triples + self.pseudo_triples
            )
        else:
            self.all_true_triples = set(
                self.train_triples + self.valid_triples + self.test_triples
            )

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)

    # def get_ndcg_test_data(self):
    #     # 使用test_triples构造成hr_map
    #     with open(os.path.join(self.args.data_path, "ndcg_test.pickle"), 'rb') as f:
    #         self.hr_map = pickle.load(f)  # unpickle

        # self.hr_map = {}
        # for triple in self.test_triples:
        #     h, r, t, w = triple
        #     if self.hr_map.get(h) is None:
        #         self.hr_map[h] = {}
        #     if self.hr_map[h].get(r) is None:
        #         self.hr_map[h][r] = {t: w}
        #     else:
        #         self.hr_map[h][r][t] = w

        # tmp = 0
        # for h in self.hr_map:
        #     for r in self.hr_map[h]:
        #         tmp = max(tmp, len(self.hr_map[h][r].keys()))
        #
        # print(self.hr_map)
        # print(len(self.hr_map.keys()))
        # print(tmp)
        # exit(0)


    def get_hr2t_rt2h_from_train(self):
        """Get the set of hr2t and rt2h from train dataset, the data type is numpy.

        Update:
            self.hr2t_train: The set of hr2t.
            self.rt2h_train: The set of rt2h.
        """

        for h, r, t, w in self.train_triples:
            self.hr2t_train[(h, r)].add(t)
            self.rt2h_train[(r, t)].add(h)
        for h, r in self.hr2t_train:
            self.hr2t_train[(h, r)] = np.array(list(self.hr2t_train[(h, r)]))
        for r, t in self.rt2h_train:
            self.rt2h_train[(r, t)] = np.array(list(self.rt2h_train[(r, t)]))

    # Add by Messi
    def get_hr2t_rt2h_from_total(self):
        """Get the set of hr2t and rt2h from whole dataset(train+val+test), the data type is numpy.

        Update:
            self.hr2t_total: The set of hr2t in whole dataset.
            self.rt2h_total: The set of rt2h in whole dataset.
        """
        for h, r, t, w in self.all_true_triples:
            self.hr2t_total[(h, r)].add(t)
            self.rt2h_total[(r, t)].add(h)
        for h, r in self.hr2t_total:
            self.hr2t_total[(h, r)] = np.array(list(self.hr2t_total[(h, r)]))
        for r, t in self.rt2h_total:
            self.rt2h_total[(r, t)] = np.array(list(self.rt2h_total[(r, t)]))

    @staticmethod
    def count_frequency(triples, start=4):
        '''Get frequency of a partial triple like (head, relation) or (relation, tail).

        The frequency will be used for subsampling like word2vec.

        Args:
            triples: Sampled triples.
            start: Initial count number.

        Returns:
            count: Record the number of (head, relation).
        '''
        count = {}
        for head, relation, tail, w in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    def get_h2rt_t2hr_from_train(self):
        """Get the set of h2rt and t2hr from train dataset, the data type is numpy.

        Update:
            self.h2rt_train: The set of h2rt.
            self.t2rh_train: The set of t2hr.
        """
        for h, r, t, w in self.train_triples:
            self.h2rt_train[h].add((r, t))
            self.t2rh_train[t].add((r, h))
        for h in self.h2rt_train:
            self.h2rt_train[h] = np.array(list(self.h2rt_train[h]))
        for t in self.t2rh_train:
            self.t2rh_train[t] = np.array(list(self.t2rh_train[t]))

    def get_hr_train(self):
        '''Change the generation mode of batch.
        Merging triples which have same head and relation for 1vsN training mode.

        update:
            self.train_triples: The tuple(hr, t) list for training
        '''
        self.t_triples = self.train_triples
        self.train_triples = [(hr, list(t)) for (hr, t) in self.hr2t_train.items()]


class UKGEBaseSampler(UKGData):
    """
        data processing for UKG data, the sampling method is consistent with NeuralKG (https://github.com/zjukg/NeuralKG)
    """
    def __init__(self, args):
        super().__init__(args)
        self.get_hr2t_rt2h_from_train()
        self.get_hr2t_rt2h_from_total()

    def corrupt_head(self, t, r, num_max=1):
        """Negative sampling of head entities.

        Args:
            t: Tail entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated

        Returns:
            neg: The negative sample of head entity filtering out the positive head entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.rt2h_total[(r, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def corrupt_tail(self, h, r, num_max=1):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple.
            r: Relation in triple.
            num_max: The maximum of negative samples generated

        Returns:
            neg: The negative sample of tail entity filtering out the positive tail entity.
        """
        tmp = torch.randint(low=0, high=self.args.num_ent, size=(num_max,)).numpy()
        if not self.args.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.hr2t_total[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def head_batch(self, h, r, t, neg_size=None):
        """Negative sampling of head entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of head entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_head(t, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def tail_batch(self, h, r, t, neg_size=None):
        """Negative sampling of tail entities.

        Args:
            h: Head entity in triple
            t: Tail entity in triple.
            r: Relation in triple.
            neg_size: The size of negative samples.

        Returns:
            The negative sample of tail entity. [neg_size]
        """
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.corrupt_tail(h, r, num_max=(neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def get_train(self):
        return self.train_triples

    def get_valid(self):
        return self.valid_triples

    def get_test(self):
        return self.test_triples

    def get_PSL(self):
        return self.PSL_triples

    def get_all_true_triples(self):
        return self.all_true_triples

    def get_hr_map(self):
        return self.hr_map


class GMUCData(object):
    """Data processing for few-shot GMUC & GMUC+ data.

    Attributes:
        args: Some pre-set parameters, such as dataset path, etc.
        ent2id: Encoding the entity in triples, type: dict.
        rel2id: Encoding the relation in triples, type: dict.
        symbol2id: Encoding the entity and relation in triples, type: dict.
        e1rel_e2: Record the tail corresponding to the same head and relation, type: defaultdict(class:list).
        rele1_e2: Record the tail corresponding to the same relation and head, type: defaultdict(class:dict).
        train_tasks: Record the triples for training, type: dict.
        dev_tasks: Record the triples for validation, type: dict.
        test_tasks: Record the triples for testing, type: dict.
        task_pool: A task is a relation, type: list.
        path_graph: Background triples, type: list.
        type2ents: All entities of the same type, type: defaultdict(class:set).
        known_rels: The triples of path_graph and train_tasks, type: defaultdict(class:list).
        num_tasks: The number of tasks, type: int.
        rel2candidates: Record the entities corresponding to the same relation, type: list.
        ent2ic: Calculate IIC for every entity, type: dict.
        rel_uc1: Calculate IIC for every relation, type: dict.
        rel_uc2: Calculate IIC for every relation, type: dict.
        connections: Neighbor information for each entity, type: numpy.
        e1_rele2: Record the relation and tail corresponding to the same head, type: defaultdict(class:list).
        e1_degrees: Record the number of neighbors per entity, type: defaultdict(class:int).
    """

    def __init__(self, args):

        self.args = args

        self.ent2id = {}
        self.rel2id = {}
        self.symbol2id = {}  # ent + rel
        self.rel2candidates = {}
        self.e1rel_e2 = ddict(list)
        self.rele1_e2 = ddict(dict)

        self.train_tasks = json.load(open(self.args.data_path + '/train_tasks.json'))
        self.dev_tasks = json.load(open(self.args.data_path + '/dev_tasks.json'))
        self.test_tasks = json.load(open(self.args.data_path + '/test_tasks.json'))
        self.task_pool = list(self.train_tasks.keys())
        self.num_tasks = len(self.task_pool)
        self.path_graph = open(self.args.data_path + '/path_graph').readlines()
        self.type2ents = ddict(set)
        self.known_rels = ddict(list)

        self.get_tasks()
        self.get_rel2candidates()
        self.get_e1rel_e2()
        self.get_rele1_e2()
        degrees = self.build_graph(max_=self.args.max_neighbor)

        if args.has_ont:  # for GMUC+
            self.ent2ic = {}
            self.rel_uc1 = {}
            self.rel_uc2 = {}
            self.get_ontology()

    def get_tasks(self):
        """Get entity/relation id, and entity/relation number.

        Update:
            self.ent2id: Entity to id.
            self.rel2id: Relation to id.
            self.symbol2id: Entity and relation to id.
            self.args.num_ent: Entity number.
            self.args.num_rel: Relation number.
        """

        eid = -1
        rid = -1

        for k, v in self.train_tasks.items():
            if self.rel2id.get(k) is None:
                rid += 1
                self.rel2id[k] = rid
            for triple in v:
                e1, rel, e2, s = triple[0], triple[1], triple[2], triple[3]
                if self.ent2id.get(e1) is None:
                    eid += 1
                    self.ent2id[e1] = eid

                if self.ent2id.get(e2) is None:
                    eid += 1
                    self.ent2id[e2] = eid

        for k, v in self.dev_tasks.items():
            if self.rel2id.get(k) is None:
                rid += 1
                self.rel2id[k] = rid
            for triple in v:
                e1, rel, e2, s = triple[0], triple[1], triple[2], triple[3]
                if self.ent2id.get(e1) is None:
                    eid += 1
                    self.ent2id[e1] = eid

                if self.ent2id.get(e2) is None:
                    eid += 1
                    self.ent2id[e2] = eid

        for k, v in self.test_tasks.items():
            if self.rel2id.get(k) is None:
                rid += 1
                self.rel2id[k] = rid
            for triple in v:
                e1, rel, e2, s = triple[0], triple[1], triple[2], triple[3]
                if self.ent2id.get(e1) is None:
                    eid += 1
                    self.ent2id[e1] = eid

                if self.ent2id.get(e2) is None:
                    eid += 1
                    self.ent2id[e2] = eid

        for line in self.path_graph:
            e1, k, e2, s = line.strip().split()
            if self.rel2id.get(k) is None:
                rid += 1
                self.rel2id[k] = rid
            if self.rel2id.get(k + '_inv') is None:
                rid += 1
                self.rel2id[k + '_inv'] = rid
            if self.ent2id.get(e1) is None:
                eid += 1
                self.ent2id[e1] = eid

            if self.ent2id.get(e2) is None:
                eid += 1
                self.ent2id[e2] = eid

        self.args.num_ent = len(self.ent2id)
        self.args.num_rel = len(self.rel2id)

        symbol_id = {}
        i = 0
        for key in self.rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in self.ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.args.num_symbols = self.num_symbols
        self.pad_id = self.num_symbols

    def get_rel2candidates(self):
        """Get the set of rel2candidates. A candidate is an entity under a relationship.

        Update:
            self.known_rels: The set of known_rels.
            self.type2ents: The set of type2ents.
            self.rel2candidates: Obtain 1000 entities of the same type as a candidate set.
        """
        # known_rels = path_graph + train_tasks
        for line in self.path_graph:
            e1, rel, e2, s = line.strip().split()
            self.known_rels[rel].append([e1, rel, e2, s])

        for key, triples in self.train_tasks.items():
            self.known_rels[key] = triples

        all_reason_relations = list(self.known_rels.keys()) + list(self.dev_tasks.keys()) + list(self.test_tasks.keys())
        all_reason_relation_triples = list(self.known_rels.values()) + list(self.dev_tasks.values()) + list(
            self.test_tasks.values())
        assert len(all_reason_relations) == len(all_reason_relation_triples)

        # rel2candidates = {rel: [tail entity candidates]}
        if self.args.dataset_name == "nl27k":
            ents = self.ent2id.keys()
            for ent in ents:
                type_ = ent.split(':')[1]
                self.type2ents[type_].add(ent)

            for rel, triples in zip(all_reason_relations, all_reason_relation_triples):
                # all possible tail entity type
                possible_types = set()
                for triple in triples:
                    tail_type = triple[2].split(':')[1]
                    possible_types.add(tail_type)

                # all possible tail entity candiates
                candidates = []
                for tail_type in possible_types:
                    candidates.extend(self.type2ents[tail_type])

                candidates = list(set(candidates))
                if len(candidates) > 1000:
                    candidates = candidates[:1000]
                self.rel2candidates[rel] = candidates
        else:
            for rel, triples in zip(all_reason_relations, all_reason_relation_triples):
                candidates = []

                for triple in triples:
                    h, r, t, s = triple
                    candidates.append(h)
                    candidates.append(t)

                candidates = list(set(candidates))
                if len(candidates) > 1000:
                    candidates = candidates[:1000]
                self.rel2candidates[rel] = candidates

    def get_e1rel_e2(self):
        """Get the set of e1rel_e2 from all dataset.

        Update:
            self.e1rel_e2: The set of e1rel_e2.
        """
        task_triples = list(self.train_tasks.values()) + list(self.dev_tasks.values()) + list(self.test_tasks.values())
        for task in task_triples:
            for triple in task:
                e1, rel, e2, s = triple
                self.e1rel_e2[e1 + rel].append(e2)

    def get_rele1_e2(self):
        """Get the set of get_rele1_e2 from dev and test dataset.

        Update:
            self.rele1_e2: The set of rele1_e2.
        """
        tasks = list(self.dev_tasks.values()) + list(self.test_tasks.values())

        for task in tasks:
            d = ddict(list)
            for triple in task:
                h, r, t, s = triple
                d[h].append([t, s])
            self.rele1_e2[r] = d

    def get_ontology(self):
        """Get the IIC of the entity and UC of the relation.

        IIC(c) = 1-\frac{log(hypo(c)+1)}{log(n)}

        UC_r1(r)=\sum_{h \in D_r, t \in R_r} (UC_e(h) + UC_e(t))
        UC_r2(r) = |D_r| * |R_r|

        Update:
            self.ent2ic: The set of rele1_e2.
            self.rel_uc1: The set of rele1_e2.
            self.rel_uc2: The set of rele1_e2.
        """
        df_ont = pd.read_csv(self.args.data_path + '/ontology.csv')
        pairs = []
        concept_set = set()

        for i in range(len(df_ont)):
            if df_ont.at[i, 'rel'] == 'is_A':
                c1 = df_ont.at[i, 'h']
                c2 = df_ont.at[i, 't']
                pairs.append((c1, c2))  # c1是c2的子类
                concept_set.add(c1)
                concept_set.add(c2)

        # Extract the range and domain of the relationship
        # Get all the relationships
        relation_set = set()
        df_domain = df_ont[df_ont.rel == 'domain']
        for i in df_domain.index:
            e = df_domain.at[i, 'h']
            relation_set.add(e)
        # Extract entity types
        entity_set = concept_set.difference(relation_set)

        # Calculate the sub-words for each entity type
        hypo_dict = dict()  # The next subordinate word of the entity
        for i in entity_set:
            hypo_list = []
            # Look for the subterfuge of this entity
            for pair in pairs:
                if pair[1] == i:
                    hypo_list.append(pair[0])
            hypo_dict[i] = hypo_list

        real_hypo_dict = copy.deepcopy(hypo_dict)
        for i in entity_set:
            for j in real_hypo_dict[i]:
                real_hypo_dict[i].extend(real_hypo_dict[j])
            real_hypo_dict[i] = list(set(real_hypo_dict[i]))

        # Calculate the IC value of the entity
        ent_type_ic = dict()
        for i in entity_set:
            ent_type_ic[i] = 1 - math.log(len(real_hypo_dict[i]) + 1) / math.log(292)

        ent2uc = ddict(float)
        rel2uc1 = ddict(float)  # uc(h)+uc(t)
        rel2uc2 = ddict(float)  # |domain|*|range|

        ent2type = {}
        df_type = df_ont[df_ont.rel == 'type']
        for i in df_type.index:
            h = df_type.at[i, 'h']
            t = df_type.at[i, 't']
            ent2type[h] = t

        ent = self.ent2id.keys()
        for i in ent:
            type_ = ent2type[i]
            ent2uc[i] = 1 - ent_type_ic[type_]
        ent_uc = list(ent2uc.values())
        mu = np.mean(ent_uc)
        sigma = np.std(ent_uc)
        for k, v in ent2uc.items():
            ent2uc[k] = (v - mu) / sigma
        self.ent2ic = ent2uc

        # Calculate the UC value of the relation
        for k, v in self.known_rels.items():
            rel = k
            domain_ = set()
            range_ = set()
            for _ in v:
                h, r, t, s = _
                h_type = h.split(':')[1]
                t_type = t.split(':')[1]
                domain_.add(h_type)
                range_.add(t_type)
            ic_sum = 0
            count = 0
            for i in domain_:
                for j in range_:
                    count += 1
                    ic_sum += (1 - ent_type_ic['concept:' + i] + 1 - ent_type_ic['concept:' + j]) / 2

            rel2uc1[rel] = ic_sum / count
            rel2uc2[rel] = count

        # normalization
        rel_uc1 = np.array(list(rel2uc1.values()))
        rel_uc2 = np.array(list(rel2uc2.values()))
        mu1, mu2 = np.mean(rel_uc1), np.mean(rel_uc2)
        sigma1, sigma2 = np.std(rel_uc1), np.std(rel_uc2)
        for k, v in rel2uc1.items():
            rel2uc1[k] = (v - mu1) / sigma1
        for k, v in rel2uc2.items():
            rel2uc2[k] = (v - mu2) / sigma2

        self.rel_uc1 = rel2uc1
        self.rel_uc2 = rel2uc2

    def build_graph(self, max_=50):
        """Build the graph according to path_graph.

        Update:
            self.connections: The set of connections.
            self.e1_rele2: The set of e1_rele2.
            self.e1_degrees: The set of e1_degrees.
        """

        self.connections = (np.ones((self.args.num_ent, max_, 3)) * self.pad_id)
        self.e1_rele2 = ddict(list)
        self.e1_degrees = ddict(int)

        with open(self.args.data_path + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2, s = line.strip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2], float(s)))
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1], float(s)))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            # Take out a neighbor of a head entity
            neighbors = self.e1_rele2[ent]
            # The number of this neighbor is limited only to max_
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]  # relation
                self.connections[id_, idx, 1] = _[1]  # tail entity
                self.connections[id_, idx, 2] = _[2]  # confidence
        return degrees


class GMUCBaseSampler(GMUCData):
    """Traditional GMUC random sampling mode.
    """
    def __init__(self, args):
        super().__init__(args)

    def generate_false(self, query_triples, candidates):
        """Generate false triples.

        Args:
            query_triples: Query set.
            candidates: All entities that belong to the relationship.

        Returns:
            false_pairs: False triples.(confidence = 0)
            false_left: The head entity of false triples.
            false_right: The tail entity of false triples.
        """
        false_pairs = []
        false_left = []
        false_right = []

        for triple in query_triples:
            for i in range(self.args.num_neg):
                e_h = triple[0]
                rel = triple[1]
                e_t = triple[2]
                while True:
                    noise = random.choice(candidates)
                    if (noise not in self.e1rel_e2[e_h + rel]) and noise != e_t:
                        break
                false_pairs.append([self.symbol2id[e_h], self.symbol2id[noise], 0.0])  # (h, t, s)
                false_left.append(self.ent2id[e_h])
                false_right.append(self.ent2id[noise])

        return false_pairs, false_left, false_right

    def get_train(self):
        return [{k: v} for k, v in self.train_tasks.items()]

    def get_valid(self):
        return [{k: v} for k, v in self.dev_tasks.items()]

    def get_test(self):
        return [{k: v} for k, v in self.test_tasks.items()]
