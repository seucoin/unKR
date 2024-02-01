# -*- coding: utf-8 -*-
import argparse
import os
import yaml
import pytorch_lightning as pl

"""
    if user set args.origin_data == True, it indicates that the data imported by the user is raw data, 
    and these data entities and relationships are not represented by pre processed digital IDs.
    In this case, the user only needs to provide the specific path of the data_path in the yaml file, 
    and then set relation_id_path and entity_id_path directly to False.

    if user set args.origin_data == False, This indicates that the entities and relationships of the data 
    imported by the user have already been processed in the form of IDs. 
    In this case, since the PSL program cannot handle purely numerical entities or relationships, 
    the user needs to provide additional relation_id_path and entity_id_path to provide mapping between name and ID.
    In addition, when writing PSL rules in yaml, the real names of entities and relationships must also be used instead of ID numbers.

    For a clearer understanding of parameter setting methods, you can refer to the example config files (the path is unKR/generate_psl/config) we provide.

    If the original entities and relationships in the knowledge graph are represented purely by numbers, 
    we suggest that you construct a set of mapping relationships between numbers and characters, 
    and then convert the knowledge graph into a knowledge graph containing characters for entities and relationships. 
    You can refer to the case where args. origin_data == False to provide the converted KG, entity_id_path, and relation_id_path, so that you can use the PSL program
"""

def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Basic arguments
    parser.add_argument('--data_path', default="data/train.tsv", type=str, help='Path of the Original File.')
    parser.add_argument('--relation_id_path', default='data/relation_id.tsv', type=str,
                        help='Path of the the map of relation to id. (Only for ID data)')
    parser.add_argument('--out_data_path', default="./data_for_psl", type=str, help='Path to Store the Generated Files, which are Required for PSL Inference.')
    # parser.add_argument('--Generate_Full_Target', default=False, type=bool, help='Generate Pairwise Combinations of All Entities in Each Relationship and Generate Target Files.')
    parser.add_argument('--Auto_remove_duplicate_triples', default=False, type=bool, help='If set to True, automatically remove duplicate triplets from the training set.')
    parser.add_argument('--psl_model_name', default='demo', type=str, help='The name of PSL model.')
    parser.add_argument('--inference_method', default='LazyMPEInference', type=str, help='The method of PSL inference.')
    parser.add_argument('--predicates', default=[{'name': 'BINDING', 'size': 2, 'closed': False}], type=list, help='List of predicates')
    parser.add_argument('--rules', default=["BINDING(P1, P2) & BINDING(P2, P3) -> BINDING(P1, P3) ."], type=list, help='List of rules')
    parser.add_argument('--use_target', default=False, type=bool, help='to indicate whether you need to use target files. If you use the LazyMPEInformation method, you do not need to provide target files')
    parser.add_argument('--inferred_result_path', default='test', type=str, help='Path of the folder where the inferred result is stored')
    parser.add_argument('--out_softlogic_filename', default='softlogic.tsv', type=str, help='File name of the tsv form of inferred result')

    return parser
