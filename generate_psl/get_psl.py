from utils.setup_parse import setup_parser
from utils.tools import load_config
import csv
import os
import itertools
from collections import defaultdict

"""
    The purpose of this program is to convert tsv files into obs.txt files, which will be used as input for the PSL program.
    The processed txt file will be placed in folder -- data_for_psl by default.  
"""


def main(arg_path):
    print('Start Converting File Format')
    args = setup_parser()  # set the parse
    args = load_config(args, arg_path)

    data_path = args.data_path  # the path of the file that will be processed as obs.txt

    # import the data
    data = []
    with open(data_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            data.append(row)

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
    # if args.origin_data == True, generate a tmp relation2id and entity2id files for data, and it will be used in infer_psl
    if args.origin_data == True:
        quadruples_list = data.copy()
        entity_to_id = {}
        relation_to_id = {}
        new_quadruples = []
        for quadruple in quadruples_list:
            head_entity, relation, tail_entity, confidence = quadruple
            # deal with head entities and tail entities
            if head_entity not in entity_to_id:
                entity_to_id[head_entity] = len(entity_to_id) + 1
            if tail_entity not in entity_to_id:
                entity_to_id[tail_entity] = len(entity_to_id) + 1
            # deal with relations
            if relation not in relation_to_id:
                relation_to_id[relation] = len(relation_to_id) + 1
            # get new quadruples with tmp ids
            new_quadruple = [
                str(entity_to_id[head_entity]),
                str(relation_to_id[relation]),
                str(entity_to_id[tail_entity]),
                confidence
            ]
            new_quadruples.append('\t'.join(new_quadruple))
        print(relation_to_id)
        dealed_data_folder = args.dealed_data_folder  # user need to provide dealed_data_folder when using raw data

        if not os.path.exists(dealed_data_folder):
            os.makedirs(dealed_data_folder)
        # output entity_id
        entity_id_file_path = os.path.join(dealed_data_folder, 'entity_id.tsv')
        with open(entity_id_file_path, 'w') as entity_id_file:
            for entity, entity_id in entity_to_id.items():
                entity_id_file.write(f"{entity}\t{entity_id}\n")
        # output relation_id
        relation_id_file_path = os.path.join(dealed_data_folder, 'relation_id.tsv')
        with open(relation_id_file_path, 'w') as relation_id_file:
            for relation, relation_id in relation_to_id.items():
                relation_id_file.write(f"{relation}\t{relation_id}\n")
        # output quadruples represented by id (data.tsv)
        processed_quadruples_file_path = os.path.join(dealed_data_folder, 'data.tsv')
        with open(processed_quadruples_file_path, 'w') as processed_quadruples_file:
            for quadruple in new_quadruples:
                processed_quadruples_file.write(f"{quadruple}\n")

    data_dict = {} # use dict to store samples
    for sublist in data:
        key = sublist[1]  # use relation as key
        value = [sublist[0], sublist[2], sublist[3]]
        if key in data_dict:
            data_dict[key].append(value)
        else:
            data_dict[key] = [value]
    print(data_dict.keys())

    duplicate_values = {}
    for key, values in data_dict.items():
        seen = set()
        duplicates = []
        for value in values:
            value_tuple = tuple(value)
            if value_tuple[0:2] in seen:
                duplicates.append(value_tuple)
            else:
                seen.add(value_tuple[0:2])
        if duplicates:
            duplicate_values[key] = duplicates
    if duplicate_values:
        # If there are duplicate samples, output these samples
        print("There are duplicate samples")
        if args.Auto_remove_duplicate_triples == False:
            print("The duplicate samples are as follows:")
            print(duplicate_values)
            raise Exception("There are Duplicate Triples!")
        else:
            # If set Auto_remove_duplicate_triples as true, we will automatically deduplicate the samples
            print("Remove duplicate triples automatically.")
            for key, values in data_dict.items():
                unique_dict_temp = {}
                for element in data_dict[key]:
                    key_unique = tuple(element[:2])
                    unique_dict_temp[key_unique] = element
                data_dict[key] = list(unique_dict_temp.values())
    # create a folder to store processed data, these data will be used in infer_psl
    folder_name = args.out_data_path
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # create corresponding txt files for each relation.
    for key, values in data_dict.items():
        if args.relation_id_path == False:
            file_name = os.path.join(folder_name, f"{key}_obs.txt")
            with open(file_name, 'w') as file:
                for value in values:
                    file.write('\t'.join(value) + '\n')
        else:
            result_dict = {}
            if args.origin_data == False:
                relation_id_path = args.relation_id_path
            else:
                relation_id_path = os.path.join(args.dealed_data_folder, "relation_id.tsv")
            with open(relation_id_path) as tsv_file:
                reader = csv.reader(tsv_file, delimiter='\t')
                for row in reader:
                    result_dict[row[1]] = row[0]
                    result_dict[row[0]] = row[1]
            if args.origin_data == False:
                file_name = os.path.join(folder_name, f"{result_dict[key]}_obs.txt")
            else:
                file_name = os.path.join(folder_name, f"{key}_obs.txt")
            with open(file_name, 'w') as file:
                for value in values:
                    file.write('\t'.join(value) + '\n')

if __name__ == "__main__":
    main(arg_path='config/nl27k.yaml')