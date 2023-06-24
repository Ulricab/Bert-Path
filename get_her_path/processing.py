import pandas as pd


def get_ent_id2name(ent_ids_list):
    ent_dict = {}
    for ent in ent_ids_list:
        ent_dict[ent[0]] = ent[1]

    return ent_dict


def get_ancestors(ent_id, triple_list, ent_id_dict, n):
    ancestors = []

    for i in range(n):
        for triple in triple_list:
            if triple[0] == ent_id:
                ancestors.insert(0, ent_id_dict[triple[2]])
                ent_id = triple[2]
                break
    return ancestors


def get_lower(ent_id, triple_list, ent_id_dict, n):
    lowers = []

    for i in range(n):
        for triple in triple_list:
            if triple[2] == ent_id:
                lowers.append(ent_id_dict[triple[0]])
                ent_id = triple[0]
                break
    return lowers


if __name__ == '__main__':
    ent_ids_df = pd.read_table('ent_ids_2', header=None)
    ent_ids_list = ent_ids_df.values.tolist()
    ent_dict = get_ent_id2name(ent_ids_list)

    triple_df = pd.read_table('./triples_2', header=None)
    triple_list = triple_df.values.tolist()

    path_list = []

    for ent in ent_ids_list:
        ancestors = get_ancestors(ent[0], triple_list, ent_dict, 1)
        lowers = get_lower(ent[0], triple_list, ent_dict, 0)
        print(', '.join(ancestors))
        path_ent = [ent[0], ', '.join(ancestors), ', '.join(lowers)]
        path_list.append(path_ent)

    path_df = pd.DataFrame(columns=None, data=path_list)
    path_df.to_csv('./paths_2', index=False, sep='\t', header=None)
