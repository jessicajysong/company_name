import pandas as pd
import glob
import os
import collections
import networkx as nx
from thefuzz import fuzz
from tqdm import tqdm
from string_grouper import match_strings

def create_ids(df):
    ids = df[['interim_name']].apply(tuple, axis=1)
    df['id'] = pd.factorize(ids)[0]
    print('Unique IDs: {}'.format(df['id'].nunique()))
    df['our_id'] = df['country_iso3'] + df['id'].astype(str)
    return df

def get_first_word(string):

    l = string.split()
    if len(l) < 2:
        return string
    temp_word = l[0]
    if len(temp_word) > 1:
        return temp_word
        # return (' ').join(l[1:])
    else:
        return '{} {}'.format(temp_word, l[1])

def is_similar_neigh(n1, n2, df, type):
    """
        Levenstein similarity
    """

    name1 = df.iloc[n1][type]
    name2 = df.iloc[n2][type]

    r = fuzz.ratio(name1, name2)
    # s1 = set(x for x in name1.split() if not x.isdigit() and len(x) > 1)
    # s2 = set(x for x in name2.split() if not x.isdigit() and len(x) > 1)

    if r > 80:
        return True
    else:
        return False


def correct_neighbors(df, type):
    """
        Sort dataframe and merge similar consecutive words
    """

    n1 = 1
    n2 = 0
    end = False

    with tqdm(total=len(df)) as pbar:
        while True:

            while df.iloc[n1]['ID'] == df.iloc[n2]['ID']:
                n1 += 1
                pbar.update(1)
                if n1 >= len(df):
                    end = True
                    break

            if end:
                break

            n2 = n1 - 1

            similar = is_similar_neigh(n1, n2, df, type)

            if similar:
                # change ID
                df.loc[df['ID'] == df.iloc[n2]['ID'], 'ID'] = df.iloc[n1]['ID']
                df.loc[df['ID'] == df.iloc[n2]['ID'], 'cleaned_name'] = df.iloc[n1]['cleaned_name']
            else:
                n2 = n1

    return df

def get_longest_name(serie):
    frequency = collections.Counter(serie.to_list())
    max_freq = max(frequency.values())

    max_name = None
    for name, freq in frequency.items():
        if freq == max_freq:
            max_name = name

    for x in serie:
        if frequency[x] < max_freq:
            continue
        if len(str(x)) > len(max_name):
            max_name = str(x)
    return max_name

def read_country_files(data_path):

    ### Gather all countries dataset and save into a dictionary
    file_list_exp = glob.glob(os.path.join(data_path, "country", "*.csv"))
    df_dict = {}
    count = 0
    l = 0
    for file_path in file_list_exp:

        df = pd.read_csv(file_path, index_col=None, encoding="utf-8", header=0, dtype='str')
        df = df.dropna()
        df = df.drop_duplicates()
        l += len(df)
        df_dict[count] = df
        count += 1

    print('Len after reading: {}'.format(l))
    return df_dict

def match_names(df, iso, base_sim=0.7, first_w_sim=0.9):

    print('Match names...')
    matches1 = match_strings(master=df['interim_name'],
                             master_id=df['our_id'],
                             min_similarity=base_sim,
                             n_blocks='auto')
    matches1['similarity'] = matches1['similarity'].round(decimals=3)
    matches1 = matches1.drop(columns=['left_index', 'right_index'], axis=1).drop_duplicates()

    # print('Match first word...')
    # matches2 = match_strings(master=df['first_word'],
    #                          master_id=df['our_id'],
    #                          min_similarity=first_w_sim,
    #                          n_blocks='auto')
    # matches2['similarity'] = matches2['similarity'].round(decimals=3)
    # matches2 = matches2.drop(columns=['left_index', 'right_index'], axis=1).drop_duplicates()

    # print('Merge matches...')
    # matches = pd.merge(matches1,
    #                    matches2,
    #                    left_on=['left_our_id', 'right_our_id'],
    #                    right_on=['left_our_id', 'right_our_id'])

    matches = matches1

    nodupes = matches[matches['left_our_id'] != matches['right_our_id']]
    nodupes = nodupes.sort_values(by='similarity').reset_index()

    if len(nodupes) > 0:
        G = nx.Graph()
        for id1, id2 in zip(nodupes['left_our_id'], nodupes['right_our_id']):
            G.add_edge(id1, id2)

        id_mapping = {}
        sub_graphs = nx.connected_components(G)
        for sg in sub_graphs:
            nodes = [int(x[3:]) for x in sg]
            m = min(nodes)
            for x in nodes:
                if x != m:
                    id_mapping[f'{iso}{x}'] = f'{iso}{m}'

        matches['left_our_id'] = matches['left_our_id'].apply(
            lambda x: id_mapping[x] if x in id_mapping else x)
        matches['right_our_id'] = matches['right_our_id'].apply(
            lambda x: id_mapping[x] if x in id_mapping else x)

        assert len(matches[matches['left_our_id'] == matches['right_our_id']]) > 0

    dft1 = matches[['left_interim_name', 'left_our_id']].rename(columns={
        'left_interim_name': 'interim_name',
        'left_our_id': 'ID'
    })
    dft2 = matches[['right_interim_name', 'right_our_id']].rename(columns={
        'right_interim_name': 'interim_name',
        'right_our_id': 'ID'
    })

    dft = pd.concat([dft1, dft2])
    dft = dft.drop_duplicates().reset_index()

    return dft

def break_big_ids(df, iso, min_sim=0.65):
    #TODO improve method

    df['n'] = df.groupby('ID').transform('count')['domestic']
    big_id_n = 20
    df_big = df[df['n'] >= big_id_n]
    df_small = df[df['n'] < big_id_n]
    df_big = df_big.drop('n', axis=1)
    df_small = df_small.drop('n', axis=1)

    ids = df_big[['preprocessed_name']].apply(tuple, axis=1)
    df_big['new_id'] = pd.factorize(ids)[0]
    df_big['new_id'] = df_big['new_id'].astype(str)

    matches = match_strings(master=df_big['domestic'],
                             master_id=df_big['new_id'],
                             min_similarity=min_sim,
                             n_blocks='auto')
    matches['similarity'] = matches['similarity'].round(decimals=3)
    matches = matches.drop(columns=['left_index', 'right_index'], axis=1).drop_duplicates()
    matches = pd.merge(matches, df_big[['ID', 'new_id']].rename(columns={'ID': 'ID_left'}),
                        left_on='left_new_id', right_on='new_id', how='left')
    matches = pd.merge(matches, df_big[['ID', 'new_id']].rename(columns={'ID': 'ID_right'}),
                        left_on='right_new_id', right_on='new_id', how='left')

    full_id_list = list(df_small['ID'].unique())
    full_id_list = [int(x[3:]) for x in full_id_list]
    free_ids = set(range(1000000)).difference(full_id_list)

    id_mapping = {}
    # list big ID one by one
    for fid in tqdm(df_big['ID'].unique()):
        matches3_temp = matches[(matches['ID_left'] == fid) &
                                 (matches['ID_right'] == fid)]

        G = nx.Graph()
        for id1, id2 in zip(matches3_temp['left_new_id'], matches3_temp['right_new_id']):
            G.add_edge(id1, id2)

        sub_graphs = nx.connected_components(G)
        for sg in sub_graphs:
            m = min(free_ids)
            free_ids.remove(m)
            for x in sg:
                id_mapping[x] = f'{iso}{m}'

    matches['left_new_id'] = matches['left_new_id'].apply(
        lambda x: id_mapping[x] if x in id_mapping else x)
    matches['right_new_id'] = matches['right_new_id'].apply(
        lambda x: id_mapping[x] if x in id_mapping else x)

    dft1 = matches[['left_domestic', 'left_new_id']].rename(columns={
        'left_domestic': 'domestic',
        'left_new_id': 'new_ID'
    })
    dft2 = matches[['right_domestic', 'right_new_id']].rename(columns={
        'right_domestic': 'domestic',
        'right_new_id': 'new_ID'
    })

    dft = pd.concat([dft1, dft2])
    dft = dft.drop_duplicates().reset_index()

    df_big = pd.merge(df_big, dft, left_on='domestic', right_on='domestic', how='left')
    df_big = df_big[['domestic', 'preprocessed_name', 'new_ID']].rename(columns={
        'new_ID': 'ID'
    })

    df_big['cleaned_name'] = df_big.groupby('ID')['preprocessed_name'].transform(get_longest_name)
    print('Null values: {}'.format(df_big.isnull().values.any()))

    df = pd.concat([df_big, df_small])
    return df


def run_name_cleaning(type, data_path, domestic_iso, preprocessed_names, base_sim, first_w_sim, big_id_sim):

    # TODO break into two functions run_name_cleaning
    if type == 'domestic':

        df = pd.read_csv(os.path.join(data_path, preprocessed_names))
        df = df.dropna().drop_duplicates()
        df['interim_name'] = df['interim_name'].astype(str)


        df['country_iso3'] = domestic_iso
        df = create_ids(df)
        df['first_word'] = df['interim_name'].apply(lambda x: get_first_word(x))
        dft = match_names(df, domestic_iso, base_sim=base_sim, first_w_sim=first_w_sim)
        df = pd.merge(df, dft, left_on='interim_name', right_on='interim_name', how='left')
        df = df[['domestic', 'interim_name', 'ID']].rename(columns={
            'interim_name': 'preprocessed_name'
        })

        df['cleaned_name'] = df.groupby('ID')['preprocessed_name'].transform(get_longest_name)
        df = df.sort_values(by=['domestic'])
        df = correct_neighbors(df, type)

        # TODO improve break big IDs
        df = break_big_ids(df, domestic_iso, min_sim=big_id_sim)
        # TODO move path to the config
        df.to_csv(f'{domestic_iso}_Unique_DomesticNames_cleaned.csv', index=False)

    elif type == 'foreign':
        df_dict = read_country_files(data_path)

        df_agg = pd.DataFrame()
        for key in df_dict.keys():
            df = df_dict[key]
            df = df.rename(columns = {'foreigncountry_iso3': 'country_iso3'})
            iso = df['country_iso3'].to_list()[0]
            country_name = df['foreigncountry_cleaned'].to_list()[0]

            print(">>>START WORKING ON {} DATA...".format(country_name))
            df['interim_name'] = df['interim_name'].astype(str)
            df = create_ids(df)
            df['first_word'] = df['interim_name'].apply(lambda x: get_first_word(x))

            dft = match_names(df, iso)
            df = pd.merge(df, dft, left_on='interim_name', right_on='interim_name', how='left')
            df = df[['foreign', 'interim_name', 'ID']].rename(columns={
                'interim_name': 'preprocessed_name'
            })

            df['cleaned_name'] = df.groupby('ID')['preprocessed_name'].transform(get_longest_name)
            df['country_name'] = country_name

            if len(df) > 1:
                print('Correct neighbors...')
                df = df.sort_values(by=['foreign'])
                df = correct_neighbors(df, type)

            df_agg = pd.concat([df_agg, df])

        # TODO correct paths
        df_agg.to_csv(f'{domestic_iso}_Unique_DomesticNames_cleaned.csv', index=False)
    else:
        print('Wrong type. Please use domestic or foreign.')
