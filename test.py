### Import Packages ###
import pandas as pd
import string
from cleanco import cleanco
from collections import Counter
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, file_name):
        self.missing_values = ["n/a", "na", "--"]
        self.data = pd.read_csv(file_name, na_values=self.missing_values).dropna()        

class Parse: 
    def __init__(self, df, stopwords):
        self.data = df
        self.stopwords = stopwords
        self.keywords = None

    def _clean_name(self, name):
        name = name.translate(str.maketrans('', '', string.punctuation))
        name = cleanco(name).clean_name()
        name = name.lower()

        return name
        

    def clean_dataset(self, name_col, new_col_name='cleaned_name'):
        self.data[new_col_name] = self.data[name_col].apply(lambda x: self._clean_name(x))
        print(self.data.head())
        self.data[new_col_name] = self.data[new_col_name].apply(lambda x: ' '.join([word for word in x.split(" ") if word not in self.stopwords]))



class Match:
    def __init__(self, df):
        self.data = df
        self.counter = Counter()
        self.keywords = []

    def _get_keywords(self, len_limit=3, col_name='cleaned_name'):
        for name in self.data[col_name].unique():
            self.counter.update(str(name).split(" "))
        self.keywords = [word for (word,_) in self.counter.most_common(30)]
        self.keywords = [word for word in self.keywords if len(word)>len_limit]

    def _check_keywords(self, name):
        is_key_in_name = True
        for word in self.keywords:
            if word in name:
                is_key_in_name = False
        
        return is_key_in_name

    def match_levenshtein(self, start_idx, end_idx, parsed_col='cleaned_name', alias_col='alias', score_col='score'):
        for i in range(start_idx, end_idx+1):
            if pd.isna(self.data[alias_col].iloc[i]):
                self.data[alias_col].iloc[i] = self.data[parsed_col].iloc[i]
                self.data[score_col].iloc[i] = 100

            for j in range(i+1, end_idx+1):
                if pd.isna(self.data[alias_col].iloc[j]):
                    match_score = fuzz.token_sort_ratio(self.data[parsed_col].iloc[i],self.data[parsed_col].iloc[j])
                    if not self._check_keywords(self.data[parsed_col].iloc[j]):
                        match_score -= 20
                    if (match_score > 80):
                        self.data[alias_col].iloc[j] = self.data[alias_col].iloc[i]
                        self.data[score_col].iloc[j] = match_score
                        
    
if __name__=='__main__':

    data = Dataset('Test_Names.csv').data

    df_match = pd.DataFrame(columns=['group', 'original', 'alias','score'])
    names = data['domestic'].unique()
    names.sort()
    df_match['original'] = names
    df_match['group'] = df_match['original'].apply(lambda x: x[0])
    
    stopwords = ['pvt', 'exp', 'ltd', 'lt', 'co', 'corp']
    Parse(df_match, stopwords).clean_dataset('original')
    df_match = Parse(df_match, stopwords).data

    sort_groups = df_match['group'].unique()
    for group in sort_groups:
        curr_group = df_match.groupby(['group']).get_group(group)
        start_idx = curr_group.index.min()
        end_idx = curr_group.index.max()

        Match(df_match).match_levenshtein(start_idx, end_idx)

    df_match = df_match.assign(uid=(df_match['alias'].astype('category').cat.codes))

    df_match['domestic'] = df_match['original']

    df_final = pd.merge(data['domestic'], df_match[['domestic', 'alias', 'uid']], on='domestic')

    df_final.to_csv("final_df.csv")









