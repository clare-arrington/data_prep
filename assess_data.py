#%%
from dotenv import dotenv_values
from nltk.corpus import stopwords as stops
import pandas as pd
import json

stopwords = stops.words('english')

## Only variable you need to change; could also be a terminal prompt 
config_path = 'configs/semeval.json'
with open(config_path) as d:
    config = json.load(d)

data_path = dotenv_values(".env")['data_path']
# Path looks like: .../corpus_data/d_name/subset
data_path += "corpus_data/" + config['dataset_name'] 
subset_path = data_path + config['subset_path']
target_path = data_path + '/targets/'

#%%
min_limit = 500
max_limit = 10000

data = []
for corpus_name in config['corpora_target_file']:
    word_path = subset_path + f'/{corpus_name}_indexed_words.pkl'
    word_data = pd.read_pickle(word_path)

    print('Trimming words for ' + corpus_name)
    vc = word_data.target.value_counts()
    print(f'\t{len(vc)} words')

    # Lower limit
    vc = vc[vc > min_limit]
    print(f'\t{len(vc)} words after {min_limit} min cutoff')
    
    # Upper limit 
    vc = vc[vc < max_limit]
    print(f'\t{len(vc)} words after {max_limit} max cutoff')
    
    # Stopword removal
    vc = vc[~vc.keys().isin(stopwords)]
    print(f'\t{len(vc)} words after stopwords removed\n')

    # Short word 
    vc = vc[vc.keys().str.len() > 2]
    print(f'\t{len(vc)} words after 2 letter words removed\n')

    vc.name = corpus_name
    data.append(vc)

data = pd.DataFrame(data).T
print(f'{len(data)} words in all corpora combined')

data.dropna(inplace=True)
print(f'{len(data)} words in both corpora')

targets = list(data.index)

with open(target_path + 'found_targets.txt', 'w') as f:
    for target in targets:
        print(target, file=f)

data.to_csv(target_path + 'target_counts.csv')

# %%
