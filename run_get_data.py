#%%
from get_data import index_data, postprocess_data, \
                     get_lines, save_data
from nltk.corpus import stopwords as stops
from dotenv import dotenv_values
from collections import defaultdict
from pathlib import Path
import pickle
import json
import re

def setup_params(config):
    if 'pattern' in config:
        config['pattern'] = re.compile(config['pattern'])
    else:
        config['pattern'] = None

    if 'language' in config:
        lang = config['language']
        config['stopwords'] = stops.words(lang)
        # if 'no' in stopwords:
        #     stopwords.remove('no')
    else:
        config['stopwords'] = stops.words('english')

    if 'delimiter' in config:
        config['delimiter'] = re.compile(config['delimiter'])
    else:
        config['delimiter'] = re.compile('\n')

    ## If targets are POS labeled, specify how they're split
    ## Ex. head.V or head_V
    if 'POS_delimiter' not in config or 'POS_labels' not in config:
        config['POS_delimiter'] = None
        config['POS_labels'] = [None]

## Primary file for data_prep 
## Only variable you need to change; could also be a terminal prompt 
config_path = 'configs/semeval.json' 

## Establish paths 
with open(config_path) as d:
    config = json.load(d)
data_path = dotenv_values(".env")['data_path']
data_path += f"corpus_data/{config['dataset_name']}"
corpora_path = f'{data_path}/corpora'

subset_path = data_path + config['subset_path']
print(f'Results will be saved to {subset_path}\n')
Path(subset_path).mkdir(parents=True, exist_ok=True)

## Load additional parameters from the config
setup_params(config)

## Parse data for every corpus listed in the config
sent_idx = 0
word_idx = defaultdict(int)
for corpus_name in config['corpora']:
    print(f"===== Prepping data for {corpus_name} corpus ====")
    corpus_path = corpora_path + f'/{corpus_name}.txt'
    lines = get_lines(corpus_path, config['delimiter'])
    sentence_data, word_data, sent_idx = index_data(
        sent_idx, word_idx, lines, config)
    print(f'{len(word_idx)} words indexed')
    print(word_idx)
    
    ## 
    sentence_data, word_data = postprocess_data(
        word_data, sentence_data, word_idx)
    print(f'{len(word_idx)} words after infrequent dropped')
    print(f'{sent_idx} sentences indexed')
    
    save_data(sentence_data, word_data, subset_path, corpus_name)

with open(f'{subset_path}/word_index_freq.pkl', 'wb') as f:
    pickle.dump(word_idx, f)

print('All done!')
#%%
## Code for seeing the target counts  
# t_path = f'{data_path}/targets/targets.txt'
# with open(t_path) as f:
#     targets = f.read().strip().split('\n')

# vc = word_data.target.value_counts()
# for t in targets:
#     print(t, end=', ')
#     print(vc[t], end=', ')
#     print()
    # print(vc[t[:-3]])

#%%
