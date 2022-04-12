#%%
from get_data import index_word_data, save_data
from dotenv import dotenv_values
from pathlib import Path
import json
import re

## Only variable you need to change; could also be a terminal prompt 
config_path = 'configs/semeval.json'
with open(config_path) as d:
    config = json.load(d)

data_path = dotenv_values(".env")['data_path']
data_path += f"corpus_data/{config['dataset_name']}"
corpora_path = f'{data_path}/corpora'

subset_path = data_path + config['subset_path']
print(f'Results will be saved to {subset_path}\n')
Path(subset_path).mkdir(parents=True, exist_ok=True)

for corpus_name, target_file in config['corpora_target_file'].items():
    print(f"===== Prepping data for {corpus_name} corpus ====")
    
    pattern = config['pattern']
    pattern = re.compile(pattern)
    corpus_path = corpora_path + f'/{corpus_name}.txt'
    sentence_data, word_data = index_word_data(corpus_path,  pattern)

    save_data(sentence_data, word_data, subset_path, corpus_name)

print('All done!')
#%%
