#%%
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import spacy 
import pickle
import re

## Trim the sentence around a term, either the section before or after
## Similar approach for both, just differently added
def trim(old_sent, pre=True, cutoff=100):
    new_sent = ''
    words = old_sent.split()
    if pre:
        words = reversed(words)
    
    for word in words:
        if pre:
            new_sent = f'{word} {new_sent}'
        else:
            new_sent = f'{new_sent} {word}'

        if len(new_sent) > cutoff:
            break

    return new_sent

def parse_sentences(lines, pattern):

    sent_idx = 0
    sentence_data = []
    word_data = []
    word_idx = defaultdict(int)

    for line in tqdm(lines):
        line = line.lower().strip()
        filtered_words = re.findall(pattern, line)

        word_index_sent = []
        for i, word in enumerate(filtered_words):

            index = word_idx[word]
            word_idx[word] += 1

            ## TODO: this is just for semeval. Needed?
            word, *etc = word.split('_')
            word_index = f'{word}.{index}'

            # Get the words before and after the current target
            pre = ' '.join(filtered_words[:i])
            post = ' '.join(filtered_words[i + 1:])

            pre = trim(pre)
            post = trim(post, pre=False)

            formatted_sent = (pre, word, post)
            length = len(pre) + len(post)

            target_info = [word_index, word, formatted_sent, length, sent_idx]
            word_data.append(target_info)

            word_index_sent.append(word_index)

        sentence_info = [sent_idx, line, word_index_sent]
        sentence_data.append(sentence_info)
        sent_idx += 1

    return sentence_data, word_data

#%%
def index_word_data(corpus_path, pattern): 
    
    ## TODO: need to go through the preprocess step here if needed
    # data = pd.read_pickle(data_path)
    # # data['processed_sentence'] = data['processed_sentence'].apply(literal_eval)
    # print(f'\nAll Sents: {len(data):,}')
    # data.drop_duplicates(subset=['sentence'], inplace=True)
    # print(f'All Sents after duplicates removed: {len(data):,}')

    with open(corpus_path) as fin:
        lines = [line.lower().strip() for line in fin.readlines()]
        print(f'\t{len(lines):,} sentences pulled')
        lines = list(set(lines))
        print(f'\t{len(lines):,} sentences left after duplicates removed')

    sentence_data, word_data = parse_sentences(lines, pattern)
    
    ## Convert to dataframes
    word_data = pd.DataFrame(word_data, columns=[
        'word_idx', 'target', 'formatted_sent', 'length', 'sent_idx'])
    sentence_data = pd.DataFrame(sentence_data, columns=[
        'sent_idx', 'sent', 'word_idx_sent'])

    return sentence_data, word_data

def save_data(sentence_data, word_data, output_path, corpus_name):
    sentence_data.set_index('sent_idx', inplace=True)
    sentence_data.to_pickle(f'{output_path}/{corpus_name}_indexed_sentences.pkl')
    print('\nSentence data saved!')

    word_data.set_index('word_idx', inplace=True)
    word_data.to_pickle(f'{output_path}/{corpus_name}_indexed_words.pkl')
    print('Word data saved!\n\n')
# %%
## TODO: fixing for date is bad; but leave it for now
def preprocess_data(docs, corpus_name, path):
    nlp = spacy.load("en_core_web_sm")

    sentences = []
    sent_id = 0
    processed = nlp.pipe(docs.content, batch_size=50, 
        n_process=1, disable=["ner", "textcat"])
    for date, doc in tqdm(zip(docs.date, processed), total=len(docs)):
        ## TODO: this is unideal; not splitting on \n
        for sent in doc.sents:
            ## TODO: fix for more than covid-19 
            p_sent = []
            for token in sent:
                t = token.text.lower()
                if t.isalpha() == True:
                    p_sent.append(token.lemma_.lower())
                elif ('covid' in t):
                    p_sent.extend(re.findall(r'^[a-z]+', t))

            # p_sent = [token.lemma_.lower() for token in sent 
            #         if token.text.isalpha() == True or ('covid' in token.text)]
            
            if p_sent == []:
                continue
            sent_id += 1
            sent_info = [sent_id, corpus_name, str(sent), p_sent, date]
            sentences.append(sent_info)
            
    sentences = pd.DataFrame(sentences,
                columns=['sent_id', 'corpus', 'sentence', 'processed_sentence', 'date']
                )
    sentences.set_index('sent_id', inplace=True)
    sentences.to_pickle(path)

