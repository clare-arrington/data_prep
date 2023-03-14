#%%
from tqdm import tqdm
import pandas as pd
import spacy 
import pickle
import re

## Helper file for run_get_data; contains the called functions

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

#%%
## Load corpus and preprocess sentences
def get_lines(corpus_path, delimiter):
    with open(corpus_path) as fin:
        lines = fin.read()
        lines = re.split(delimiter, lines)
    lines = [line.lower().strip() for line in lines]
    print(f'\t{len(lines):,} sentences pulled')
    lines = list(set(lines))
    print(f'\t{len(lines):,} sentences left after duplicates removed')

    return lines

## Find every target and index its location information
def index_data(sent_idx, word_idx, lines, config): 
    sentence_data = []
    word_data = []

    for line in tqdm(lines):
        if line == '':
            continue
        if config['pattern'] is not None:
            filtered_words = re.findall(config['pattern'], line)
        else:
            filtered_words = line.split(' ')

        # print(filtered_words)
        word_index_sent = []
        for i, word in enumerate(filtered_words):
            if config['POS_delimiter'] is not None: 
                try:
                    head, pos = word.split(config['POS_delimiter'])
                except:
                    if  ('__' in word or 
                        'www' in word or 
                        'http' in word or 
                        '.pdf' in word or 
                        '.doc' in word):                       
                        continue
                    ## TODO: why was this added?
                    if '_' in word:
                        words = word.split('_')
                        pos = words.pop()
                        head = words.pop()
                        for word in words:
                            word_index_sent.append(word)
                    else:
                        word_index_sent.append(word)
                    continue
            else:
                head = word
                pos = None

            ## Skip certain words
            if  (len(head) < 2    or 
                head in config['stopwords'] or 
                pos not in config['POS_labels']):
                word_index_sent.append(word)
                continue

            index = word_idx[word]
            word_idx[word] += 1
            word_index = f'{word}.{index}'

            # Construct the context around the target word
            pre = ' '.join(filtered_words[:i])
            pre = trim(pre)
            post = ' '.join(filtered_words[i + 1:])
            post = trim(post, pre=False)

            formatted_sent = (pre, word, post)
            length = len(pre) + len(post)

            ## Store all the relevant target information
            target_info = [word_index, word, formatted_sent, length, sent_idx]
            word_data.append(target_info)
            word_index_sent.append(word_index)

        ## Store all the relevant sentence information
        sentence_info = [sent_idx, line, word_index_sent]
        sentence_data.append(sentence_info)
        sent_idx += 1

    return sentence_data, word_data, sent_idx

## Cleanup and store indexed data
def postprocess_data(word_data, sentence_data, word_idx, cutoff_count=50):

    ## Convert to dataframes
    word_data = pd.DataFrame(word_data, columns=[
        'word_idx', 'target', 'formatted_sent', 'length', 'sent_idx'])
    sentence_data = pd.DataFrame(sentence_data, columns=[
        'sent_idx', 'sent', 'word_idx_sent'])

    ## Filter out indexed words that are too infrequent
    ## TODO: the indexed word will still be in the modified sentence.
    ## Need to brainstorm a clean way to remove but it doesn't harm anything
    vc = word_data.target.value_counts()
    drop_words = vc[vc <= cutoff_count].index
    drop_rows = word_data.target.isin(drop_words)
    word_data = word_data[~drop_rows]
    for word in drop_words:
        word_idx.pop(word, None)

    return sentence_data, word_data

## Output the final dataframes
def save_data(sentence_data, word_data, output_path, corpus_name):
    print('\nBeginning to save sentence data')
    sentence_data.set_index('sent_idx', inplace=True)
    sentence_data.to_pickle(
        f'{output_path}/{corpus_name}_indexed_sentences.pkl',
        protocol=pickle.HIGHEST_PROTOCOL)
    print('Sentence data saved!')

    print('\nBeginning to save word data')
    word_data.set_index('word_idx', inplace=True)
    # word_data.drop_duplicates(inplace=True)
    word_data.to_pickle(
        f'{output_path}/{corpus_name}_indexed_words.pkl', 
        protocol=pickle.HIGHEST_PROTOCOL)
    print('Word data saved!\n\n')
# %%
## TODO: fixing for date is bad; but leave it for now
## Is this old?
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
