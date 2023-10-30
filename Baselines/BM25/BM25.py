import pickle
import pandas as pd
import itertools
from collections import defaultdict, OrderedDict
from ranx import Qrels, Run, evaluate
from functools import reduce
from tabulate import tabulate
from ast import literal_eval
import os
from tqdm import tqdm
import string
import argparse
import json
import gzip

def write_json(obj, filename):    
    
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)
        
    return

import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    
def translate_tags(a, punc):
    return [elem.translate(str.maketrans(punc)) for elem in a]

punc = {
    '+': 'plus', 
    '-': 'hyphen', 
    '.': 'dot',
    ',': 'comma',
    "'": 'apostrophe',
    '"': 'quotation',
    '?': 'question',
    '!': 'exclamation',
    '-': 'hyphen',
    '(': 'parenthesis',
    ')': 'parenthesis',
    '[': 'bracket',
    ']': 'bracket',
    '{': 'braces',
    '}': 'braces',
    ':': 'colon',
    ';': 'semicolon',
    '&': 'and',
    '|': 'bar',
    '#': 'hash',
    '@': 'at',
    '$': 'dollar',
    '/':' slash',
    '%': 'percent',
    '<': 'angular',
    '>': 'angular',
    '=': 'equal',
    '_': 'underscore',
    '*': 'asterisk'
    }
    
def terrier_query(query, lowercase=True):
    if lowercase:
        words = [word.lower() for word in query.split()]
        new_query = " ".join(words)
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    # Use the translation table to remove punctuation
    new_query = new_query.translate(translator)
    return new_query

def split_tuples(lst):

    freq = defaultdict(int) 
    scores = defaultdict(list)

    for elem in lst:
        freq[elem[0]]+=1
        scores[elem[0]].append(elem[1])

    return freq, scores

def merge_lists(a, b):

    j = 0
    i = 0
    c = OrderedDict()
    l = []
    
    while(i<len(a)) or (j<len(b)):

        while(i<len(a)) and (a[i][0] in c):
            i+=1

        if(i<len(a)):
            c[a[i][0]] = 1
            l.append(a[i])
            i+=1

        while(j<len(b)) and (b[j][0] in c):
            j+=1

        if(j<len(b)):
            c[b[j][0]] = 1
            l.append(b[j])
            j+=1

    return l

def compute_sorted_sim(question_answerer, values, names_id):
    
    lst = [(question_answerer.get(int(elem[0])), elem[1]) for elem in values if names_id.get(question_answerer.get(int(elem[0])), None)]
    return lst

def BM25(data_name, label, n_samples):
    
    print('Starting BM25 baseline')
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    baseline_dir = f'./Dataset/{data_name}/{label}/Baselines/BM25/structures/'
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
    
    test = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                                    converters={'Tags': literal_eval, 'Topic': literal_eval})
    test = test[:n_samples]
    
    graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
    names_id = pickle.load(open(struc_dir+'names_id', 'rb'))
    question_answerer = pickle.load(open(struc_dir+'question_answerer', 'rb'))
    
    indexers_tag = {}
    for topic in graphs.keys():
        indexref = pt.IndexFactory.of(struc_dir + "./Indexes/pd_indexTag"+str(topic)+"/data.properties")
        br = pt.BatchRetrieve(indexref, wmodel='BM25')
        indexers_tag[topic] = br

    indexers_body = {}
    for topic in graphs.keys():
        indexref = pt.IndexFactory.of(struc_dir + "./Indexes/pd_indexText"+str(topic)+"/data.properties")
        br = pt.BatchRetrieve(indexref, wmodel='BM25')
        indexers_body[topic] = br

    print('Computing')
    run_dict = defaultdict(dict)
    qrels_dict = defaultdict(dict)

    for layers, tags, text, qid, acc in tqdm(list(zip(test.Topic.values, test.Tags.values, 
                                            test.CleanedContent.values, test.Id.values, 
                                            test.AcceptedAnswerer.values))):
        layers = set(layers)
        experts = []
        l_tags = []
        l_text = []
        for l in layers:
            
            translated_tags = translate_tags(tags, punc)
            tags_str = ' '.join(translated_tags)
            res_tag = indexers_tag[l].search(tags_str)
            res_body = indexers_body[l].search(terrier_query(text))
            
            values_tags = [elem for elem in list(zip(res_tag.docno.values, res_tag.score.values)) if graphs[l].vs()[names_id[l][question_answerer[l][int(elem[0])]]]['expert'] == 1]
            values_text = [elem for elem in list(zip(res_body.docno.values, res_body.score.values)) if graphs[l].vs()[names_id[l][question_answerer[l][int(elem[0])]]]['expert'] == 1]

            l_tags.append(values_tags)
            l_text.append(values_text)        
            
        l_tags = list(itertools.chain.from_iterable(l_tags))
        l_text = list(itertools.chain.from_iterable(l_text))
        
        l_tags.sort(key=lambda x: x[1], reverse=True)
        l_text.sort(key=lambda x: x[1], reverse=True)
        
        l_tags = reduce(lambda lu,i:i[0] in dict(lu).keys() and lu or lu+[i], l_tags, [])
        l_text = reduce(lambda lu,i:i[0] in dict(lu).keys() and lu or lu+[i], l_text, [])
        
        merged_list = merge_lists(l_tags, l_text)
        sorted_sim = compute_sorted_sim(question_answerer[l], merged_list, names_id[topic])
        experts = reduce(lambda lu,i:i[0] in dict(lu).keys() and lu or lu+[i], sorted_sim, [])      
        experts = [(str(int(e[0])), 100-i) for i,e in enumerate(experts[:100])]
        
        run_dict[qid] = dict(experts)
        qrels_dict[qid][str(int(acc))] = 1

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    results = evaluate(qrels, run, ["precision@1", "ndcg@3", "recall@10", "recall@100", 'mrr'])
    table = [list(results.keys()), list(results.values())]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    write_json(results, baseline_dir + 'measures')
    
    return

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--label', required=True, type=str)
    parser.add_argument('--testsize', required=False, type=int, default=50000)
    
    args = parser.parse_args()
    
    return args
        

if __name__ == "__main__":
    
    args = parse_arguments()

    data_name = args.dataset
    label = args.label
    n_samples = args.testsize
    
    BM25(data_name, label, n_samples)
