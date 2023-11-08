import json
import gzip
import string
import pickle
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from ranx import Qrels, Run, evaluate
from collections import defaultdict
import argparse

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

def read_json(filename):    
    
    with gzip.open(filename, "rt") as f:
        obj = json.load(f)
    
    return obj

def write_json(obj, filename):    
    
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)
        
    return

def terrier_query(query, lowercase=True):
    if lowercase:
        words = [word.lower() for word in query.split()]
        new_query = " ".join(words)
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    # Use the translation table to remove punctuation
    new_query = new_query.translate(translator)
    return new_query

def fill_nan(df):
    df['StepsNetwork'] = df['StepsNetwork'].fillna(np.nanmax(df.StepsNetwork))
    df['VisitCountNetwork'] = df['VisitCountNetwork'].fillna(0)
    df['StepsContent'] = df['StepsContent'].fillna(np.nanmax(df.StepsContent))
    df['VisitCountContent'] = df['VisitCountContent'].fillna(0)
    return df

def intra_layer(data, layer):
    a = data['Network'][layer]
    b = data['Content'][layer]
    df_a = pd.DataFrame(a['Weights']).T
    df_a = df_a.rename(columns={"Weight": "VisitCountNetwork", "Time": "StepsNetwork"})
    df_a['Expert'] = df_a.index
    df_a = df_a[df_a['RW']==0]
    df_a.index = range(len(df_a))

    df_b = pd.DataFrame(b['Weights']).T
    df_b = df_b.rename(columns={"Weight": "VisitCountContent", "Time": "StepsContent"})
    df_b['Expert'] = df_b.index
    df_b = df_b[df_b['RW']==0]
    df_b.index = range(len(df_b))
    
    df_conc = pd.concat([df_a, df_b])
    agg_dict = {'MethodNetwork': 'sum', 
                'MethodContent': 'sum',
                'RW': 'min'}
    for col in df_conc.columns.difference(['Expert', 'Weight', 'MethodNetwork', 'MethodContent']):
        agg_dict[col] = 'first'
    
    df_count = df_conc.groupby('Expert').size().reset_index(name='Count')
    df_grouped = df_conc.groupby('Expert').agg(agg_dict).reset_index()
    df = pd.merge(df_grouped, df_count, on='Expert')
    df['Expert'] = df['Expert'].astype(int)
    df = fill_nan(df)
    df['Weight'] = df['MethodNetwork'] + df['MethodContent']
    
    return df

def inter_layer(values):
    
    df_conc = pd.concat(values)
    agg_dict = {'StepsContent': 'min', 
            'StepsNetwork': 'min',
            'BetweennessScore': 'max',
            'BetweennessPos': 'min',
            'Closeness': 'max',
            'PageRank': 'max',
            'Eigenvector': 'max',
            'Degree': 'max',
            'AvgWeights': 'max',
            'MethodNetwork': 'max',
            'MethodContent': 'max',
            'RW': 'min'}
    for col in df_conc.columns.difference(['Expert'] + list(agg_dict.keys())):
        agg_dict[col] = 'sum'
    
    df_count = df_conc.groupby('Expert').size().reset_index(name='LayerCount')
    df_grouped = df_conc.groupby('Expert').agg(agg_dict).reset_index()
    df = pd.merge(df_grouped, df_count, on='Expert')
    df['Expert'] = df['Expert'].astype(int)
    
    return df

def extract_features(data, graphs, names_id):
    
    nodes = {int(layer):list(graphs[int(layer)].vs()) for layer in graphs.keys()}

    for elem in data:
        for method in elem.keys():
            for layer in elem.get(method).keys():
                for expert in elem.get(method).get(layer).get('Weights'):
                    node = nodes[int(layer)][names_id[int(layer)][int(expert)]]
                    elem[method][layer]['Weights'][expert]['Closeness'] = node['closeness'] #closeness[int(layer)][names_id[int(layer)][int(expert)]]
                    elem[method][layer]['Weights'][expert]['PageRank'] = node['pagerank'] #pr[int(layer)][names_id[int(layer)][int(expert)]]
                    elem[method][layer]['Weights'][expert]['Eigenvector'] = node['eigen']#eigen[int(layer)][names_id[int(layer)][int(expert)]]
                    elem[method][layer]['Weights'][expert]['Degree'] = node['degree']#degree[int(layer)][names_id[int(layer)][int(expert)]]
                    elem[method][layer]['Weights'][expert]['AvgWeights'] = node['avg_weights']#avg_weights[int(layer)][names_id[int(layer)][int(expert)]]

                    if method == 'Network':
                        elem[method][layer]['Weights'][expert]['MethodNetwork'] = 1
                        elem[method][layer]['Weights'][expert]['MethodContent'] = 0
                    else:
                        elem[method][layer]['Weights'][expert]['MethodNetwork'] = 0
                        elem[method][layer]['Weights'][expert]['MethodContent'] = 1
    
    return data

def join_layer_experts(data, l2r_df, exp_df, exp_layer_ans, hit = False):

    dfs = []
    # create new dataframe with the same index
    for i, (qid, acc, topics) in enumerate(tqdm(list(zip(l2r_df.Id, l2r_df.AcceptedAnswerer, l2r_df.Topic)))):
        
        layers = []
        topics = set(topics)
        for topic in topics:
            layers.append(intra_layer(data[i], str(topic)))
        df = inter_layer(layers)
        
        if (hit and acc in df.Expert.values) or not hit:     
            df['Expert'] = df['Expert'].astype(int)                    
            df_agg = pd.merge(df, exp_df[['Reputation', 'Expert', 'Answers', 'AcceptedAnswers', 'Ratio', 'AvgHours', 'StdHours']], on='Expert', how='left')
            df_agg['qid'] = qid
            df_agg['relevance'] = df_agg['Expert'].apply(lambda x: int(x==int(acc)))
            df_agg.index = np.arange(len(df_agg))

            #-----------------------------------------------------------------
            ratio = []
            for e in df_agg.Expert:
                answers = len(set(itertools.chain.from_iterable([exp_layer_ans['answers'][e][t] for t in topics])))
                accepted = len(set(itertools.chain.from_iterable([exp_layer_ans['accepted'][e][t] for t in topics])))
                ratio.append(accepted/answers)
            df_agg['QueryKnowledge'] = ratio

            #-----------------------------------------------------------------
            dfs.append(df_agg)
            
    return dfs

def evaluate_pred(predictions, ground_truth, metrics, baseline_dir = False):
    
    run_dict = defaultdict(dict)
    for qid, exp, s in zip(predictions.qid, predictions.Expert, predictions.score):
        run_dict[qid][str(int(exp))] = s
        
    qrels_dict = defaultdict(dict)
    for qid, exp in ground_truth:
        qrels_dict[qid][str(int(exp))] = 1
        
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    
    if baseline_dir:
        write_json(run_dict, baseline_dir + 'run_dict')
        
    return evaluate(qrels, run, metrics)

def parse_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--label', required=True, type=str)
    parser.add_argument('--ltrsize', required=False, type=int, default=50000)
    parser.add_argument('--maxevals', required=False, type=int, default=300)
    parser.add_argument('--testsize', required=False, type=int, default=50000)
    
    args = parser.parse_args()
    
    return args