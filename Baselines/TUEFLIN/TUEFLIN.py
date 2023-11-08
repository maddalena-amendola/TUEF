import pickle
import pandas as pd
import numpy as np
import itertools
from collections import Counter, defaultdict
from ranx import Qrels, Run, evaluate
from ast import literal_eval
from tqdm import tqdm
from tabulate import tabulate
import os
import argparse
import sys
import json
import gzip

def write_json(obj, filename):    
    
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)
        
    return

def get_AACT(df, u, tags, pt_values):
    
    profile_tags = set(pt_values.get(u))
    ct = set.intersection(profile_tags, set(tags))
    
    if len(ct)==0:
        return 0
    
    aact = len(df[df['Tags'].apply(lambda x: any(item in x for item in ct))])/len(ct)
    #aact = len(df[pd.DataFrame(df.Tags.tolist()).isin(ct).any(1).values])/len(ct)
    
    return aact    

def get_segment(x):
    if 0 <= x < 3:
        return 0
    if 3 <= x < 6:
        return 1
    if 6 <= x < 12:
        return 2
    if x >= 12:
        return 3

def get_ram(u_acc, tags, q_date, w=[0.4, 0.3, 0.2, 0.1]):
    
    u_acc_t = u_acc[u_acc['Tags'].apply(lambda x: any(item in x for item in tags))].copy()
    #u_acc_t = u_acc[pd.DataFrame(u_acc.Tags.tolist()).isin(tags).any(1).values].copy()

    u_acc_t["Segment"] = u_acc_t['CreationDate'].apply(lambda x: get_segment((q_date-x).days/30))
    freq = u_acc_t.groupby('Segment', as_index = False).count()[['Segment', 'Id']]
    
    ram = 0
    for s, c in zip(freq.Segment, freq.Id):
        ram += (c*w[s])
        
    return ram

def evaluate_lin(data_dir, struc_dir, baseline_dir, n_samples, users, questions):
    
    test_df = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                          converters={'Tags': literal_eval, 'Topic': literal_eval})
    test_df = test_df[:n_samples]
    test_df['CreationDate'] = pd.to_datetime(test_df['CreationDate'])
        
    data = pickle.load(open(struc_dir + 'test_data', 'rb'))
    
    aa_values = dict(zip(users.Expert, users.AcceptedAnswers))
    pt_values = dict(zip(users.Expert, users.ProfileTags))
    npt_values = dict(zip(users.Expert, users.NPT))
    tt_values = dict(zip(users.Expert, users.TT))

    ranked = []
    for i, (tags, date) in enumerate(tqdm(list(zip(test_df.Tags, test_df.CreationDate)))):
        
        date = pd.to_datetime(date)
        
        experts = set(data[i].Expert.values)
        
        predictions = []
        for e in experts:
            u_acc = questions[questions['AcceptedAnswerer']==e]
            
            aact = get_AACT(u_acc, e, tags, pt_values)
            ram = get_ram(u_acc, tags, date)
            
            score = tt_values.get(e) * np.log10(npt_values.get(e) + 10) * np.log10(aa_values.get(e)+10) * aact * ram
            predictions.append((str(int(e)), score))
            
        predictions = sorted(predictions, key = lambda x: x[1], reverse=True)
        ranked.append(predictions)        

    run_dict = defaultdict(dict)
    for i, qid in enumerate(test_df.Id):
        run_dict[int(qid)] = dict(ranked[i])
            
    qrels_dict = defaultdict(dict)
    for qid, exp in zip(test_df.Id, test_df.AcceptedAnswerer):
        qrels_dict[int(qid)][str(int(exp))] = 1
    
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    results = evaluate(qrels, run, ["precision@1", "ndcg@3", "recall@10", "recall@100", 'mrr'])
    
    table = [list(results.keys()), list(results.values())]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    write_json(run_dict, baseline_dir + 'run_dict')


def TUEFLIN(data_name, label, n_samples):
    
    print('Starting TUEFLIN baseline')
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    baseline_dir = f'./Dataset/{data_name}/{label}/Baselines/TUEFLIN/structures/'
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
        
    questions = pd.read_csv(data_dir + "train.csv.gz", compression='gzip', 
                                    converters={'Tags': literal_eval, 'Answerers': literal_eval, 'Topic': literal_eval})
    questions['CreationDate'] = pd.to_datetime(questions['CreationDate'])
    
    if os.path.exists(baseline_dir + 'users.csv.gz'):
        users = pd.read_csv(baseline_dir + 'users.csv.gz', compression='gzip', converters={'ProfileTags': literal_eval})
    else:    
        users = pd.read_csv(data_dir + 'users.csv.gz', compression='gzip')

        profile_tags = []
        tt_values = []    
        for u in tqdm(users.Expert.values): 
                
            #exp_ans = questions[pd.DataFrame(questions.Answerers.tolist()).isin([u]).any(1).values]
            exp_ans = questions[questions['Answerers'].apply(lambda x: u in x)]
            exp_acc = questions[questions['AcceptedAnswerer'] == u]
            
            count_elem = Counter(itertools.chain.from_iterable(exp_ans.Tags))
            values = sorted(list(count_elem.values()))
            u_tag = [x for x in count_elem if count_elem[x]>=np.percentile(values,75)]
            profile_tags.append(u_tag)
            
            #tt = len(exp_acc[pd.DataFrame(exp_acc.Tags.tolist()).isin(u_tag).any(1).values])/len(exp_acc)
            tt = len(exp_acc[exp_acc['Tags'].apply(lambda x: any(item in x for item in u_tag))])/len(exp_acc)
            tt_values.append(tt)
        
        users['ProfileTags'] = profile_tags
        users['TT'] = tt_values
        users['NPT'] = users['ProfileTags'].apply(lambda x: len(x))
        
        users.to_csv(baseline_dir + 'users.csv.gz', compression='gzip', index=False)
    
    evaluate_lin(data_dir, struc_dir, baseline_dir, n_samples, users, questions)
    
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

    TUEFLIN(data_name, label, n_samples)