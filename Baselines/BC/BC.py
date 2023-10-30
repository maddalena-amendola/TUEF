import pickle
import itertools
from collections import defaultdict
import pandas as pd
from ast import literal_eval
from tabulate import tabulate
from ranx import Qrels, Run, evaluate
import os
import argparse
import json
import gzip

def write_json(obj, filename):    
    
    with gzip.open(filename, "wt") as f:
        json.dump(obj, f)
        
    return

def BC(data_name, label, n_samples):
    
    print('Starting Betweenness Centrality (BC) baseline')
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    baseline_dir = f'./Dataset/{data_name}/{label}/Baselines/BC/structures/'
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
    
    test = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                                    converters={'Tags': literal_eval, 'Topic': literal_eval})
    test = test[:n_samples]

    betweenness = pickle.load(open(struc_dir+'betweenness', 'rb'))
    graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
    names_id = pickle.load(open(struc_dir+'names_id', 'rb'))

    run_dict = defaultdict(dict)
    qrels_dict = defaultdict(dict)

    for qid, layers, acc in zip(test.Id.values, test.Topic.values, test.AcceptedAnswerer.values):
        layers = set(layers)
        experts = []
        for l in layers:
            experts.append([elem for elem in betweenness[l] if elem[0] in names_id[l]][:10])
        
        experts = list(itertools.chain.from_iterable(experts))
        experts.sort(key=lambda x: x[1], reverse=True)
        experts = [(str(int(e[0])), e[1]) for e in experts[:100]]
        
        run_dict[qid] = dict(experts)
        qrels_dict[qid][str(int(acc))] = 1

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    results = evaluate(qrels, run, ["precision@1", "ndcg@3", "recall@10", "recall@100", "mrr"])
    table = [list(results.keys()), list(results.values())]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    write_json(results, baseline_dir + 'measures')
    
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
    
    BC(data_name, label, n_samples)