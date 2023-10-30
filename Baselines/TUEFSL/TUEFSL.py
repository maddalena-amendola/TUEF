from Utils import parse_arguments
from MultiLayerGraph import build_MLG
from LearningToRank import learning_to_rank
from Test import evaluate
import pandas as pd
import os
from ast import literal_eval
from Utils import read_json, write_json

def process_data(raw_dir, a, data_dir, struc_dir):
    
    train = pd.read_csv(raw_dir + "train.csv.gz", compression='gzip', 
                            converters={'Tags': literal_eval, 'Answerers': literal_eval, 'Topic': literal_eval})
    train['Topic'] = [[0] for _ in range(len(train))]
    
    test = pd.read_csv(raw_dir + "test.csv.gz", compression='gzip', 
                            converters={'Tags': literal_eval, 'Topic': literal_eval})
    test['Topic'] = [[0] for _ in range(len(test))]
    
    a_train = pd.read_csv(raw_dir + "answers_train.csv.gz", compression='gzip')
    users = pd.read_csv(raw_dir + "users.csv.gz", compression='gzip')
    
    clusters = read_json(a+'clusters.json.gz')
    c = {k:0 for k in clusters}
    write_json(c, struc_dir + 'clusters.json.gz')
    
    train.to_csv(data_dir + 'train.csv.gz', compression='gzip', index=False)
    test.to_csv(data_dir + 'test.csv.gz', compression='gzip', index=False)
    a_train.to_csv(data_dir + 'answers_train.csv.gz', compression='gzip', index=False)
    users.to_csv(data_dir + 'users.csv.gz', compression='gzip', index=False)

    return

def TUEFSL(data_name, label, min_k, max_k, n_features, min_accepted_answers, ltr_samples, threshold, restart, n_steps, max_evals, test_samples):
    
    print('Starting TUEF Single-Layer baseline')
    
    raw_dir = f'./Dataset/{data_name}/{label}/data/'
    a = f'./Dataset/{data_name}/{label}/structures/'
    data_dir = f'./Dataset/{data_name}/{label}/Baselines/TUEFSL/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/Baselines/TUEFSL/structures/'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(struc_dir):
        os.makedirs(struc_dir)
        
    process_data(raw_dir, a, data_dir, struc_dir)        
    build_MLG(data_dir, struc_dir, min_k, max_k, n_features, min_accepted_answers)
    learning_to_rank(data_dir, struc_dir, ltr_samples, threshold, restart, n_steps, max_evals)
    evaluate(data_dir, struc_dir, test_samples, threshold, restart, n_steps)

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
        
    min_k = args.mink
    max_k = args.maxk
    n_features = args.features
    min_accepted_answers = args.minaccans
    
    ltr_samples = args.ltrsize
    threshold = args.probability
    restart = args.restart
    n_steps = args.steps   
    max_evals = args.maxevals
    test_samples = args.testsize
    
    TUEFSL(data_name, label, min_k, max_k, n_features, min_accepted_answers, ltr_samples, threshold, restart, n_steps, max_evals, test_samples)