import numpy as np
import json
import itertools
import pickle
import pandas as pd
from tqdm import tqdm
import lightgbm as lightgbm
from collections import defaultdict
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
from ast import literal_eval
from Utils import *
import os
from Exploration import select_experts
from tabulate import tabulate

def train_ranker(strc_dir, df, best_config):

    df = pd.concat(df, ignore_index=True)
    # Creating a numpy array which contains group
    qids_train = df.groupby("qid")["qid"].count().to_numpy()
    # Keeping only the features on which we would train our model 
    X_train = df.drop(["qid", "relevance", "Expert", "RW"], axis = 1)
    # Relevance label for train
    y_train = df['relevance'].astype(int)

    ranker = lightgbm.LGBMRanker(
        n_jobs = 50,
        objective="lambdarank",
        boosting_type = "gbdt",
        importance_type = "gain",
        metric= "ndcg",
        #label_gain = [0,1],
        label_gain =[i for i in range(max(y_train.max(), y_train.max()) + 1)],
        learning_rate = best_config.get('learning_rate'), 
        max_depth = int(best_config.get('max_depth')), 
        min_data_in_leaf = int(best_config.get('min_data_in_leaf')), 
        n_estimators = int(best_config.get('n_estimators')), 
        num_leaves = int(best_config.get('num_leaves')))

    # Training the model
    ranker.fit(
        X=X_train,
        y=y_train,
        group=qids_train)
    
    ranker.booster_.save_model(strc_dir + 'ranker.txt')    
    
    return ranker

def grid_search(train, test, max_evals):
    
    train = pd.concat(train, ignore_index=True)
    test = pd.concat(test, ignore_index=True)

    # Creating a numpy array which contains group
    qids_train = train.groupby("qid")["qid"].count().to_numpy()
    # Keeping only the features on which we would train our model 
    X_train = train.drop(["qid", "relevance", "Expert", "RW"], axis = 1)
    # Relevance label for train
    y_train = train['relevance'].astype(int)

    # Creating a numpy array which contains eval_group
    qids_test = test.groupby("qid")["qid"].count().to_numpy()
    # Keeping only the features on which we would validate our model
    X_test = test.drop(["qid", "relevance", "Expert", "RW"], axis = 1)
    # Relevance label for test
    y_test = test['relevance'].astype(int)
    #y_expert = test[test['relevance'] == 1].Expert.values
    ground_truth = list(zip(test[test['relevance'] == 1].qid.values, test[test['relevance'] == 1].Expert.values))

    algorithm=tpe.suggest    

    param_grid = {
        
        "n_estimators": hp.quniform('n_estimators',50,150, 5),
        "learning_rate": hp.uniform('learning_rate', 0.0001, 0.15),
        "num_leaves": hp.quniform("num_leaves", 50, 200, 10),
        "max_depth": hp.randint('max_depth', 8, 15),
        "min_data_in_leaf": hp.quniform("min_data_in_leaf", 150, 500, 25),
        }

    def objective(param_grid):
        ranker = lightgbm.LGBMRanker(
            n_jobs = 50,
            boosting_type = 'gbdt',
            objective="lambdarank",
            importance_type = "gain",
            metric= "ndcg",
            label_gain =[i for i in range(max(y_train.max(), y_test.max()) + 1)],
            n_estimators = int(param_grid['n_estimators']),
            min_data_in_leaf = int(param_grid['min_data_in_leaf']),
            max_depth = int(param_grid['max_depth']),
            num_leaves = int(param_grid['num_leaves']))
        
        ranker.fit(
        X=X_train,
        y=y_train,
        group=qids_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        eval_group=[qids_train, qids_test],
        eval_at=[1, 4])

        y_pred = ranker.predict(X_test)
        prediction_df = {
            "qid": test["qid"],
            "Expert": test["Expert"],
            "score": y_pred
        }
        
        prediction_df = pd.DataFrame(prediction_df)
        results = evaluate_pred(prediction_df, ground_truth, ["precision@1", "mrr"])

        return {'loss': 1/results['mrr'], 'status': STATUS_OK}

    best_params = fmin(
    fn=objective,
    space=param_grid,
    algo=algorithm,
    max_evals=max_evals,
    rstate=np.random.default_rng(1))

    best_config = space_eval(param_grid, best_params)
    
    return best_config

def print_result(data_test, ranker):
    
    test = pd.concat(data_test, ignore_index=True)
    
    # Creating a numpy array which contains eval_group
    qids_test = test.groupby("qid")["qid"].count().to_numpy()
    # Keeping only the features on which we would validate our model
    X_test = test.drop(["qid", "relevance", "Expert", "RW"], axis = 1)
    # Relevance label for test
    y_test = test['relevance'].astype(int)
    #y_expert = test[test['relevance'] == 1].Expert.values
    ground_truth = list(zip(test[test['relevance'] == 1].qid.values, test[test['relevance'] == 1].Expert.values))

    predictions = ranker.predict(X_test)
    prediction_df = {
        "qid": test["qid"],
        "Expert": test["Expert"],
        "score": predictions
    }
    prediction_df = pd.DataFrame(prediction_df)
            
    results = evaluate_pred(prediction_df, ground_truth, ["precision@1", "ndcg@3", "recall@10", "recall@100", 'mrr'])
    
    table = [list(results.keys()), list(results.values())]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

def learning_to_rank(data_dir, struc_dir, ltr_samples, test_samples, threshold, restart, n_steps, max_evals):
    
    print('Starting Grid Search')
    
    train = pd.read_csv(data_dir + "train.csv.gz", compression='gzip', 
                                converters={'Tags': literal_eval, 'Answerers': literal_eval, 'Topic': literal_eval})
    test = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                                converters={'Tags': literal_eval, 'Answerers': literal_eval, 'Topic': literal_eval})
    exp_df = pd.read_csv(data_dir + 'experts.csv.gz', compression='gzip')    
    
    print('\tPreparing data')
    ltr_df = train[train['AcceptedAnswerer'].isin(exp_df.Expert.values)]
    ltr_df = ltr_df[-ltr_samples:]
    ltr_df.index = range(len(ltr_df))
    
    test = test[:test_samples]
    test.index = range(len(test))
    
    print('Learning to Rank data:', len(ltr_df))
    print('Test data:', len(test))
    
    graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
    names_id = pickle.load(open(struc_dir+'names_id', 'rb'))
    exp_layer_ans = pickle.load(open(struc_dir+'exp_layer_ans', 'rb'))
    
    filename_train = 'selected_experts_ltr_'+str(len(ltr_df))+'.json.gz' 
    if os.path.exists(struc_dir + filename_train):
        print('Loading LtR collected experts')
        data_train = read_json(struc_dir + filename_train)
    else:
        data_train = select_experts(struc_dir, ltr_df, 'ltr', threshold, restart, n_steps)
        
    filename_test = 'selected_experts_test_'+str(len(test))+'.json.gz' 
    if os.path.exists(struc_dir + filename_test):
        print('Loading test collected experts')
        data_test = read_json(struc_dir + filename_test)
        print(len(data_test))
    else:
        data_test = select_experts(struc_dir, test, 'test', threshold, restart, n_steps)
        
    filename_train = 'ltr_data' 
    if os.path.exists(struc_dir + filename_train):
        print('Loading ltr extracted data')
        data_train = pickle.load(open(struc_dir + 'ltr_data', 'rb'))
    else:
        data_train = [data_train[str(i)] for i in range(len(data_train))]
        print('\tFeatures Extraction')
        data_train = extract_features(data_train, graphs, names_id)
        data_train = join_layer_experts(data_train, ltr_df, exp_df, exp_layer_ans, hit=True)
        pickle.dump(data_train, open(struc_dir + 'ltr_data', 'wb'))
        
    filename_test = 'test_data' 
    if os.path.exists(struc_dir + filename_test):        
        print('Loading test extracted data')
        data_test = pickle.load(open(struc_dir + 'test_data', 'rb'))        
    else:        
        data_test = [data_test[str(i)] for i in range(len(data_test))]
        data_test = extract_features(data_test, graphs, names_id)
        data_test = join_layer_experts(data_test, test, exp_df, exp_layer_ans, hit=True)
        
        tids = set([str(int(list(set(data_test[i].qid))[0])) for i in range(len(data_test))])
        pickle.dump(tids, open(struc_dir + 'tids', 'wb'))
        pickle.dump(data_test, open(struc_dir + 'test_data', 'wb'))

    print('\tAccuracy Train:', len(data_train)/len(ltr_df))
    print('\tAccuracy Test:', len(data_test)/len(test))

    if os.path.exists(struc_dir + 'ranker.txt'):
        ranker = lightgbm.Booster(model_file=struc_dir+'ranker.txt')
    else:
        best_config = grid_search(data_train, data_test, max_evals)
        print('\tParameters configuration:', best_config)
        
        print('\tTraining ranker')
        ranker = train_ranker(struc_dir, data_train, best_config)
        
    print_result(data_test, ranker)

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    
    threshold = args.probability
    restart = args.restart
    n_steps = args.steps
    ltr_samples = args.ltrsize
    test_samples = args.testsize
    max_evals = args.maxevals
    
    learning_to_rank(data_dir, struc_dir, ltr_samples, test_samples, threshold, restart, n_steps, max_evals)