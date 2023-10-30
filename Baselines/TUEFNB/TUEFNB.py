import pickle
import pandas as pd
import numpy as np
from tabulate import tabulate
import lightgbm as lightgbm
from ast import literal_eval
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK
import os
from glob import glob
from Utils import parse_arguments, evaluate_pred, read_json, extract_features, join_layer_experts, write_json

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
    
    return

def grid_search(data, max_evals):
    
    train_size = 0.8
    test_size = 0.2
    train = pd.concat(data[:int(len(data)*train_size)], ignore_index=True)
    test = pd.concat(data[-int(len(data)*test_size):], ignore_index=True)

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

def learning_to_rank(data_dir, struc_dir, baseline_dir, n_samples, max_evals):
    
    print('Starting Grid Search')
    
    questions = pd.read_csv(data_dir + "train.csv.gz", compression='gzip', 
                                converters={'Tags': literal_eval, 'Answerers': literal_eval, 'Topic': literal_eval})
    exp_df = pd.read_csv(data_dir + 'users.csv.gz', compression='gzip')    
    
    print('\tPreparing data')
    ltr_df = questions[questions['AcceptedAnswerer'].isin(exp_df.Expert.values)]
    ltr_df = ltr_df[-n_samples:]
    ltr_df.index = range(len(ltr_df))
    
    graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
    names_id = pickle.load(open(struc_dir+'names_id', 'rb'))
    exp_layer_ans = pickle.load(open(struc_dir+'exp_layer_ans', 'rb'))    
    
    data = read_json(glob(struc_dir + 'selected_experts_ltr*')[0])
    data = [data[str(i)] for i in range(len(data))]
    print('\tFeatures Extraction')
    data = extract_features(data, graphs, names_id)
    data = join_layer_experts(data, ltr_df, exp_df, exp_layer_ans, hit=True)
    pickle.dump(data, open(baseline_dir + 'ltr_data', 'wb'))
            
    best_config = grid_search(data, max_evals)    
    print('\tParameters configuration:', best_config)
    
    print('\tTraining ranker')
    train_ranker(baseline_dir, data, best_config)
    
def evaluate(data_dir, struc_dir, baseline_dir, n_samples):
    
    test_df = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                          converters={'Tags': literal_eval, 'Topic': literal_eval})
    test_df = test_df[:n_samples]
    
    ranker = lightgbm.Booster(model_file=baseline_dir+'ranker.txt')
        
    exp_df = pd.read_csv(data_dir + 'users.csv.gz', compression='gzip')
    graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
    names_id = pickle.load(open(struc_dir+'names_id', 'rb'))
    exp_layer_ans = pickle.load(open(struc_dir+'exp_layer_ans', 'rb'))
    
    data = read_json(glob(struc_dir + 'selected_experts_test*')[0])
    data = [data[str(i)] for i in range(len(data))]    
    
    print('Features Extraction')    
    data = extract_features(data, graphs, names_id)
    data = join_layer_experts(data, test_df, exp_df, exp_layer_ans)    
    pickle.dump(data, open(baseline_dir + 'test_data', 'wb'))
    dfs = pd.concat(data, ignore_index=True)

    # Keeping only the features on which we would validate our model
    X_test = dfs.drop(["qid", "relevance", "Expert", "RW"], axis = 1)

    predictions = ranker.predict(X_test)
    prediction_df = {
        "qid": dfs["qid"],
        "Expert": dfs["Expert"],
        "score": predictions
    }
    prediction_df = pd.DataFrame(prediction_df)
    ground_truth = [str(int(exp)) for exp in test_df.AcceptedAnswerer]
    
    ground_truth = list(zip(test_df.Id, test_df.AcceptedAnswerer))
    results = evaluate_pred(prediction_df, ground_truth, ["precision@1", "ndcg@3", "recall@10", "recall@100", 'mrr'])
    
    table = [list(results.keys()), list(results.values())]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    
    write_json(results, baseline_dir + 'measures')
    
def TUEFNB(data_name, label, ltr_size, max_evals, test_size):
    
    print('Starting TUEF Network-Based (TUEFNB) baseline')
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    baseline_dir = f'./Dataset/{data_name}/{label}/Baselines/TUEFNB/structures/'
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
        
    learning_to_rank(data_dir, struc_dir, baseline_dir, ltr_size, max_evals)
    evaluate(data_dir, struc_dir, baseline_dir, test_size)

if __name__ == "__main__":
    
    args = parse_arguments()

    data_name = args.dataset
    label = args.label  
    ltr_samples = args.ltrsize 
    test_samples = args.testsize
    max_evals = args.maxevals
    
    TUEFNB(data_name, label, ltr_samples, max_evals, test_samples)
    