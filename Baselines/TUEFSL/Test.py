from tabulate import tabulate
import pickle
import pandas as pd
import lightgbm as lightgbm
from Utils import read_json, extract_features, join_layer_experts, evaluate_pred, parse_arguments, write_json
from ast import literal_eval
from Exploration import select_experts
import os

def evaluate(data_dir, struc_dir, n_samples, threshold, restart, n_steps):
    
    print('Testing TUEF')
    
    test_df = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                          converters={'Tags': literal_eval, 'Topic': literal_eval})
    test_df = test_df[:n_samples]
    
    ranker = lightgbm.Booster(model_file=struc_dir+'ranker.txt')
    
    if os.path.exists(struc_dir + 'test_data'): 
          
        print('Loading pre-computed data') 
        data = pickle.load(open(struc_dir + 'test_data', 'rb'))
    
    else:    
        
        filename = 'selected_experts_test_'+str(len(test_df))+'.json.gz'
        
        if os.path.exists(struc_dir + filename):
            
            print('Loading Exploration-phase results')
            data = read_json(struc_dir + filename)
            
        else:
            
            data = select_experts(struc_dir, test_df, 'test', threshold, restart, n_steps)
        
        data = [data[str(i)] for i in range(len(data))]
        
        exp_df = pd.read_csv(data_dir + 'users.csv.gz', compression='gzip')
        graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
        names_id = pickle.load(open(struc_dir+'names_id', 'rb'))
        exp_layer_ans = pickle.load(open(struc_dir+'exp_layer_ans', 'rb'))       
        
        print('Features Extraction')    
        data = extract_features(data, graphs, names_id)
        data = join_layer_experts(data, test_df, exp_df, exp_layer_ans)   
    
    pickle.dump(data, open(struc_dir + 'test_data', 'wb'))
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
    results = evaluate_pred(prediction_df, ground_truth, ["precision@1", "ndcg@3", "recall@10", "recall@100", 'mrr'], struc_dir)
    
    table = [list(results.keys()), list(results.values())]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    
    write_json(results, struc_dir + 'measures')

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    
    threshold = args.probability
    restart = args.restart
    n_steps = args.steps
    n_samples = args.testsize
    
    evaluate(data_dir, struc_dir, n_samples, threshold, restart, n_steps)