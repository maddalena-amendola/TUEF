from Utils import parse_arguments
from Preprocessing import process_data
from MultiLayerGraph import build_MLG
from LearningToRank import learning_to_rank
import os

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    
    raw_dir = f'../Dataset/{data_name}/{label}/data/'
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    
    if not os.path.exists(f'./Dataset/{data_name}/{label}/'):
        os.makedirs(f'./Dataset/{data_name}/{label}/')
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    if not os.path.exists(struc_dir):
        os.mkdir(struc_dir)
    
    start_date = args.startdate
    end_date = args.enddate  
    
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
    
    if os.path.exists(struc_dir + 'graphs'):
        print('MLG already computed!')
    else:        
        process_data(raw_dir, data_dir)
        build_MLG(data_dir, struc_dir, min_k, max_k, n_features, min_accepted_answers)
    
    learning_to_rank(data_dir, struc_dir, ltr_samples, test_samples, threshold, restart, n_steps, max_evals)