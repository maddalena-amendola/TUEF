import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from collections import Counter
from sklearn.cluster import KMeans
from Utils import write_json, parse_arguments
from ast import literal_eval
import os
import pickle

def cluster_data(df, min_k, max_k):
    
    silhouette_scores = []
    labels = []

    for k in range(min_k, max_k+1):

        model = KMeans(n_clusters = k, random_state = 1, n_init = 1)
        model = model.fit(df)
        labels_k = model.labels_
        labels.append(labels_k)
        
        score = metrics.silhouette_score(df, labels_k)
        silhouette_scores.append(round(score, 4))
        
    return silhouette_scores, labels

def remove_correlated_features(df):
        
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]
    # Drop features 
    df.drop(to_drop, axis=1, inplace=True)

    return df

def get_experts(df, p):
    
    #build the dictionary where the keys are the Id of the users and the values are the frequencies of answers provided
    answers_freq = dict(zip(df.Id, df.Answers))
    accepted_freq = dict(zip(df.Id, df.AcceptedAnswers))
    candidates = [k for k,v in accepted_freq.items() if v>=p]
    
    #build the dictionary where the keys are the Id of the users and the values are the ratio between
    #the number of accepted answers and the number of answers provided
    accepted_ratio = {k:(accepted_freq.get(k)/answers_freq.get(k)) for k in candidates}
    #we compute the mean of the ratio
    mean = np.sum(list(accepted_ratio.values()))/len(accepted_ratio)
    experts = set([k for k in accepted_ratio.keys() if accepted_ratio.get(k)>mean])
    
    return experts

def compute_cooccurence(questions, tags, features):
    
    occ = np.zeros((len(tags), len(features)))
    tag_id = dict(zip(tags, range(len(tags))))
    feature_id = dict(zip(features, range(len(features))))
    set_cols = set(features)

    for elem in questions.Tags:
        l = set.intersection(set(elem), set_cols)
        for t in elem:
            for f in l:
                occ[tag_id[t], feature_id[f]]+=1            
                
    data = pd.DataFrame(columns = features, index = tags, data = occ)
    
    return data

def compute_clusters(struc_dir, questions, tags, n_features, min_k, max_k):
    
    count_elem = Counter(list(itertools.chain.from_iterable(questions.get('Tags'))))
    columns = [elem[0] for elem in count_elem.most_common(n_features)]

    data = compute_cooccurence(questions, tags, columns)

    data = remove_correlated_features(data)
    features = data.columns.values
    #remove all tags that does not co-occure with features
    data = data.loc[(data.sum(axis=1) != 0)]
    print('\tFeatures considered:', list(data.columns))
    print('\tNumber of tags considered:', len(data))

    data_norm = data.pipe(lambda x: (x*100).div(data.sum(axis = 1), axis='index'))

    s, l = cluster_data(data_norm, min_k, max_k)
    peak = s.index(np.max(s))
    labels_cl = l[peak]
    data_norm['label'] = labels_cl

    print('\tNumber of clusters:', len(set(labels_cl)))
    clusters = dict(zip(data_norm.index, data_norm.label))
    write_json(clusters, struc_dir + 'clusters.json.gz')
    
    return clusters

def update_train(data_dir, questions, clusters):
    
    answers = pd.read_csv(data_dir + "answers_train.csv.gz", compression='gzip')
    questions['Topic'] = questions.Tags.apply(lambda x: [clusters.get(elem) for elem in x if clusters.get(elem) is not None])

    questions_df = questions[questions['Topic'].str.len()>0]
    questions_df.index = range(len(questions_df))
    del questions

    answers_df = answers[answers['ParentId'].isin(questions_df.Id)]
    answers_df.index = list(range(len(answers_df)))
    del answers

    #save questions train and answers
    questions_df.to_csv(data_dir + 'train.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')
    answers_df.to_csv(data_dir + 'answers_train.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')

    print('\tTrain size:', len(questions_df))
    return questions_df, answers_df

def compute_users_features(data_dir, users):
    
    exp_df = users[users['Expert']==1].copy()
    exp_df = exp_df[['Id', 'Reputation', 'Answers', 'AcceptedAnswers']]
    exp_df['Ratio'] = exp_df['AcceptedAnswers']/exp_df['Answers']
    exp_df['Reputation'] = exp_df['Reputation'].astype(float)
    exp_df.loc[:, 'Reputation'] = np.log(exp_df['Reputation'])
    exp_df = exp_df.rename(columns = {'Id':'Expert'})
    exp_df['Expert'] = exp_df['Expert'].astype(int)
    exp_df.index = range(len(exp_df))

    a_train = pd.read_csv(data_dir + "answers_train.csv.gz", compression='gzip')
    a_train['CreationDate'] = pd.to_datetime(a_train['CreationDate'])

    print('\tActivity Level')
    mean = []
    std = []
    for e in exp_df.Expert.values:
        a = a_train[a_train['OwnerUserId'] == e]
        a['Difference'] = (a['CreationDate'] - a['CreationDate'].shift(+1))
        a['Hours'] = a['Difference'].apply(lambda x: x.total_seconds()/3600)
        mean.append(np.mean(a.Hours.values[1:]))
        std.append(np.std(a.Hours.values[1:]))
    exp_df['AvgHours'] = mean
    exp_df['StdHours'] = std
                            
    return exp_df


def update_users(data_dir, questions_df, answers_df):
    
    users = pd.read_csv(data_dir + "users.csv.gz", compression='gzip')
    #we select all users who have not asked or answered any other questions
    users_tokeep = set.union(set(questions_df.OwnerUserId), set(answers_df.OwnerUserId))
    users_df = users[users['Id'].isin(users_tokeep)]
    users_df.index = list(range(len(users_df)))
    del users

    answers_freq = dict(Counter(answers_df.OwnerUserId))
    users_df['Answers'] = users_df.Id.apply(lambda x: answers_freq.get(x, 0))

    accepted_freq = dict(Counter(questions_df.AcceptedAnswerer))
    users_df['AcceptedAnswers'] = users_df.Id.apply(lambda x: accepted_freq.get(x, 0))

    #compute the experts
    exps = get_experts(users_df, 3)
    print('\tNumber of experts', len(exps))

    users_df['Expert'] = users_df.Id.apply(lambda x: 1 if x in exps else 0)
    users_df = compute_users_features(data_dir, users_df)
    
    users_df.to_csv(data_dir + 'users.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')

    return exps

def update_test(data_dir, struc_dir, clusters, exps):
    
    test = pd.read_csv(data_dir + "test.csv.gz", compression='gzip',
                    converters={'Tags': literal_eval, 'Answerers': literal_eval})

    test = test[test['AcceptedAnswerer'].isin(exps)]
    test.index = range(len(test))

    test['Topic'] = test.Tags.apply(lambda x: [clusters.get(elem) for elem in x if clusters.get(elem) is not None])
    test = test[test['Topic'].str.len()>0]
    test.index = range(len(test))
    
    a_test = pd.read_csv(data_dir + "answers_test.csv.gz", compression='gzip')
    a_test = a_test[a_test['ParentId'].isin(test.Id)]
    a_test.index = range(len(a_test))
    
    train = pd.read_csv(data_dir + "train.csv.gz", compression='gzip',
                    converters={'Tags': literal_eval, 'Answerers': literal_eval})
    a_train = pd.read_csv(data_dir + "answers_train.csv.gz", compression='gzip')
    
    pids = [str(elem) for elem in set.union(set(train.Id), set(a_train.Id),
                                            set(test.Id), set(a_test.Id))]
    pickle.dump(pids, open(struc_dir + 'pids', 'wb'))
    
    size = len(test) + len(train)
    test_size = int(size*0.1)
    
    train = pd.concat([train, test[:-test_size]])
    train['CreationDate']= pd.to_datetime(train['CreationDate'])
    train = train.sort_values(by='CreationDate')
    train.index = range(len(train))
    
    test = test[-test_size:]
    test.index = range(len(test))       
    
    print('Train updated:', len(train))    
    print('Test updated:', len(test))         
    
    test.to_csv(data_dir + 'test.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')
    train.to_csv(data_dir + 'train.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')
    
    
    return

def extract_topics(data_dir, struc_dir, n_features, min_k, max_k):
        
    questions = pd.read_csv(data_dir + "train.csv.gz", compression='gzip', 
                            converters={'Tags': literal_eval, 'Answerers': literal_eval})
    tags = sorted(list(set(itertools.chain.from_iterable(questions.Tags))))
    
    print('Number of train questions:', len(questions))
    print('Number of Tags:', len(tags))
    
    print('Computing clusters')
    clusters = compute_clusters(struc_dir, questions, tags, n_features, min_k, max_k)
    print('Updating train data')
    questions, answers = update_train(data_dir, questions, clusters)
    print('Updating users data')
    experts = update_users(data_dir, questions, answers)
    print('Updating test data')
    update_test(data_dir, struc_dir, clusters, experts)
    
    return questions, clusters

if __name__ == "__main__":
        
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    
    min_k = args.mink
    max_k = args.maxk
    n_features = args.features
    min_accepted_answers = args.minaccans
    
    if not os.path.exists(struc_dir+'Indexes/'):
        os.mkdir(struc_dir+'Indexes/')
    
    _, _ = extract_topics(data_dir, struc_dir, n_features, min_k, max_k)