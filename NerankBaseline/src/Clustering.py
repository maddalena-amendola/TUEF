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
    
    answers_freq = dict(zip(df.Id, df.Answers))
    accepted_freq = dict(zip(df.Id, df.AcceptedAnswers))
    
    candidates = [k for k in df.Id if accepted_freq.get(k)>=p]
    accepted_ratio = {k:(accepted_freq.get(k)/answers_freq.get(k)) for k in candidates}
    
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


def compute_users_features(data_dir, users, a_train):
    
    exp_df = users[users['Expert']==1].copy()
    exp_df = exp_df[['Id', 'Reputation', 'Answers', 'AcceptedAnswers']]
    exp_df['Ratio'] = exp_df['AcceptedAnswers']/exp_df['Answers']
    exp_df['Reputation'] = exp_df['Reputation'].astype(float)
    exp_df.loc[:, 'Reputation'] = np.log(exp_df['Reputation'])
    exp_df = exp_df.rename(columns = {'Id':'Expert'})
    exp_df['Expert'] = exp_df['Expert'].astype(int)
    exp_df.index = range(len(exp_df))
    
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

def split_train_test(data_dir, questions_df, answers_df, users):
    
    uid_ask = dict(Counter(questions_df.OwnerUserId))
    uid_acc = dict(Counter(questions_df.AcceptedAnswerer))
    
    test = questions_df[[uid_ask.get(uid, 0) >=3 and uid_acc.get(acc, 0) >= 3 for uid, acc in zip(questions_df['OwnerUserId'], questions_df['AcceptedAnswerer'])]]
    train = questions_df[~questions_df['Id'].isin(test.Id)]
    
    a_train = answers_df[answers_df['ParentId'].isin(train.Id)]
    a_train.index = range(len(a_train))
    
    uids = set.union(set(train.OwnerUserId.values), set(a_train.OwnerUserId.values))
    u_train = users[users['Id'].isin(uids)]
    u_train.index = range(len(u_train))

    answers_freq = dict(Counter(a_train.OwnerUserId))
    u_train['Answers'] = u_train.Id.apply(lambda x: answers_freq.get(x, 0))

    accepted_freq = dict(Counter(train.AcceptedAnswerer))
    u_train['AcceptedAnswers'] = u_train.Id.apply(lambda x: accepted_freq.get(x, 0))
    exps = get_experts(u_train, 5)
    print('\tNumber of experts', len(exps))

    u_train['Expert'] = u_train.Id.apply(lambda x: 1 if x in exps else 0)
    u_train = compute_users_features(data_dir, u_train, a_train)
    test = test[test['AcceptedAnswerer'].isin(u_train.Expert)]
    test_size = int(len(questions_df)*0.1)
    test = test[-test_size:]
    test.index = range(len(test))
    
    train = questions_df[~questions_df['Id'].isin(test.Id)]
    train.index = range(len(train))
    
    a_train = answers_df[answers_df['ParentId'].isin(train.Id)]
    a_train.index = range(len(a_train))

    a_test = answers_df[answers_df['ParentId'].isin(test.Id)]
    a_test.index = range(len(a_test)) 
    
    print('\tTrain size:',len(train))
    print('\tTest size:',len(test))
    
    test_candedate_ids = test
    
    train.to_csv(data_dir + 'train.csv.gz', compression='gzip', index=False)
    test.to_csv(data_dir + 'test.csv.gz', compression='gzip', index=False)
    u_train.to_csv(data_dir + 'experts.csv.gz', compression='gzip', index=False)
    
    return

def update_data(data_dir, struc_dir, questions, clusters):
    
    answers = pd.read_csv(data_dir + "answers.csv.gz", compression='gzip')
    questions['Topic'] = questions.Tags.apply(lambda x: [clusters.get(elem) for elem in x if clusters.get(elem) is not None])

    questions_df = questions[questions['Topic'].str.len()>0]
    questions_df.index = range(len(questions_df))
    del questions

    answers_df = answers[answers['ParentId'].isin(questions_df.Id)]
    answers_df.index = list(range(len(answers_df)))
    del answers

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
    
    pids = [str(elem) for elem in set.union(set(questions_df.Id), set(answers_df.Id))]
    pickle.dump(pids, open(struc_dir + 'pids', 'wb'))

    return questions_df, answers_df, users_df

def extract_topics(data_dir, struc_dir, n_features, min_k, max_k):
        
    questions = pd.read_csv(data_dir + "questions.csv.gz", compression='gzip', 
                            converters={'Tags': literal_eval, 'Answerers': literal_eval})
    tags = sorted(list(set(itertools.chain.from_iterable(questions.Tags))))
    
    print('Number of train questions:', len(questions))
    print('Number of Tags:', len(tags))
    
    print('Computing clusters')
    clusters = compute_clusters(struc_dir, questions, tags, n_features, min_k, max_k)
    print('Updating train data')
    questions, answers, users = update_data(data_dir, struc_dir, questions, clusters)
    print('Splitting data into train and test sets')
    split_train_test(data_dir, questions, answers, users)
    
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