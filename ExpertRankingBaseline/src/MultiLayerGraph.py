import pandas as pd
import numpy as np
import os
import itertools
from itertools import combinations
from collections import defaultdict
from collections import Counter
from numpy import dot
from numpy.linalg import norm
import multiprocessing
from ast import literal_eval
from tqdm import tqdm
import igraph as ig
from Utils import punc, read_json, translate_tags, parse_arguments
import pickle
from Clustering import extract_topics

import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def compute_layers(layers, user_vector, data_path):

    manager = multiprocessing.Manager()
    d = manager.dict({key : defaultdict(list) for key in layers})

    processes = []
    for layer in layers:
        print('\t\tLayer:', layer)
        process = multiprocessing.Process(target=get_edges, 
                                          args=(layer, user_vector[layer],d))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()

    for p in processes:
        p.close()

    edges = dict(d)  
    
    return edges

def get_edges(layer, nodes, d):

    edges = []
    combs = list(combinations(nodes.keys(), 2))
    print('\t\t\tNumber of pairs: ', layer, len(combs))

    for pair in combs:
        sim = cosine_similarity(nodes.get(pair[0]), nodes.get(pair[1]))
        if sim>0:
            if sim>1:
                sim = 1
            dissim = 1-sim+1e-5
            edges.append((pair[0], pair[1], sim, dissim))

    d[layer] = edges


def compute_user_vector(questions, n_features, clusters, min_accepted_answers, layers):
    
    tag_freq = Counter(list(itertools.chain.from_iterable(questions.Tags.values)))
    to_remove = set([elem[0] for elem in tag_freq.most_common(n_features)])
    
    #associate each layer with the list of its tags
    layer_tags = {elem: [] for elem in layers}
    for elem in clusters.keys():
        if elem not in to_remove:
            layer_tags[clusters.get(elem)].append(elem)
    
    # To ecah tag in each layer, associate an ID that will correspond to its position in the tag vector
    tag_id = {layer: dict(list(zip(layer_tags.get(layer), np.arange(len(layer_tags.get(layer)))))) for layer in layers}
    
    tags_to_consider = set(clusters.keys()) - to_remove
    df = questions[questions['Tags'].apply(lambda x: any(item in x for item in tags_to_consider))]
    #df = questions[pd.DataFrame(questions.Tags.tolist()).isin(tags_to_consider).any(1).values]
    df = df[~df['AcceptedAnswerer'].isna()]
    accepted_freq = dict(Counter(df.AcceptedAnswerer.values))

    user_vector = {layer: dict() for layer in layers}
    for answerer, tags in tqdm(list(zip(df.AcceptedAnswerer.values, df.Tags.values))):
        if(accepted_freq.get(answerer)>=min_accepted_answers):
            tags = set(tags) - to_remove
            for t in tags:
                if clusters.get(t) is not None:
                    layer = clusters.get(t)
                    if user_vector.get(layer).get(answerer) is None:
                        user_vector[layer][answerer] = np.array([0 for i in range(len(tag_id[layer].values()))])
                    user_vector[layer][answerer][tag_id[layer][t]] +=1
                    
    for layer in layers:
        for user in user_vector.get(layer).keys():
            user_vector[layer][user] = user_vector.get(layer).get(user)/accepted_freq.get(user)

    for key in user_vector.keys():
        print('Layer:', key, 'Users:', len(user_vector.get(key)))
        
    return user_vector

def cut_edges(edges, threshold = 0.5):
    
    cutted_edges_tag = {}
    for layer in edges.keys():
        l = [edg for edg in edges.get(layer) if edg[2]>=threshold]
        cutted_edges_tag[layer] = l
        
    return cutted_edges_tag

def compute_indexes(df, topic, struc_dir):
    
    print('\t\tBuilding Indexes')
    #Indexing the Body of the questions
    dir_path = struc_dir + "./Indexes/pd_indexText"+str(topic)
    docno = [str(elem) for elem in df.Id.values]
    text = [str(elem) for elem in df.CleanedContent.values]
    df_index = pd.DataFrame(data = {'docno': docno, 'text': text})
    pd_indexer = pt.DFIndexer(dir_path)
    _ = pd_indexer.index(df_index["text"], df_index["docno"])

    #Indexing the Tags of the questions
    dir_path = struc_dir + "./Indexes/pd_indexTag"+str(topic) 
    translated_tag = [translate_tags(elem, punc) for elem in df.Tags.values]
    questions_tags = [' '.join(elem) for elem in translated_tag]
    docno = [str(elem) for elem in df.Id.values]
    df_index = pd.DataFrame(data = {'docno': docno, 'text': questions_tags})
    pd_indexer = pt.DFIndexer(dir_path, stemmer=None, stopwords=None, tokeniser=pt.TerrierTokeniser.whitespace)
    _ = pd_indexer.index(df_index["text"], df_index["docno"])
    
    return

def compute_nodes_attributes(struc_dir, g, questions, topic, experts, names_id):
    
    print('\t\tComputing nodes attributes')
    
    df = questions[questions['Topic'].apply(lambda x: any(item in x for item in [topic]))]
    #df = questions[pd.DataFrame(questions.Topic.tolist()).isin([topic]).any(1).values]

    freq_accepted = dict(Counter(df.AcceptedAnswerer.values))
    answers_counter = dict(Counter(itertools.chain.from_iterable(df.Answerers.values)))
    max_answers = np.max(list(answers_counter.values()))

    layer_experts = list(experts - (experts - set(df.AcceptedAnswerer.values)))
    layer_experts = [elem for elem in layer_experts if elem in names_id[topic]]
    exps = [names_id[topic][e] for e in layer_experts]
    
    tot_edges = g.ecount()   

    btw = g.betweenness(vertices=exps, directed=False, weights='dissimilarity')
    n = len(g.vs())
    factor = 2/((n-1)*(n-2))
    norm_values = [elem*factor for elem in btw]    
    betweenness = dict(list(zip(exps, norm_values)))
    sorted_betweenness = sorted(list(zip(layer_experts, norm_values)), key = lambda x: x[1], reverse=True)
    
    cls = g.closeness(vertices=exps, mode='all', weights='dissimilarity', normalized=True)
    closeness = dict(list(zip(exps, cls)))
    
    eig = g.eigenvector_centrality(directed=False, scale=True, weights='weight')
    eigen = dict(list(zip(exps, [eig[i] for i in exps])))
    
    pr = g.personalized_pagerank(vertices=exps, directed=False, damping=0.85, weights='weight')
    pagerank = dict(list(zip(exps, pr)))  
    
    for i, node in enumerate(g.vs):
        
        if i in exps:
            
            name=node['name']
            g.vs()[i]['accepted_answers'] = freq_accepted.get(name)
            g.vs()[i]['answers'] = answers_counter.get(name)
            g.vs()[i]['expert'] = 1
            
            list_of_candidates = g.neighbors(node, mode="all")
            g.vs()[i]['degree'] = len(list_of_candidates)/tot_edges
            g.vs()[i]['avg_weights'] = np.mean([g.es[g.get_eid(node, n, error=False)]['weight'] for n in list_of_candidates])
            
            g.vs()[i]['betweenness'] = betweenness[i]
            g.vs()[i]['closeness'] = closeness[i]
            g.vs()[i]['eigen'] = eigen[i]
            g.vs()[i]['pagerank'] = pagerank[i]
            
            #compute the probability
            accepted_perc = freq_accepted[name]/answers_counter[name]
            g.vs()[i]['probability'] = 1 - (accepted_perc * (np.log(answers_counter[name])/np.log(max_answers)))
    
    question_answerer = dict(zip(df.Id, df.AcceptedAnswerer))
    df = df[df['AcceptedAnswerer'].isin(layer_experts)]
            
    compute_indexes(df, topic, struc_dir)
    
    return g, question_answerer, sorted_betweenness

def compute_users_structures(struc_dir, questions, experts):
    
    print('\tTag knowledge')
    exp_layer_ans = {'accepted':{e:defaultdict(list) for e in experts},
                     'answers':{e:defaultdict(list) for e in experts}}
    for (qid, ans, topics, acc) in zip(questions.Id, questions.Answerers, questions.Topic, questions.AcceptedAnswerer):
        for a in ans:
            if int(a) in experts:
                set_t = set(topics)
                if acc == a:
                    for t in set_t:
                        exp_layer_ans['answers'][a][t].append(qid)
                        exp_layer_ans['accepted'][a][t].append(qid)
                else:
                    for t in set_t:
                        exp_layer_ans['answers'][a][t].append(qid)
                        
    pickle.dump(exp_layer_ans, open(struc_dir+'exp_layer_ans', 'wb'))
    
    return

def build_MLG(data_dir, struc_dir, min_k, max_k, n_features, min_accepted_answers):
    
    if not os.path.exists(struc_dir+'Indexes/'):
        os.mkdir(struc_dir+'Indexes/')
    
    if os.path.exists(struc_dir + 'clusters.json.gz'):    
        
        print('Topic identification already done')    
        clusters = read_json(struc_dir+'clusters.json.gz')
        questions = pd.read_csv(data_dir + "train.csv.gz", compression='gzip', 
                                    converters={'Tags': literal_eval, 'Answerers': literal_eval, 'Topic': literal_eval})
    else:
        questions, clusters = extract_topics(data_dir, struc_dir, n_features, min_k, max_k)
    
    #questions, clusters = extract_topics(data_dir, struc_dir, n_features, min_k, max_k)
    users = pd.read_csv(data_dir + "users.csv.gz", compression='gzip')
   
    layers = set(clusters.values())
    
    if os.path.exists(struc_dir + 'edges'):  
        print("Loading users' relationships")
        edges = pickle.load(open(struc_dir + 'edges', 'rb'))
    else:    
        print("Computing users' knowledge vectors")
        user_vector = compute_user_vector(questions, n_features, clusters, min_accepted_answers, layers)
        print("Computing users' relationships")
        edges = compute_layers(layers, user_vector, struc_dir)
        edges = cut_edges(edges)
        pickle.dump(edges, open(struc_dir + 'edges', 'wb'))

    print('Building layers')
    graphs = {int(key): ig.Graph.TupleList(edges.get(key), directed=False, edge_attrs=["weight", "dissimilarity"]) for key in edges.keys()}
    
    experts = set([int(e) for e in users.Expert.values])
    names_id = {key: defaultdict(int) for key in graphs.keys()}
    for layer in graphs.keys():
        g = graphs.get(layer)
        for i, node in enumerate(g.vs):
            names_id[layer][node["name"]] = i
            
    question_answerer, soretd_betweenness = dict(), dict()
    
    for layer in graphs.keys():
        print('\tLayer:', layer)
        g, qa, sb = compute_nodes_attributes(struc_dir, graphs.get(layer), questions, layer, experts, names_id)
        graphs[layer] = g
        question_answerer[layer] = qa
        soretd_betweenness[layer] = sb
    
    print('Computing experts attributes')  
    compute_users_structures(struc_dir, questions, experts)
    
    pickle.dump(graphs, open(struc_dir + 'graphs', 'wb'))
    pickle.dump(soretd_betweenness, open(struc_dir + 'betweenness', 'wb'))
    pickle.dump(question_answerer, open(struc_dir + 'question_answerer', 'wb'))
    pickle.dump(names_id, open(struc_dir + 'names_id', 'wb'))
    
    return

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
    
    build_MLG(data_dir, struc_dir, min_k, max_k, n_features, min_accepted_answers)