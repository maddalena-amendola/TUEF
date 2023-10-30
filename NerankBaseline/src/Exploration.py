import numpy as np
import pandas as pd
import multiprocessing
from timeit import default_timer as timer
from numpy.random import choice
from collections import OrderedDict
from itertools import repeat
from collections import Counter
from collections import defaultdict
import pickle
from ast import literal_eval
from Utils import *

import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

###############################################################################################
np.random.seed(1)

def randomwalk(g, start_node, restart, node_count, time, steps):

    time+=1

    for j in range(restart):        
        node_id = start_node
        
        for i in range(steps):
            #get neighbors
            list_of_candidates = g.neighbors(node_id, mode="all")
            
            if(len(list_of_candidates)>0):
                #get edges weight
                weights = [g.es[g.get_eid(node_id, n, error=False)]['weight'] for n in list_of_candidates]
                #compute the probability distribution
                probability_distribution = weights/np.sum(weights)
                #choose one node at random from the probability distribution
                node_id = int(choice(list_of_candidates, 1, p=probability_distribution)[0])
                node = g.vs()[node_id]
                time+=1
                
                if node['expert'] == 1:
                    if node['name'] not in node_count:
                        node_count[int(node['name'])] = {'Weight': 1, 'Time': time, 'RW': 1}
                    else:
                        node_count[int(node['name'])]['Weight']+=1
            else:
                break

    return node_count, time    

def parallel_layer(g, names_id, node_count, topic, dict_, restart, n_steps, lst_pr, lst_body, lst_tags):
    
    d = dict()
    time = int(np.max([node_count.get(key).get('Time') for key in node_count.keys()]))
    experts = [elem[0] for elem in sorted([(a[0], a[1]['Time']) for a in list(node_count.items())], key=lambda x:x[1])]
    
    for elem in experts:
        node = int(elem)
        #compute the random walk 
        node_count, time = randomwalk(g, names_id[node], restart, node_count, time, n_steps)

    text_freq, text_scores = get_freq_score(lst_body)
    tag_freq, tag_scores = get_freq_score(lst_tags)
    
    for expert in node_count:    
        node_count[int(expert)]['FreqIndexText'] = int(text_freq.get(expert, 0))
        node_count[int(expert)]['ScoreIndexText'] = float(np.max(text_scores.get(expert, [0])))

        node_count[int(expert)]['FreqIndexTag'] = int(tag_freq.get(expert, 0))
        node_count[int(expert)]['ScoreIndexTag'] = float(np.max(tag_scores.get(expert, [0])))

        ind = lst_pr.index(list(filter(lambda x:expert==x[0], lst_pr))[0])
        node_count[int(expert)]['BetweennessScore'] = float(lst_pr[ind][1])
        node_count[int(expert)]['BetweennessPos'] = int(ind)
    
    d['Weights'] = node_count
    dict_[str(topic)] = d

    return

def get_freq_score(lst):

    freq = defaultdict(int) 
    scores = defaultdict(list)

    for elem in lst:
        freq[elem[0]]+=1
        scores[elem[0]].append(elem[1])

    return freq, scores

def merge_lists(a, b):

    j = 0
    i = 0
    c = OrderedDict()
    while(i<len(a)) or (j<len(b)):

        while(i<len(a)) and (a[i] in c):
            i+=1

        if(i<len(a)):
            c[a[i]] = 1
            i+=1

        while(j<len(b)) and (b[j] in c):
            j+=1

        if(j<len(b)):
            c[b[j]] = 1
            j+=1

    return list(c.keys())

def compute_sorted_sim(question_answerer, values, names_id):
    
    lst = [question_answerer.get(int(elem)) for elem in values if question_answerer.get(int(elem)) in names_id]
    return lst

def extract_experts(lst, threshold, g, names_id):

    p = 1
    i = 0
    node_count = dict()

    while(p>threshold and i<len(lst)):

        node_name = lst[i]
        if node_name in names_id:
            node = g.vs()[names_id[node_name]]

            if(node['expert'] == 1):
                if(node_name not in node_count):
                    p = p * node['probability']
                    node_count[int(node_name)] = {'Weight': 1, 'Time': i, 'RW': 0}
                else:
                    node_count[int(node_name)]['Weight'] += 1

        i+=1

    return node_count

def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def process_question(topics, retrieved_docs, question_answerer, graphs, names_id, betweenness, threshold, restart, n_steps):

    topics = set(topics)     
    #build the shared dictionary which collects, for each layer, the set of experts, the reached probability and 
    #the number of random walks computed
    manager = multiprocessing.Manager()
    
    pr_dict = manager.dict({topic: dict() for topic in topics})
    qs_dict = manager.dict({topic: dict() for topic in topics})

    processes = []
    for topic in topics:
        
        graph = graphs[topic]
        
        ######################## QUESTION SIM RW ########################
        
        sim_questions_tags = retrieved_docs[topic]['tags']
        sim_questions_body = retrieved_docs[topic]['body']

        merged_list = merge_lists(sim_questions_tags.docno.values, sim_questions_body.docno.values)
        sorted_sim = compute_sorted_sim(question_answerer[topic], merged_list, names_id[topic])
        
        node_count = extract_experts(sorted_sim, threshold, graph, names_id[topic])
        lst_body = list(zip(compute_sorted_sim(question_answerer[topic], sim_questions_body.docno.values, names_id[topic]),
                            sim_questions_body.score.values))
        lst_tag = list(zip(compute_sorted_sim(question_answerer[topic], sim_questions_tags.docno.values, names_id[topic]),
                            sim_questions_tags.score.values))
        
        process = multiprocessing.Process(target=parallel_layer, args=(graph,
                                                                       names_id[topic],
                                                                       node_count,
                                                                       topic,
                                                                       qs_dict,
                                                                       restart,
                                                                       n_steps,
                                                                       betweenness[topic],
                                                                       lst_body,
                                                                       lst_tag))
        process.start()
        processes.append(process)
        
        ######################## PAGE RANK RW ########################
        
        node_count = extract_experts([elem[0] for elem in betweenness[topic]], threshold, graph, names_id[topic])
        
        # Create and start a process that updates the shared list
        process = multiprocessing.Process(target=parallel_layer, args=(graph,
                                                                       names_id[topic],
                                                                       node_count,
                                                                       topic,
                                                                       pr_dict,
                                                                       restart,
                                                                       n_steps,
                                                                       betweenness[topic],
                                                                       lst_body,
                                                                       lst_tag))    
        
        process.start()
        processes.append(process)
        
    for p in processes:
        p.join()

    for p in processes:
        p.close()

    return {'Network': dict(pr_dict), 'Content': dict(qs_dict)}

def parallel_questions(questions, shared_dict, retrieved_docs, question_answerer, graphs, names_id, betweenness, threshold, restart, n_steps):
    
    for index, (i, body, topic, tags, id_) in enumerate(questions):
        shared_dict[str(i)] = process_question(topic, retrieved_docs[i], question_answerer, graphs, names_id, betweenness, threshold, restart, n_steps)
        if((index+1)%100 == 0):
            print('Elaborated:', index+1)
    return

def retrieve(q, indexers_tag, indexers_body, punc):

    translated_tags = translate_tags(q[3], punc)
    tags_str = ' '.join(translated_tags)
    result = defaultdict(dict)
    for topic in set(q[2]):
        
        qid = str(q[4])

        res_tag = indexers_tag[topic].search(tags_str)
        res_tag = res_tag[res_tag['docno'] != qid]
        result[topic]['tags'] = res_tag

        res_body = indexers_body[topic].search(terrier_query(q[1]))
        res_body = res_body[res_body['docno'] != qid]
        result[topic]['body'] = res_body

    return result

def select_experts(struc_dir, df, label, threshold = 0.001, restart = 5, n_steps = 10):
    
    graphs = pickle.load(open(struc_dir+'graphs', 'rb'))
    names_id = pickle.load(open(struc_dir+'names_id', 'rb'))
    question_answerer = pickle.load(open(struc_dir+'question_answerer', 'rb'))
    betweenness = pickle.load(open(struc_dir+'betweenness', 'rb'))
    
    print('\tStarting Exploration phase')
    
    n_samples = len(df)
    print(f'\tNumber of samples: {n_samples}')
    
    print('\tLoading indexes')
    indexers_tag, indexers_body = dict(), dict()
    for topic in graphs.keys():
        
        indexref = pt.IndexFactory.of(struc_dir + "./Indexes/pd_indexTag"+str(topic)+"/data.properties")
        br = pt.BatchRetrieve(indexref, wmodel='BM25')
        indexers_tag[topic] = br
        
        indexref = pt.IndexFactory.of(struc_dir + "./Indexes/pd_indexText"+str(topic)+"/data.properties")
        br = pt.BatchRetrieve(indexref, wmodel='BM25')
        indexers_body[topic] = br

    filename = f'selected_experts_{label}_{n_samples}'
    
    test = list(zip(df.index.values, df.CleanedContent.values, df.Topic.values, df.Tags.values, df.Id.values))
    chunks = split(test, 50) 
    
    manager = multiprocessing.Manager()
    results = manager.dict()

    start = timer()

    print('\tRetrieving historical documents')
    quetions_processes = []
    for elem in chunks:       

        retrieved_docs = dict()
        for q in elem:
            retrieved_docs[q[0]] = retrieve(q, indexers_tag, indexers_body, punc)

        process_q = multiprocessing.Process(target = parallel_questions, args=(elem, results, retrieved_docs, question_answerer, graphs, names_id, betweenness, threshold, restart, n_steps))
        process_q.start()
        quetions_processes.append(process_q)
    
    print('\tStarting the exploration')
    for pr in quetions_processes:
        pr.join()

    for pr in quetions_processes:
        pr.close()

    print(f'\tTotal time: {(timer()-start)/60}')

    results = dict(results)
    write_json(results, struc_dir + filename + '.json.gz')
    
    return json.loads(json.dumps(results))

if __name__ == "__main__":
        
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    struc_dir = f'./Dataset/{data_name}/{label}/structures/'
    
    threshold = args.probability
    restart = args.restart
    n_steps = args.steps
    test_samples = args.ltrsize
    
    df = pd.read_csv(data_dir + "test.csv.gz", compression='gzip', 
                                converters={'Tags': literal_eval, 'Topic': literal_eval})
    df = df[:test_samples]
    df.index = range(len(df))   
        
    select_experts(struc_dir, df, label, threshold = 0.001, restart = 5, n_steps = 10)