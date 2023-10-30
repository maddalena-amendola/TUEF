import re
from bs4 import BeautifulSoup
import pandas as pd
import os
from lxml import etree
from collections import OrderedDict
from Utils import parse_arguments
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def clean_text(text):
    return BeautifulSoup(text, features = "html.parser").get_text()

def select_data(raw_dir, data_dir):
    
    questions = pd.read_csv(raw_dir + 'extracted_questions.csv.gz', compression='gzip')
    answers = pd.read_csv(raw_dir + 'extracted_answers.csv.gz', compression='gzip')
    users = pd.read_csv(raw_dir + 'extracted_users.csv.gz', compression='gzip')
    
    print('\tQuestions:', len(questions))
    print('\tAnswers:', len(answers))
    print('\tUsers:', len(users))
    print('\tSelecting questions and answers')
    #select the questions with specified owner and accepted answerer
    questions_df = questions[~questions['OwnerUserId'].isna() 
                            & ~questions['AcceptedAnswerId'].isna() 
                            & ~questions['Id'].isna()
                            & questions['OwnerUserId'].isin(users.Id)].copy()
    questions_df.index = range(len(questions_df))
    questions_df.shape

    #we select only the answers with a specified OwnerUserId (the Id of the answerer) and ParentId (the Id of the question)
    answers_df = answers[~answers['OwnerUserId'].isna() 
                        & ~answers['ParentId'].isna() 
                        & ~answers['Id'].isna()
                        & answers['OwnerUserId'].isin(users.Id)].copy()
    answers_df.index = range(len(answers_df))

    print('\tAdding the information about the answerers')
    #adding the information about the answerers
    question_answer = {qid:[] for qid in questions_df.Id.values}
    for qid, aid in zip(answers_df.ParentId.values, answers_df.OwnerUserId.values):
        if qid in question_answer.keys():
            question_answer[qid].append(aid)
            
    #we are considering all answerers: whoever answered twice will be considered once
    answerers = []
    for qid in questions_df.Id.values:
        answerers.append(list(set(question_answer.get(qid))))
    questions_df['Answerers'] = answerers
    questions_df.shape

    print('\tAdding the information about the best answerer')
    answers_dict = dict(zip(answers_df.Id.values, answers_df.OwnerUserId.values))
    accepted_answerers = []
    for aid in questions_df.AcceptedAnswerId.values:
        if aid in answers_dict.keys():
            accepted_answerers.append(answers_dict.get(aid))
        else:
            accepted_answerers.append(None)
    questions_df['AcceptedAnswerer'] = accepted_answerers

    print('\tRemove questions where the asker is the best answerer')
    questions_df = questions_df[questions_df['AcceptedAnswerer']!=questions_df['OwnerUserId']]
    questions_df.index = range(len(questions_df))
    questions_df.shape

    print('\tRemoving all questions with mismatch on the AcceptedAnswerId column')
    aid_qid =  dict(zip(answers_df.Id.values, answers_df.ParentId.values))
    indexes, null_aids = [], []
    for index, (qid, aid) in enumerate(zip(questions_df.Id.values, 
                                            questions_df.AcceptedAnswerId.values)):
        if(aid_qid.get(aid, None)):
            if(aid_qid.get(aid) == qid):
                indexes.append(index)
        else:
            indexes.append(index)
            null_aids.append(index)
            
    questions_df.loc[null_aids, "AcceptedAnswerId"] = None
    questions_df = questions_df.loc[indexes]
    questions_df.shape    
    
    questions_df = questions_df[-20000:]
    questions_df.index = range(len(questions_df))    
    
    print("\tCleaning questions' content")
    questions_df['Content'] = questions_df['Title'] + '\n' + questions_df['Body']
    questions_df['CleanedContent'] = [clean_text(elem) for elem in questions_df.Content.values]

    print('\tSelecting the answers')
    qids = set(questions_df.Id)
    answers_df = answers_df[answers_df['ParentId'].isin(qids)]
    answers_df.index = range(len(answers_df))

    print('\tSelecting the users')
    users_tokeep = set.union(set(questions_df.OwnerUserId), set(answers_df.OwnerUserId))
    len(users_tokeep)

    users_df = users[users['Id'].isin(users_tokeep)].copy()
    users_df.index = range(len(users_df))
    users_df.shape
    
    print('Number of questions:', len(questions_df))
    print('Number of answers:', len(answers_df))
    print('Number of users:', len(users_df))
    
    questions_df.to_csv(data_dir + 'questions.csv.gz', compression='gzip', index=False)
    answers_df.to_csv(data_dir + 'answers.csv.gz', compression='gzip', index=False)
    users_df.to_csv(data_dir + 'users.csv.gz', compression='gzip', index=False)
    
    return 

def process_data(raw_dir, data_dir):
      
    print('Processing the questions')   
    select_data(raw_dir, data_dir)

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    
    raw_dir = f'../Dataset/{data_name}/data/'
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    
    if not os.path.exists(f'./Dataset/{data_name}/{label}/'):
        os.mkdir(f'./Dataset/{data_name}/{label}/')
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
                
    process_data(raw_dir, data_dir)
  