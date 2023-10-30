import re
from bs4 import BeautifulSoup
import pandas as pd
import os
from lxml import etree
from collections import OrderedDict
from Utils import parse_arguments

def clean_text(text):
    return BeautifulSoup(text, features = "html.parser").get_text()

def split_post(raw_dir, data_dir, start_date, end_date):
    """ Split the post

    Split post to question and answer,
    keep all information, output to file

    Args:
        data_dir - data directory
    """
    if os.path.exists(data_dir + "extracted_questions.csv.gz") \
        and os.path.exists(data_dir + "extracted_answers.csv.gz"):
            
        print("\tPosts already splitted")        
        questions = pd.read_csv(data_dir + 'extracted_questions.csv.gz', compression='gzip', dtype=str)
        answers = pd.read_csv(data_dir + 'extracted_answers.csv.gz', compression='gzip', dtype=str)
        return questions, answers

    questions = OrderedDict()
    n_questions = 0 
    
    answers = OrderedDict()
    n_answers = 0 
    
    parser = etree.iterparse(raw_dir + 'Posts.xml', events=('end',), tag='row')
    print('\tStarting split')
    for event, elem in parser:
        attr = dict(elem.attrib)

        if attr['CreationDate']>=end_date:
            break

        if attr['CreationDate']>=start_date and attr['CreationDate']<end_date:
            
            if attr['PostTypeId'] == '1':
                attr['Tags'] = list(filter(None,re.split(r"<|>",attr['Tags'])))
                questions[n_questions] = attr
                n_questions+=1

            elif attr['PostTypeId'] == '2':
                answers[n_answers] = attr
                n_answers+=1
    
    questions = pd.DataFrame.from_dict(questions, orient='index')
    questions = questions[['Id', 'OwnerUserId', 'AcceptedAnswerId', 'Tags', 'Body', 'Title', 'CreationDate']]
    questions.to_csv(data_dir + 'extracted_questions.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')
    
    answers = pd.DataFrame.from_dict(answers, orient='index')
    answers = answers[['Id', 'OwnerUserId', 'ParentId', 'CreationDate']]
    answers.to_csv(data_dir + 'extracted_answers.csv.gz', compression='gzip', index=False)#, index=True, index_label = 'Index')
                
    return

def extract_users(raw_dir, data_dir):
    
    if os.path.exists(data_dir + "extracted_users.csv.gz"):
        print("\tUsers already extracted")
        users = pd.read_csv(data_dir + 'extracted_users.csv.gz', compression='gzip', dtype=str)#, index_col='Index')
        return users
    
    users = OrderedDict()
    n_users = 0
    
    parser = etree.iterparse(raw_dir + 'Users.xml', events=('end',), tag='row')
    for event, elem in parser:
        attr = dict(elem.attrib)
        users[n_users] = attr
        n_users+=1
    
    users = pd.DataFrame.from_dict(users, orient='index')
    users = users[['Id', 'Reputation']]
    users.to_csv(data_dir + 'extracted_users.csv.gz', compression='gzip', index=False)
    
    return users

def select_data(data_dir):
    
    questions = pd.read_csv(data_dir + 'extracted_questions.csv.gz', compression='gzip')
    answers = pd.read_csv(data_dir + 'extracted_answers.csv.gz', compression='gzip')
    users = pd.read_csv(data_dir + 'extracted_users.csv.gz', compression='gzip')
    
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
    
    return questions_df, answers_df, users_df

def split_train_test(data_dir, questions, answers, users):
    
    train_size = int(len(questions)*0.8)
    train = questions[:train_size]
    train.index = range(len(train))
    
    test = questions[train_size:]
    test.index = range(len(test))

    a_train = answers[answers['ParentId'].isin(train.Id)]
    a_train.index = range(len(a_train))

    a_test = answers[answers['ParentId'].isin(test.Id)]
    a_test.index = range(len(a_test)) 
        
    print('\tTrain size:',len(train))
    print('\tTest size:',len(test))

    uids = set.union(set(train.OwnerUserId.values), set(a_train.OwnerUserId.values))
    u_train = users[users['Id'].isin(uids)]
    u_train.index = range(len(u_train))
    len(users), len(u_train)
    
    train.to_csv(data_dir + 'train.csv.gz', compression='gzip', index=False)
    test.to_csv(data_dir + 'test.csv.gz', compression='gzip', index=False)
    a_train.to_csv(data_dir + 'answers_train.csv.gz', compression='gzip', index=False)
    a_test.to_csv(data_dir + 'answers_test.csv.gz', compression='gzip', index=False)
    u_train.to_csv(data_dir + 'users.csv.gz', compression='gzip', index=False)
    
    return

def process_data(raw_dir, data_dir, start_date, end_date):
    
    print('Splitting questions and answers')
    #split_post(data_dir, '2020-07-01 00:00:00', '2021-01-01 00:00:00')
    #split_post(data_dir, '2008-01-01 00:00:00', '2024-01-01 00:00:00')
    split_post(raw_dir, data_dir, start_date, end_date)
    print('Extracting users')
    extract_users(raw_dir, data_dir)    
    print('Processing the questions')   
    questions, answers, users = select_data(data_dir)
    split_train_test(data_dir, questions, answers, users)

if __name__ == "__main__":
    
    args = parse_arguments()
    
    data_name = args.dataset
    label = args.label
    start_date = args.startdate
    end_date = args.enddate
    
    raw_dir = f'./Dataset/{data_name}/data/'
    data_dir = f'./Dataset/{data_name}/{label}/data/'
    
    if not os.path.exists(f'./Dataset/{data_name}/{label}/'):
        os.mkdir(f'./Dataset/{data_name}/{label}/')
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
                
    process_data(raw_dir, data_dir, start_date, end_date)
  