from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense, Merge, Permute, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import os
from collections import OrderedDict


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):

    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    #data = parse_stories(f.readlines(), only_supporting=only_supporting)
    ff = open(f,'r')
    data = parse_stories(ff.readlines(), only_supporting=only_supporting)
    ff.close()
#
#    flatten = lambda data: reduce(lambda x, y: x + y, data)
#    
#    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    
    
    return data



print('Extracting Input')
#path = get_file('babi-tasks-v1-2.tar.gz',
#                origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
#tar = tarfile.open(path)

#tar.extractall()


#path ='tasks_1-20_v1-2/en-10k/'
path ='tasks_1-20_v1-2/en/'
challenges = OrderedDict()

challenges[1] = "qa1_single-supporting-fact"
challenges[2] = "qa2_two-supporting-facts"
challenges[3] = "qa3_three-supporting-facts"
challenges[4] = "qa4_two-arg-relations"
challenges[5] = "qa5_three-arg-relations"
challenges[6] = "qa6_yes-no-questions"
challenges[7] = "qa7_counting"
challenges[8] = "qa8_lists-sets"
challenges[9] = "qa9_simple-negation"
challenges[10] = "qa10_indefinite-knowledge"
challenges[11] = "qa11_basic-coreference"
challenges[12] = "qa12_conjunction"
challenges[13] = "qa13_compound-coreference"
challenges[14] = "qa14_time-reasoning"
challenges[15] = "qa15_basic-deduction"
challenges[16] = "qa16_basic-induction"
challenges[17] = "qa17_positional-reasoning"
challenges[18] = "qa18_size-reasoning"
challenges[19] = "qa19_path-finding"
challenges[20] = "qa20_agents-motivations"
challenges[21] = "qa_joint"


print('challenge list')
for i, k in challenges.iteritems() :
    print(`i` + ' : ' + k)


for i in range(1,22) :
    ind_ch = i
    #challenge_type = 'single_supporting_fact_10k'
    #challenge_type = challenges.keys()[ind_ch]
    #challenge_type = challenges[ind_ch]
    challenge = challenges[ind_ch] 
    challenge_file = path + challenge + '_{}.txt'
    
    print('Extracting stories for the challenge:', challenge)
    
    
    #train_stories = get_stories(tar.extractfile(challenge_file.format('train')))
    #test_stories = get_stories(tar.extractfile(challenge_file.format('test')))
    
    train_stories = get_stories((challenge_file.format('train')))
    test_stories = get_stories((challenge_file.format('test')))
    
    #print challenge_file+{}.format('_train.txt')
    #print '{}'.format('_train.txt')
    
    #print (challenge_file.format('a'))
    
    # print train set
    train_sentences = []
    train_queries = []
    train_answers = []
    
    for i, k in enumerate(train_stories) :
        train_sentences.append(train_stories[i][0])
        train_queries.append(train_stories[i][1])
        train_answers.append(train_stories[i][2])
    
    
    
    f_train_set=open(challenge+'_train_set','w')
    
    
    f_train_set.write('\n+NS+\n')
    f_train_set.write(`len(train_stories)`+'\n')
    
    for i in range(len(train_stories)) :
        f_train_set.write('\n+I+\n')
        f_train_set.write(`i`+'\n')
        f_train_set.write('+S+\n')
        f_train_set.write(`len(train_sentences[i])`+'\n')
        for ii in range(len(train_sentences[i])) :
            for iii in range(len(train_sentences[i][ii])) :
                if train_sentences[i][ii][iii] == '.' :
                    #f_train_set.write(train_sentences[i][ii][iii]+'\n')
                    f_train_set.write('\n')
                else :
                    f_train_set.write(train_sentences[i][ii][iii]+' ')
        
        f_train_set.write('+Q+\n')
        for ii in range(len(train_queries[i])-1) :
            f_train_set.write(train_queries[i][ii]+' ')
        
        f_train_set.write('\n+A+\n')
        f_train_set.write(train_answers[i]+'\n')
        
    f_train_set.close()
    
    
    # print test set
    test_sentences = []
    test_queries = []
    test_answers = []
    
    for i, k in enumerate(test_stories) :
        test_sentences.append(test_stories[i][0])
        test_queries.append(test_stories[i][1])
        test_answers.append(test_stories[i][2])
    
    
    f_test_set=open(challenge+'_test_set','w')
        
    f_test_set.write('\n+NS+\n')
    f_test_set.write(`len(test_stories)`+'\n')
    
    for i in range(len(test_stories)) :
        f_test_set.write('\n+I+\n')
        f_test_set.write(`i`+'\n')
        f_test_set.write('+S+\n')
        f_test_set.write(`len(test_sentences[i])`+'\n')
        for ii in range(len(test_sentences[i])) :
            for iii in range(len(test_sentences[i][ii])) :
                if test_sentences[i][ii][iii] == '.' :
                    #f_test_set.write(test_sentences[i][ii][iii]+'\n')
                    f_test_set.write('\n')
                else :
                    f_test_set.write(test_sentences[i][ii][iii]+' ')
        
        f_test_set.write('+Q+\n')
        for ii in range(len(test_queries[i])-1) :
            f_test_set.write(test_queries[i][ii]+' ')
        
        f_test_set.write('\n+A+\n')
        f_test_set.write(test_answers[i]+'\n')
    
    f_test_set.close()


