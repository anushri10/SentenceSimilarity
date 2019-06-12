from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from networkx.algorithms.components.connected import connected_components
import os
import json
import warnings
import networkx
import time
import numpy as np
import pylab as p

THRESHOLD_SCORE = 0.4

def toGraph(l):
    '''
    It takes in a list of lists and returns a graph object, 
    assigning nodes and edges from each sub-list object
    '''
    G = networkx.Graph()
    
    for part in l:
        G.add_nodes_from(part)
        G.add_edges_from(toEdges(part))
    return G

def toEdges(l):
    '''
    It treats args(1) 'l' as a graph and returns (implicitly) it's edges 
    '''
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current 

def similarityIndex(s1, s2, wordmodel):
    '''
    To compare the two sentences for their similarity using the gensim wordmodel 
    and return a similarity index
    '''
    if s1 == s2:
        return 1.0

    s1words = s1.split()
    s2words = s2.split()

    s1words = set(s1words)    
    for word in s1words.copy():
        if word in stopwords.words('english'):
            s1words.remove(word)
    
    s2words = set(s2words)
    for word in s2words.copy():
        if word in stopwords.words('english'):
            s2words.remove(word)

    s1words = list(s1words)
    s2words = list(s2words)    

    s1set = set(s1words)
    s2set = set(s2words)

    vocab = wordmodel.vocab
    
    if len(s1set & s2set) == 0:
        #print(s1set,'\n',s2set)
        return 0.0
    for word in s1set.copy():
        if (word not in vocab):
            s1words.remove(word)
        if (len(s1words))==0:
            print("Sentence with no words in vocab: ", s1set,'\n')
            return 0.0
    for word in s2set.copy():
        if (word not in vocab):
            s2words.remove(word)
        if(len(s2words))==0:
            print("Sentence with no words in vocab: ", s2set,'\n')
            return 0.0
    return wordmodel.n_similarity(s1words, s2words)


def categorizer():
    '''
    driver function,
    returns model output mapped on the input corpora as a dict object
    '''
    stats = open('stats.txt', 'w', encoding='utf-8')

    st = time.time()
    wordmodelfile='/Users/Anushri-MacBook/GoogleNews-vectors-negative300.bin.gz'
    #wordmodelfile = 'E:/Me/IITB/Work/CIVIS/ML Approaches/word embeddings and similarity matrix/GoogleNews-vectors-negative300.bin'
    wordmodel = KeyedVectors.load_word2vec_format(wordmodelfile, binary = True, limit=200000)
    et = time.time()
    s = 'Word embedding loaded in %f secs.' % (et-st)
    print(s)
    stats.write(s + '\n')

    #filepaths
    responsePath='./comments/'
    categoryPath='./comments/'
    #responsePath = './data/comments/'
    #categoryPath = './data/sentences/'
    responseDomains = os.listdir(responsePath)
    categoryDomains = os.listdir(categoryPath)
    
    #dictionary for populating the json output
    results = {}
    results2={}
    for responseDomain, categoryDomain in zip(responseDomains, categoryDomains):
        #instantiating the key for the domain
        #print("Response Domain is: ", responseDomain,'\n')
        domain = responseDomain[:-4]
        results[domain] = {}

        print('Categorizing %s domain...' % domain)

        temp = open(responsePath + responseDomain, 'r', encoding='utf-8-sig')
        responses = temp.readlines()
        #print("Rows length: ", len(responses),'\n')
        #print("Responses are: ",responses,'\n')
        rows = len(responses)

        temp = open(categoryPath + categoryDomain, 'r', encoding='utf-8-sig')
        categories = temp.readlines()
        #print("Columns length: ", len(categories),'\n')
        columns = len(categories)
        categories.append('Novel')

        #saving the scores in a similarity matrix
        #initializing the matrix with -1 to catch dump/false entries
        st = time.time()
        similarity_matrix = [[-1 for c in range(columns)] for r in range(rows)]
        et = time.time()
        s = 'Similarity matrix initialized in %f secs.' % (et-st)
        print(s)
        stats.write(s + '\n')

        row = 0
        st = time.time()
        for response in responses:
            print("Row: ",row,'\n')
            #print("Response is: ",response,'\n')
            column = 0
            for category in categories[:-1]:
                #print("Category is: ",category,'\n')
                #print("Row: "+str(row)+"Column: "+str(column),'\n')
                #response.split('-')[1].lstrip()
                if response!=category:
                    similarity_matrix[row][column] = similarityIndex(response, category, wordmodel)
                #print(similarity_matrix[row][column]," ")
                else:
                    similarity_matrix[row][column]=0.0
                column += 1
                #print("Row, Column: ", row," ", column,'\n')
            row += 1
        et = time.time()
        s = 'Similarity matrix populated in %f secs. ' % (et-st)
        print(s)
        stats.write(s + '\n')

        print('Initializing json output...')
        for catName in categories:
            results[domain][catName] = []

        print('Populating category files...')
        for score_row,response in zip(similarity_matrix,responses):
            #print("Score_Row: ",score_row,'\n')
            #print("Length of score_row: ",len(score_row),'\n')
            #print('Response: ',response,'\n')
            max_sim_index = len(categories)-1
            if np.array(score_row).sum() > 0:
                max_sim_index = np.array(score_row).argmax()
                #print("Max sim index: ",max_sim_index,'\n')
                temp = {}
                results2[response]=categories[max_sim_index]
                #temp = response
                temp['response'] = response
                temp['score'] = np.array(score_row).max()
            else:
                results2[response]='Novel'
                temp = response
            results[domain][categories[max_sim_index]].append(temp)
            #print("Category: ",categories[max_sim_index],'\n')
            #print("Result: ", results[domain][categories[max_sim_index]],'\n')
        #print("New result: \n",results2,'\n')
        print('Completed.\n')
        
        #converting results2 to a list of tuples
        setlist=[]
        result_final=[]
        G = toGraph(list(results2.items()))
        setlist = list(connected_components(G))
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        for cluster in setlist:
            print(cluster,'\n')
            result_final.append(list(cluster))


        #sorting domain wise categorised responses based on scores
        for domain in results:
            #print("Domain is: ",domain, '\n')
            for category in results[domain]:  
                #print("Category is: ",category,'\n')                                                                                                                                             
                temp = results[domain][category]
                #print("Result for category: ",temp,'\n')
                if len(temp)==0 or category=='Novel':
                    continue
                #print(temp)
                results[domain][category] = sorted(temp, key=lambda k: k['score'], reverse=True)
        #newlist = sorted(list_to_be_sorted, key=lambda k: k['name']) --> to sort list of dictionaries
        
        
        
        
        '''
      
        temp=[]
        count = 0
        for category in results['sentence']:
            count+=1
            print("Category is : ",category,'\n')
            for val in results[domain][category]:
                print("Values in category are: ", val,'\n')
                    #temp.append(val['response'])
            #G = toGraph(temp)
            #temp = list(connected_components(G))
            #print("List: \n",temp)
        print("Total number of categories are: ",count)
        
        setlist=[]
        for score_row,response in zip(similarity_matrix,responses):
            print('Response: ',response,'\n')
            max_sim_index = len(categories)-1
            print(np.array(score_row).sum())
            if np.array(score_row).sum() > 0:
                max_sim_index = np.array(score_row).argmax()
                sublist=[]
                #sublist.append(response)
                for data in results[domain][categories[max_sim_index]]:
                    sublist.append(data['response'])
                    #print ("Text in that set: ", sublist,'\n')
                    print("Text in that category: ", data['response'],'\n')
                #temp = {}
                #temp = response
                #temp['response'] = response
                #temp['score'] = np.array(score_row).max()
                #print("Val in results: ",results[domain][categories[max_sim_index]],'\n')
            #print ("Text in that set: ", sublist,'\n')
            if set(sublist) not in setlist:
                setlist.append(sublist)
        #print(setlist,'\n')
        G = toGraph(setlist)
        setlist = list(connected_components(G))
        print(len(setlist))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print("RESULT:/n",setlist)

        
        #initializing the matrix with -1 to catch dump/false entries for subcategorization of the novel responses
        no_of_novel_responses = len(results[domain]['Novel'])
        st = time.time()
        similarity_matrix = [[-1 for c in range(no_of_novel_responses)] for r in range(no_of_novel_responses)]
        et = time.time()
        s = 'Similarity matrix for subcategorization of novel responses for %s domain initialized in %f secs.' % (domain, (et-st))
        print(s)
        stats.write(s + '\n')
        

        #populating the matrix
        row = 0
        for response1 in results[domain]['Novel']:
            column = 0
            for response2 in results[domain]['Novel']:
                if response1 == response2:
                    column += 1
                    continue
                similarity_matrix[row][column] = similarityIndex(response1, response2, wordmodel)
                column += 1
            row += 1
        
        setlist = []
        index = 0
        for score_row, response in zip(similarity_matrix, results[domain]['Novel']):
            #print("Response is: ",response,'\n')
            #print("Score for row is: ",score_row,'\n')
            #print("Length of score_row: ",len(score_row),'\n')
            max_sim_index = index
            if np.array(score_row).sum() > 0:
                max_sim_index = np.array(score_row).argmax()
                #print("Max sim index: ",max_sim_index,'\n')
            #print("Set thus seen: ",set([response, results[domain]['Novel'][max_sim_index]]),'\n' )
            if set([response, results[domain]['Novel'][max_sim_index]]) not in setlist:
                setlist.append([response, results[domain]['Novel'][max_sim_index]])
            index += 1
        
        G = toGraph(setlist)
        setlist = list(connected_components(G))
        
        novel_sub_categories = {}
        index = 0
        for category in setlist:
            novel_sub_categories[index] = list(category)
            index += 1

        results[domain]['Novel'] = novel_sub_categories
        '''
    print('***********************************************************')

    with open('out_new_3.json', 'w') as temp:
        json.dump(result_final, temp)

    with open('out_new_2.json', 'w') as temp:
        json.dump(results, temp)
    return results

if __name__=="__main__":
    results = categorizer()
    print(len(results['sentence']))
    #G= toGraph(results['sentence'])
    #networkx.draw(G)
    #p.show()
    