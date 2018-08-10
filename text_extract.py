#preprocessing the data
import json
from nltk.corpus import stopwords
import csv
import tensorflow as tf
import numpy as np
from pathlib import Path
import gensim
from sklearn.preprocessing import LabelEncoder
'''
1) load the word2vec model
2)preprocess the data
3) convert labels to onehot encoding
4) train_x to an array on numpy shape(10,300) (10 examples with each column of 300) 
(first convert to an numpy matrix then transpose it)
5) train_x -> neuralnet()
6) convert to one_hot both labels train and test
'''
w2v_model = None

def activate_model():
    global w2v_model
    '''Loding the model which is stored in the same folder'''
    w2v_model = gensim.models.KeyedVectors.load('Gword2vec.model')
    w2v_model.init_sims(replace=True)

def PhraseToVec(phrase):
    cachedStopWords = stopwords.words("english")
    phrase = phrase.lower()
    '''Converting the words to lower and removing the stopwords
        Then processing each sentence as a list of words and getting their respective vectors with Word2vec model
        Doing a mean on the entire sentence 
    '''
    wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
    vectorSet = []
    for aWord in wordsInPhrase:
        try:
            wordVector = w2v_model[aWord]
            vectorSet.append(wordVector)
        except:
            pass
    return np.mean(vectorSet, axis=0)

'''Loading the file via path'''
def extract_data():
    path = r'C:\Users\jatoth.kumar\Desktop\machine-learning-assessment\data\docs'
    pathlist = Path(path).glob('**/*.json')  #search for only json files
    '''
    id_no : list which stores the id nos 
    text1: stores the JD if it is not empty
    rem: Storing the remaining id no.s to remove
    '''
    id_no = []
    text1 = []
    rem = []
    for i in pathlist:
        x = json.loads(open(i).read())
        y = (x['jd_information'])
        if len(str(y['description']).split()) > 3 :
            text1.append(str(y['description']).replace('&nbsp;',' '))
            id_no.append(x['_id'])
        else:
           rem.append(x['_id'])
    document_id = []
    dept = []
    '''
    Getting the Department ID's from the excel sheet (labels)
    here we use the rem list which we see the ID number in the rem list 
    document_id ; defines the Document ID list
    dept : which stores the department name
    '''
    for d in csv.DictReader(open(r'C:\Users\jatoth.kumar\Desktop\machine-learning-assessment\data\document_departments.csv')):
        y2 = d['Document ID']
        if y2 not in rem:
            document_id.append(d['Document ID'])
            dept.append(d['Department'])
    train_y_act = []
    print('printing the values of document and dept')
    print(len(document_id),len(dept))
    for i in range(len(text1)):
        ind = ''
        ind = dept[document_id.index(id_no[i])]
        train_y_act.append(ind)
    train_y_act = set(train_y_act)   #total training examples 741 size
    x4 = []
    [x4.append(i) for i in train_y_act]  #set 30 values x4
    train_y = []
    [train_y.append(i) for i in train_y_act]
    train_y = np.array(train_y)
    label = LabelEncoder()
    train_y = label.fit_transform(train_y)
    x9 = tf.keras.utils.to_categorical(train_y,num_classes=30)  #contains 30 one hot encodings
    final = []   #will contain the train values 741
    for i in range(len(dept)):
        a = x4.index(dept[i])
        final.append(x9[a])
    train_y = final
    train_x_using,train_y_using = text1[:700],train_y[:700]
    test_x_using,test_y_using = text1[700:],train_y[700:]
    final_matrix_input = []
    test_matrix_input = []
    [final_matrix_input.append(PhraseToVec(train_x_using[b])) for b in range(len(train_x_using))]
    [test_matrix_input.append(PhraseToVec(test_x_using[j])) for j in range(len(test_x_using))]
    # print(len(final_matrix_input),len(test_matrix_input),len(final_matrix_input[0]),len(test_matrix_input[0]))
    test_matrix_input_1 = np.asmatrix(test_matrix_input)
    train_matrix_input_1 = np.asmatrix(final_matrix_input)
    # print("the shapes of train and test inputs")
    # print(train_matrix_input_1.shape,test_matrix_input_1.shape)
    train_y_using = np.asmatrix(train_y_using)
    test_y = np.asmatrix(test_y_using)
    return train_matrix_input_1,train_y_using,test_matrix_input_1,test_y
def actual():
    activate_model()
    train_x,train_y,test_x,test_y = extract_data()
    return train_x,train_y,test_x,test_y
