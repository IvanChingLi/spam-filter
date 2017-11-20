##read csv file 
import csv
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import math

def preprocess(readfilename, writefilename):
    print("Preprocessing...")
    reader = csv.reader(open(readfilename))
    writer = open(writefilename,'w')
    line_num = 0
    next(reader)
    labels = []
    messages=[]
    #test_labels = []
    
    for row in reader:
        line_num += 1
        #print line_num
        if line_num % 500 == 0:
            print(line_num)
        temp_label = row[0]
        temp_text = row[1]
        #get the train label list
        if temp_label == 'spam':
            labels.append(1)
        else:
            labels.append(0)
        #Make the words to lower format and Remove the stopwords
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(temp_text)
        words_lower = [w.lower() for w in words]
        words_lower_filter_stopwords = []
        for w in words_lower:
            if w not in stopWords:
                words_lower_filter_stopwords.append(w)
        #print words_lower_filter_stopwords
        word_num = 0
        temp_sentence = ""
        for temp_word in words_lower_filter_stopwords:
            word_num += 1
            if word_num == 1:
                temp_sentence += temp_word
            else:
                temp_sentence += " " + temp_word
        temp_sentence += "\n"
        messages.append(temp_sentence)
#        print(temp_sentence)
        writer.write(temp_sentence)
    writer.close()
    print("Preprocessing is done!")
    return labels, messages
labels, messages =preprocess('train.csv','train_p.csv')
        
#tokenize the texts and apply tf-idf transform
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=3,decode_error='ignore')
X = vectorizer.fit_transform(messages)
sms_array = X.toarray()
vocab = vectorizer.vocabulary_
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False, norm='l2')
tfidf = transformer.fit_transform(sms_array)



# define individual likelihoods
def y(weights, feature):
#    try:
    result = 1/(1+math.exp(-feature.dot(weights)[0]))
#    except OverflowError:
#        if feature.dot(weights)[0]>0:
#            result = 1
#        else:
#            result = 0
    return result
# define error function
def error(weights, feature_matrix, labels):
    result = 0
    for i in range(tfidf.shape[0]):
        result+= labels[i]*math.log(y(weights, feature_matrix.getrow(i))) + (1-labels[i])*math.log(1-y(weights,feature_matrix.getrow(i)))
#        print('using k=',k)
    return -result + (k/2)*np.linalg.norm(weights)**2
## define gradient of error function
def grad(weights, feature_matrix, labels):
    result = [np.zeros(tfidf.shape[1])]
#    for i in range(tfidf.shape[0]):
    for i in range(tfidf.shape[0]):
        result = np.add(result, (y(weights,feature_matrix.getrow(i)) - labels[i])*np.array(feature_matrix.getrow(i).toarray())[0])
#    return np.add(result, k*np.array(weights))
    return result+k*np.array(weights)[0]
## define changing step size
def step(step0,t):
    return step0*t**(-0.9)


## Implement batch gradient descent logistic regression
k = 2 #k is regularization hyperparameter
step0 = 2 # step0 is also a hyperparameter
threshold = 1e-2  #1e-8 is typical threshold, can tune

def GD():
    print('starting GD with hyperparameter k=',k,'and step hyperparameter step0=',step0)
    loc = np.random.rand(tfidf.shape[1])
    t=1
    past_error = 0  #assign numbers to start while loop
    current_error = error(loc,tfidf,labels)
    while abs(past_error-current_error)/current_error > threshold:
        loc = loc - step(step0,t)*np.array(grad(loc, tfidf,labels))[0]
        loc = loc*10/np.linalg.norm(loc)
        if t%10==0:
            past_error = current_error
            current_error = error(loc,tfidf,labels)
#            print('grad has norm: ', np.linalg.norm(grad(loc,tfidf,labels)))
#            print('moving by this distance: ',np.linalg.norm(step(step0,t)*np.array(grad(loc, tfidf,labels))[0]))
#            print('current error is :',error(loc,tfidf,labels))
#            print('update: fractional error is: ',abs(past_error-current_error)/current_error)
#            print('first weight.feature has value ',tfidf.getrow(0).dot(loc)[0])
        t+=1
    print('Final error is ',error(loc,tfidf,labels))
#    print('Final weights are ',loc)
    ## Find accuracy (true positive, true negative, false positive, false negative)
    t_pos_alg = 0
    t_neg_alg = 0
    f_pos_alg = 0
    f_neg_alg = 0
    for i in range(len(labels)):
        if y(loc,tfidf.getrow(i)) > .5:
            t_pos_alg += labels[i]
            f_pos_alg += 1-labels[i]
        else:
            t_neg_alg += 1-labels[i]
            f_neg_alg += labels[i]
#    print('True proportion of ham is ',labels.count(1)/len(labels))
#    print('True proportion of spam is ',labels.count(0)/len(labels))
#    print('True positive proportion is ', t_pos_alg/len(labels))
#    print('False positive proportion is ', f_pos_alg/len(labels))
#    print('True negative proportion is ', t_neg_alg/len(labels))
#    print('False negative proportion is ', f_neg_alg/len(labels))
    print('Accuracy = ', (t_pos_alg+t_neg_alg)/len(labels))
    return [loc, (t_pos_alg+t_neg_alg)/len(labels)]
    
#k=1
#step0=1
#loc=GD()
    
#GD()
    
    
#reg_para = []
#step_para = []
#weights = []
#acc = [] #accurary
#
#for i in range(-5,5):
#    k=2**i
#    for j in range(-5,5):
#        step0=2**j
#        try:
#            result = GD()
#            acc = result[1]
#            weights.append(result)
#            reg_para.append(k)
#            step_para.append(step0)
#        except OverflowError:
#            print('overflow')
#        except ValueError:
#            print('math domain error')

########################################## now apply model to test data
    
def testGD(loc,train_vocab):
    labels, messages =preprocess('test.csv','test_p.csv')
                
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(decode_error='ignore',vocabulary=train_vocab)
    X = vectorizer.fit_transform(messages)
    sms_array = X.toarray()
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False,norm='l2')
    tfidf = transformer.fit_transform(sms_array)
    
    t_pos_alg = 0
    t_neg_alg = 0
    f_pos_alg = 0
    f_neg_alg = 0
    for i in range(len(labels)):
#        print(i)
#        print(tfidf.getrow(i).shape)
#        print(loc.shape)
#        print(y(loc,tfidf.getrow(i)))
        if y(loc,tfidf.getrow(i)) > .5:
            t_pos_alg += labels[i]
            f_pos_alg += 1-labels[i]
        else:
            t_neg_alg += 1-labels[i]
            f_neg_alg += labels[i]
    #print('True proportion of ham is ',labels.count(1)/len(labels))
    #print('True proportion of spam is ',labels.count(0)/len(labels))
    #print('True positive proportion is ', t_pos_alg/len(labels))
    #print('False positive proportion is ', f_pos_alg/len(labels))
    #print('True negative proportion is ', t_neg_alg/len(labels))
    #print('False negative proportion is ', f_neg_alg/len(labels))
    print('Accuracy is ',(t_pos_alg+t_neg_alg)/len(labels))


loc = GD()[0]
#print(loc)
testGD(loc,vocab)
