import os
import jieba
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

# 多项式贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB  
tfidf_vec = TfidfVectorizer()

label_map={'体育':0,'女性':1,"文学":2,'校园':3}
stop_words=set()

with open("./stop/stopword.txt",'rb') as f:
    stop_words = [line.strip() for line in f.readlines()]
# stop_words = open('./stop/stopword.txt', 'r', encoding='utf-8').read()
# stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
# stop_words = stop_words.split('\n') # 根据分隔符分隔
# print(stop_words)
# exit()
def LoadData(file):
    documents=[]
    labels=[]

    for root,dirs,files in os.walk(file):
        # print(root,dirs,files)
        for file_itr in files:
            #./train/分类名
            w_label = root.split('/')[-1]
            labels.append(w_label)
            filename = os.path.join(root,file_itr)

            with open(filename ,'rb') as f:
                content = f.read()
                wordlist=list(jieba.cut(content))
                # words = [item for item in wordlist if item not in stop_words]
                words = [item for item in wordlist]
                documents.append(''.join(words))
    return documents,labels


def train_fun(td,tl ,testd,testl):

    tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5)
    features = tf.fit_transform(td)
    print(features.shape)
 
    # 训练模型
    clf = MultinomialNB(alpha=0.001).fit(features, tl)

     # 模型预测
    test_tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5, vocabulary=tf.vocabulary_)
    test_features = test_tf.fit_transform(testd)
    print(test_features.shape)
    predicted_labels = clf.predict(test_features)
    # 获取结果
    x = metrics.accuracy_score(testl, predicted_labels)
    return x

if __name__ =="__main__":
    print(23232)
    d,l = LoadData('./train')
    ttd,ttl = LoadData('./test')
    # print(d,len(d))
    # exit()
    xx = train_fun(d,l,ttd,ttl)
    # print( d,l ,ttd,ttl)
    print(xx)
