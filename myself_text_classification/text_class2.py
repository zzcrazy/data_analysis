train_contents=[]
train_labels=[]
test_contents=[]
test_labels=[]
#  导入文件
import os
import io
from sklearn.feature_extraction.text import TfidfVectorizer

start=os.listdir(r'./train')
for item in start:
    test_path='./test/'+item+'/'
    train_path='./train/'+item+'/'
    for file in os.listdir(test_path):
        with open(test_path+file,encoding="GBK") as f:
            test_contents.append(f.readline())
            #print(test_contents)
            test_labels.append(item)
    for file in os.listdir(train_path):
        with open(train_path+file,encoding='gb18030', errors='ignore') as f:
            train_contents.append(f.readline())
            train_labels.append(item)
# print(train_contents,test_contents,len(train_contents),len(test_contents))
# exit()
# 导入stop word
import jieba
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB  
stop_words = [line.strip() for line in io.open('./stop/stopword.txt').readlines()]
 
# 分词方式使用jieba,计算单词的权重
tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_contents)
print(train_features.shape)
 
##模块 4：生成朴素贝叶斯分类器
# 多项式贝叶斯分类器
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
 
#模块 5：使用生成的分类器做预测
test_tf = TfidfVectorizer(tokenizer=jieba.cut,stop_words=stop_words, max_df=0.5, vocabulary=tf.vocabulary_)
test_features=test_tf.fit_transform(test_contents)
 
print(test_features.shape)
predicted_labels=clf.predict(test_features)
print(metrics.accuracy_score(test_labels, predicted_labels))