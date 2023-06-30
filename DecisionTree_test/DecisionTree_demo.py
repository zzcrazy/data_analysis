import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
#决策树 预测泰坦尼克的生存
# 数据加载
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# 数据探索
# print(train_data.info())
# print('-'*30)
# print(train_data.describe())
# print('-'*30)
# print(train_data.describe(include=['O']))
# print('-'*30)
# print(train_data.head())
# print('-'*30)
# print(train_data.tail())
# 数据清洗
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
# print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
# test_labels = test_data['Survived']

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
test_features = dvec.transform(test_features.to_dict(orient='record'))

# print(111,dvec.feature_names_)

# 弱分类器
dt_stump = DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
dt_stump.fit(train_features, train_labels)
# dt_stump_err = 1.0-dt_stump.score(test_features, test_labels)

# 构造 ID3 决策树
clf = DecisionTreeClassifier()
# 决策树训练
clf.fit(train_features, train_labels)
# test_features=dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
# pred_labels = clf.predict(test_features)
# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score 准确率为 %.4lf' % acc_decision_tree)
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))

####################abaBoost分类器###################
# 使用AdaBoost回归模型
n_estimators=200
ada = AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimators)
ada.fit(train_features, train_labels)
# print("clf train precision = ", round(ada.score(train_features, train_labels),6))
ada_score = np.mean(cross_val_score(ada, train_features, train_labels, cv=10))
print("AdaBoost分类器准确率为:%.4lf" % ada_score)