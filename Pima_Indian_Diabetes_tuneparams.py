import pandas as pd
import numpy as np
import warnings
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
# 写在前面:设置忽略警告
warnings.filterwarnings('ignore')

# 用pandas读入数据
data = pd.read_csv('pima-indians-diabetes_train.csv')   # read_csv:读入数据
data.head()                                             # 将头一行（即标签行）略去

# 数据切分——分成训练集（train）和测试集（test）
train = data[0:int(len(data) * 9 / 10)]             # 前70%作为训练集
test = data[int(len(data) * 9 / 10):len(data)]      # 后30%作为测试集

# 将数据分成“特征”和“目标”两个部分
feature_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
target_column = 'outcome'

# 数据转换→Dmatrix（xgboost只能处理Dmatrix格式的数据）
xgtrain = xgb.DMatrix(train[feature_columns].values, train[target_column].values)   # 训练用
xgtest = xgb.DMatrix(test[feature_columns].values, test[target_column].values)      # 测试用



# 进行参数优化
other_params = {'eta': 0.3, 'n_estimators': 4, 'gamma': 1,
                'max_depth': 6, 'min_child_weight': 4,
                'colsample_bytree': 1, 'colsample_bylevel': 1,
                'subsample': 1, 'reg_lambda': 3, 'reg_alpha': 1,
                'seed': 0}
# cv_params = {'reg_alpha':np.linspace(-10, 10, 21, dtype=int)}
# regress_model = xgb.XGBRegressor(**other_params)
# gs = GridSearchCV(regress_model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
# gs.fit(train[feature_columns].values, train[target_column].values)
# print('参数的最佳取值：', gs.best_params_)
# print('最佳模型得分：', gs.best_score_)


# 直接就是训练
bst = xgb.train(params=other_params, dtrain=xgtrain)

# 调用训练好的模型，进行预测，并获取其原来target
preds = bst.predict(xgtest)
# print(preds)
labels = xgtest.get_label()

# 二分类的实现
new_labels = [0]*int(len(preds))
for i in range(len(preds)):
    if preds[i] >= 0.5:
        new_labels[i] = 1
# print(new_labels)

# 保存一下(concat:将多个dataframe合成一个dataframe)
The_answer1 = pd.DataFrame(preds, columns=['prediction_labels_pro'])          # 预测的原始值，包含小数
The_answer2 = pd.DataFrame(new_labels, columns=['prediction_labels'])         # 预测值，已经01两极化
The_answer3 = pd.DataFrame(labels, columns=['pro_labels'])               # 预测值的原始标签
The_answer = pd.concat([The_answer1, The_answer2, The_answer3], axis=1)     # axis=1是按照列进行合并
# The_answer.to_csv("Pima_Indian_Diabetes_prediction.csv")

# 模型准确性统计
sum = 0
The_answer2 = np.asarray(The_answer2)   # 必须将dataFrame的数据转换成asarray的数据才能应用到循环
The_answer3 = np.asarray(The_answer3)
for i in range(len(preds)):
    if The_answer2[i] - The_answer3[i] != 0:
        sum = sum + 1
bias = sum/float(len(preds))
print('错误率是%f' % bias)

# # 模型的储存
# pickle.dump(bst, open('pima.pickle.dat', 'wb'))

# 预测模型的准确率
clf = XGBClassifier(max_depth=5, eta=0.1, subsample=0.7, colsample_bytree=0.7)
clf.fit(np.asarray(train[feature_columns].values), np.asarray(train[target_column].values))
accuracy = clf.score(test[feature_columns].values, test[target_column].values)
print('模型准确率:%f' % accuracy)

plus = accuracy + bias
print('模型准确率+错误率：%f' %plus)