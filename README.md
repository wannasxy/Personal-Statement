# Personal-Statement
我的项目
import numpy as np
import pandas as pd
import os
os.chdir('C:\\Users\\Documents\\Tencent Files\\1344408330\\FileRecv\\python评分卡\\data\\')
import matplotlib.pyplot as plt
%matplotlib inline
df =pd.read_csv("cs-training.csv")
df.info()
df.describe(include= "all")
df.isnull().sum()
import missingno as msno
msno.matrix(df)；#查看变量缺失分布图
train_data = df.iloc[:,1:]
median = train_data.MonthlyIncome.median()#
median
train_data.MonthlyIncome.fillna(value=median, inplace=True)#用收入的中位数去填补收入得缺失值
train_data.dropna(inplace=True)
train_data.isnull().mean()
train_data = train_data.loc[train_data.age>0]
import matplotlib.pyplot as plt    
columns = ["NumberOfTime30-59DaysPastDueNotWorse",
"NumberOfTime60-89DaysPastDueNotWorse",
"NumberOfTimes90DaysLate"]
train_data[columns].plot.box(vert=False);    
for col in columns:
    train_data = train_data.loc[train_data[col] < 90]    #去除逾期次数大于90的异常值
from sklearn.model_selection import train_test_split
Y = train_data['SeriousDlqin2yrs']
X = train_data.iloc[:, 1:]
X_train, X_vali, Y_train, Y_vali = train_test_split(X, Y, test_size=0.3)
model_data = pd.concat([Y_train, X_train], axis=1)
vali_data = pd.concat([Y_vali, X_vali], axis=1)#拆分数据集
from auto_bin import AutoBins#auto_bin为实现最优分箱化所调用的包
bins_data = AutoBins(model_data, "SeriousDlqin2yrs")
model_data.columns
num_bins, woe_df, iv = bins_data.auto_bins("age",n=5)
#num_bins, woe_df, iv = bins_data.auto_bins("RevolvingUtilizationOfUnsecuredLines",n = 5)
#num_bins, woe_df, iv = bins_data.auto_bins("NumberOfTime30-59DaysPastDueNotWorse",n=3)
#num_bins, woe_df, iv = bins_data.auto_bins("DebtRatio",n=5)
bins_num = {
"RevolvingUtilizationOfUnsecuredLines":5,
"age":5,
"NumberOfTime30-59DaysPastDueNotWorse":3,
"DebtRatio":5,
"MonthlyIncome":6,
"NumberOfOpenCreditLinesAndLoans":4,
"NumberOfTimes90DaysLate":2,
"NumberRealEstateLoansOrLines":4,
"NumberOfTime60-89DaysPastDueNotWorse":2,
"NumberOfDependents":4,
}#最优分箱后变量分组数
info_values = {}
woe_values = {}
bins_values = {}
for key in bins_num:
    num_bins, woe_df, iv = bins_data.auto_bins(key, n=bins_num[key], show_iv=False)
    info_values[key] = iv
    woe_values[key] = woe_df
    bins_values[key] = [x[0] for x in num_bins] + [float("inf")]
def plt_iv(info_values):
    keys,values = zip(*info_values.items())
    nums = range(len(keys))
    plt.barh(nums,values)
    plt.yticks(nums,keys)
    for i, v in enumerate(values):
        plt.text(v, i-.2, "{:.2f}".format(v))
plt_iv(info_values)
model_woe = pd.DataFrame(index=model_data.index)
for col in bins_values:
    bins = bins_values[col]
    labels = woe_values[col]["woe"]
    model_woe[col] = pd.cut(model_data[col], bins, labels=labels).astype(np.float)#对各变量进行分箱
model_woe["SeriousDlqin2yrs"] = model_data["SeriousDlqin2yrs"]
model_woe.head(5)
model_woe.to_csv('WoeData.csv',encoding="utf8", index=False)
import statsmodels.api as sm
data = pd.read_csv('WoeData.csv',encoding="utf8")
endog = data['SeriousDlqin2yrs']
X = data.drop(["SeriousDlqin2yrs",
               "NumberRealEstateLoansOrLines",
               "NumberOfDependents"],axis=1)
exog = sm.add_constant(X)
logit = sm.Logit(endog,exog)
result = logit.fit()
result.summary()#评分卡
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
vali_woe = pd.DataFrame(index=vali_data.index)
for col in bins_values:
    bins = bins_values[col]
    labels = woe_values[col]["woe"]
    vali_woe[col] = pd.cut(vali_data[col], bins, labels=labels).astype(np.float)
vali_woe["SeriousDlqin2yrs"] = vali_data["SeriousDlqin2yrs"]
vali_Y = vali_woe['SeriousDlqin2yrs']
vali_X = vali_woe.drop(["SeriousDlqin2yrs",
"NumberRealEstateLoansOrLines",
"NumberOfDependents"],axis=1)#对测试集进行最优分箱
vali_exog = sm.add_constant(vali_X)
vali_proba = result.predict(vali_exog)#测试集预测结果
import scikitplot as skplt
vali_proba_df = pd.DataFrame(vali_proba,columns=[1])
vali_proba_df.insert(0,0,1-vali_proba_df)
skplt.metrics.plot_roc(vali_Y,
                        vali_proba_df,
                        plot_micro=False,
                        plot_macro=False);#绘制auc图
result.params#各变量系数
B = 20/np.log(2)
A = 600 + B*np.log(1/60)
base_score = A - B*result.params["const"]
base_score#计算基础分
# 将评分卡写入文件
file = "ScoreData.csv"
with open(file,"w") as fdata:
    fdata.write("base_score,{}\n".format(base_score))
for col in result.params.index[1:]:
    temp = woe_values[col]
    score = temp["woe"] * (-B*result.params[col])
    score.name = "Score"
    score.index = [("{:.2f}".format(temp.loc[x, "min"]),"{:.2f}".format(temp.loc[x, "max"])) for x in temp.index]
    score.index.name = col
    score.to_csv(file,header=True,mode="a")
