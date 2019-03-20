import pandas as pd
import os
os.chdir(r"C:\Users\Documents\Tencent Files\1344408330\FileRecv\destinations")
#destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv",chunksize = 1000000)
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()#绘制各变量相关性关系图
plt.figure(figsize=(12, 6))
sns.distplot(df['hotel_cluster'])#绘制各类型酒店浏览及预订量分布图
test_ids = set(test.user_id.unique())
train_ids = set(train.user_id.unique())
intersection_count = len(test_ids & train_ids)
intersection_count == len(test_ids)#检查训练集目和测试数据中用户id总数是否一样
train["date_time"] = pd.to_datetime(train["date_time"])#将data_time列从object转换成datatime
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month#生成新的年和月特征
#选取新的训练和测试数据集
t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]
t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
t2 = t2[t2.is_booking == True]#测试数据集值包含预定事件，需要将t2简化成只包含预定
def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])#赋值0.15表明每一个酒店集群的is_booking为假；
    bookings = len(group.is_booking[group.is_booking == True])#赋值1表明每一个酒店集群的is_booking为真；

    score = bookings + 0.15 * clicks

    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score
#变换这个字典来找到每个srch_destination_id的前五个酒店集群 
import operator
cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top
#预测
preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])
metrics.mapk([[l] for l in t2["hotel_cluster"]], preds, k=5)#初步预测结果
#测试数据集匹配训练数据集的用户
match_cols = ['user_location_cuntry', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']
groups = t1.groupby(match_cols)
def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus
#合并预测
def f5(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
mapk([[l] for l in t2["hotel_cluster"]], full_preds, k=5)
#输出结果
write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(t2["id"][i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_clusters"] + write_frame
with open("predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))
exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))
