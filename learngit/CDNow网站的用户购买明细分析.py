#%%
import pandas as pd
import os
from pyecharts import options as opts
from pyecharts.charts import Bar
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
sns.set(style="white", context="talk")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

os.chdir(r'C:\Users\lenovo\Documents\WeChat Files\aiwohaiaini\FileStorage\File\2019-05')
txt = pd.read_table(r'CDNOW.txt',header=None,sep = '\s+',names = ['用户ID','购买日期','订单数','订单金额'])
 #判断列中是否有缺失值
print(txt.isnull().any())
'''商品销售静态'''
#各订单静态数量
#直方图
#%%
plt.hist(txt['订单数'],bins= 100)
plt.xlim([0,20])
plt.show()
#数量
print(txt['订单数'].value_counts())

#各订单静态金额
#直方图

plt.hist(txt['订单金额'],bins= 300)
plt.xlim([0,250])
plt.show()
#数量
a = pd.cut(txt['订单金额'],bins = [0,30,50,80,100,130,150,180,210,250],labels = ['0~30','30~50','50~80','80~100','100~130','130~150','150~180','180~210','210~250']).value_counts()

c = (
        Bar()
        .add_xaxis(a.index.tolist())
        .add_yaxis("数量", a.values.tolist())
        .set_global_opts(title_opts=opts.TitleOpts(title="订单金额"))
    )

c.render('1.html')

'''商品销售动态'''
#订单数量趋势
#%%
from pyecharts.charts import Bar

txt1 = txt[['购买日期','订单数']]
week = ['星期一','星期二','星期三','星期四','星期五','星期六','星期日']
txt1['购买日期'] = txt1['购买日期'].apply(lambda x:week[pd.to_datetime(str(x)).weekday()])
txt1 = txt1.pivot_table(index = '购买日期',aggfunc={'订单数':'sum'}).reset_index()
ax = sns.barplot(txt1['购买日期'],txt1['订单数'],data = txt1,order=['星期一','星期二','星期三','星期四','星期五','星期六','星期日'])
plt.show()


#订单金额趋势
txt1 = txt[['购买日期','订单金额']]
week = ['星期一','星期二','星期三','星期四','星期五','星期六','星期日']
txt1['购买日期'] = txt1['购买日期'].apply(lambda x:week[pd.to_datetime(str(x)).weekday()])
txt1 = txt1.pivot_table(index = '购买日期',aggfunc={'订单金额':'sum'}).reset_index()
ax = sns.barplot(txt1['购买日期'],txt1['订单金额'],data = txt1,order=['星期一','星期二','星期三','星期四','星期五','星期六','星期日'])
plt.show()

#%%
#异常值检测
print(txt.isnull().any())
print(txt.loc[(txt['订单金额'] <=0)& (txt['订单数'] >0)]) #免费的订单



'''用户消费行为静态'''
#%%
#各用户购买数量
#各用户累计购买数量
print(txt.groupby('用户ID')['订单数'].sum().sort_values())
#各用户购买数量累积贡献
print(txt.groupby('用户ID')['订单金额'].sum().sort_values())
#各用户购最大单笔购买数量
#%%
print(txt.groupby('用户ID')['订单数'].max().sort_values())


#各用户购买金额
#%%
#各用户累积购买金额
print(txt.groupby('用户ID')['订单金额'].sum().sort_values())
#各用户最大单笔金额消费
#%%
print(txt.groupby('用户ID')['订单金额'].max().sort_values())
#%%
#各用户购买次数
cc = pd.DataFrame(txt.groupby('用户ID')['订单金额'].count().sort_values()).rename(columns = {'订单金额':'订单次数'})
ax = sns.boxplot(x=cc["订单次数"])
plt.show()



'''用户消费行为动态'''
#%%
#各用户第一次购买时间
cc = pd.DataFrame(txt.groupby('用户ID')['购买日期'].min())
print(cc)
#%%
cc = pd.DataFrame(txt.groupby('用户ID')['购买日期'].max())
print(cc)
#%%
#最近一次消费与当前时间间隔
cc = pd.DataFrame(txt.groupby('用户ID')['购买日期'].max())
now = datetime.now()
cc['据今天时间间隔'] = cc['购买日期'].map(lambda x:(now - (pd.to_datetime(str(x)))).days)
print(cc)
#%%
#生命周期
txt['生命周期'] = txt.groupby('用户ID')['购买日期'].apply(lambda x:(pd.to_datetime(str(max(x))) - pd.to_datetime(str(min(x)))).days).fillna('只购买过一次')
print(txt.head())
print()

#%%
#首次回购周期

def test(x):
	if len(x.values.tolist()) == 1:
		return '只购买了一次'
	else:
		return (pd.to_datetime(str(sorted(x.values.tolist())[1])) - pd.to_datetime(str(min(x.values.tolist())))).days
	print(x)
	print(type(x))
txt['首次回购周期'] = txt.groupby('用户ID')['购买日期'].apply(lambda x:test(x))
print(txt.head())



#%%
#活跃用户和回流用户比率
#活跃用户（购买次数大于5）
cc = pd.DataFrame(txt.groupby('用户ID')['订单金额'].count().sort_values()).rename(columns = {'订单金额':'订单次数'})
print(cc.loc[cc['订单次数'] > 5,'订单次数'].count()/len(txt.groupby('用户ID')['用户ID'].count().values))

#回流用户比率

print(cc.loc[cc['订单次数'] > 1,'订单次数'].count()/len(txt.groupby('用户ID')['用户ID'].count().values))

