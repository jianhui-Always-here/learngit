import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(r'C:\Users\lenovo\Desktop')

# data = np.arange(10)
# plt.plot(data)
# plt.show()



'''matplotlib的图像都在Figure对象中，你可以使用plt.figure创建一个新的Figure'''
# fig = plt.figure()

'''不能通过空Figure绘图。必须用add_subplot创建一个或多个subplot才行'''
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)

# ax3 = fig.add_subplot(2,2,3)
'''如果执行一条绘图命令，matplotlib就会在最后一个用过的subplot上进行绘制。'''
# plt.plot(np.random.randn(50).cumsum(),'k--')

'''ax1,ax2,ax3是AxesSubplot对象，直接调用他们的实例方法就可以直接在其上绘图'''
# _ = ax1.hist(np.random.randn(100),bins = 20,color = 'k',alpha = 0.3)
# __ = ax2.scatter(np.arange(30),np.arange(30) + 3 * np.random.randn(30))
# plt.show()

'''matplotlib有一个更为方便的方法plt.subplots，它可以创建一个新的Figure,并返回一个含有已创建的subplot对象的Numpy数组'''
# fig,axes = subplots(2, 3)
'''fig 是对象，axes是一个数组其中包含了subplot对象'''



'''调整subplot周围的间距'''
# fig,axes = plt.subplots(2,2,sharex = True,sharey= True) #sharex 和 sharey 是上下两个 左右两个 是否共享 x轴和y轴
# for i in range(2):
# 	for j in range(2):
# 		axes[i,j].hist(np.random.randn(500),bins = 50,color = 'k',alpha = 0.5)
#
# plt.subplots_adjust(wspace= 0,hspace= 0)
# plt.show()


'''颜色、标记和线型'''
from numpy.random import randn
# plt.plot(randn(30).cumsum(),'ko--') #先颜色再标记类型在线型
# plt.plot(randn(30).cumsum(),color = 'k',linestyle = 'dashed',marker = 'o') #和上一句意思一样，更详细
# plt.show()

'''在线型图中，非实际数据点默认是按线性方式插值的。可以通过drawstyle选项修改'''

# data = np.random.randn(30).cumsum()
# plt.plot(data,'k--',label = 'Default')
# plt.plot(data,'k-',drawstyle = 'steps-post',label ='steps-post')
# plt.legend(loc = 'best')
#
# plt.show()

'''设置标题、轴标签、刻度以及刻度标签'''
'''matplotlib中对于大多数的图表装饰项，其主要实现方式有二：使用过程型的pyplot接口以及更为面向对象的原生matplotlib API
   pyplot接口的设计目的就是交互式使用，含有诸如xlim、xticks和sxticklables之类的方法，他们分别控制图表的范围、刻度位置、刻度标签等，其主要方式有以下两种，
   1、调用是不带参数，则返回当前的参数值（例如，plt.xlim()返回当前的x轴绘图范围）。
   2、调用时带参数，则设置参数值（例如，plt.xlim([0,10])会将x轴的范围设置为0到10）。
   所有这些方法都是对当前或最近创建的AxesSubplot起作用的。它们各自对应subplot对象上的两个方法。以
   xlim为例，就是ax.get_xlim 和ax.set_xlim。
   剑辉说：关于plt中对于大多数图表的装饰项，要不使用plt的给的接口，要不使用subplot实例对象的实例方法。殊途同归
   '''
#下面这个例子是采用第二个方法
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(np.random.randn(1000).cumsum())
# ticks = ax.set_xticks([0,250,500,750,1000])  #告诉matplotlib要将刻度放在数据范围中的哪些位置，默认情况下，这些位置也是刻度标签
# labels = ax.set_xticklabels(['one','two','three','four','five'],rotation = 30,fontsize = 'small')
# ax.set_title('My first matplotlib plot')
# ax.set_xlabel('Stages')
# plt.show()


'''添加图例

'''
# from numpy.random import randn
# fig = plt.figure();ax = fig.add_subplot(1,1,1)
# ax.plot(randn(1000).cumsum(),'k',label = 'one')
# ax.plot(randn(1000).cumsum(),'k--',label = 'two')
# ax.plot(randn(1000).cumsum(),'k.',label = 'three')
# ax.legend(loc = 'best') #在最合适的位置添加图例
# plt.show()

'''注释以及在Subplot上绘图'''

#text可以将文本绘制在图表的指定坐标(x,y),还可以加上一些自定义格式
#下面根据最新的标准普尔500指数价格绘制一张曲线图，并表示出2008年到2009年金融危机期间的一些重要日期
# from datetime import datetime
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# data = pd.read_csv(r'examples_data_for_data_analysis(1)\examples_data_for_data_analysis\spx.csv',index_col = 0,parse_dates= True)
# spx = data['SPX']
# spx.plot(ax = ax,style = 'k-')
# crisis_data = [
# 	(datetime(2007,10,11),'Peak of bull market'),
# 	(datetime(2008,3,12),'Bear Stearns Fails'),
# 	(datetime(2008,9,15),'Lehman Bankruptcy')
# ]
# for date,label in crisis_data:
# 	ax.annotate(label,xy = (date,spx.asof(date) + 75),
# 				xytext = (date,spx.asof(date) + 225),
# 				arrowprops = dict(facecolor = 'black',headwidth = 4,
# 								  width = 2,headlength = 4),
# 				horizontalalignment = 'left',verticalalignment = 'top')
# ax.set_title('Important dates in the 2008-2009 financial crisis')
# plt.show()


'''将图表保存到文件
   利用plt.savefig('figpath.svg')
   文件类型使用过文件扩展名推断出来的
'''



'''使用pandas和seaborn绘图'''

'''图-Series'''
# s = pd.Series(np.random.randn(10).cumsum(),index = np.arange(0,100,10))
# s.plot(kind = 'bar')
# plt.show()

'''图-pandas'''
# df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
# 				  columns= ['A','B','C',"D"],
# 				  index= np.arange(0,100,10))
# df.plot(kind = 'bar')
# plt.show()

'''对于DataFrame,柱状图会将每一行的值分成一组，并排显示'''
#
# df = pd.DataFrame(np.random.rand(6,4),index=['one','two','three','four','five','six'],columns = pd.Index(['A','B','C','D'],name = "Genus"))
# df.plot(kind = 'bar',legend = 'best',stacked = True,alpha = 0.5) #stacked堆积柱状图
# plt.show()


tips = pd.read_csv(r'examples_data_for_data_analysis(1)\examples_data_for_data_analysis\tips.csv')
# party_counts = pd.crosstab(tips['day'],tips['size']) #交叉表是用于统计分组频率的特殊透视表
# party_counts = party_counts.iloc[:,1:5]
# print(party_counts)
# party_pcts = party_counts.div(party_counts.sum(1),axis = 0) #div(反向除以参数） sum(1)以行来计算sum
# party_pcts.plot(kind = 'bar')
# plt.show()


'''seaborn -- barplot'''
import seaborn as sns
tips['tip_pct'] = tips['tip']/(tips['total_bill'] - tips['tip'])
# sns.barplot(x = 'tip_pct',y='day',data = tips,orient='h')
# plt.show()

# sns.barplot(x = 'tip_pct',y = 'day',hue = 'time',data = tips,orient = 'h')
# plt.show()

'''seaborn -- 直方图和密度图'''
#直方图 一种可以对值频率进行离散化显示的柱状图
# tips['tip_pct'].plot.hist(bins = 50)
# plt.show()


'''seabron 散布图或者点图'''
macro = pd.read_csv(r'examples_data_for_data_analysis(1)\examples_data_for_data_analysis\macrodata.csv')
data = macro[['cpi','m1','tbilrate','unemp']]
trans_data = np.log(data).diff().dropna()
# trans_data[-5:]
# sns.regplot('m1','unemp',data = trans_data)
# plt.title('Changes in log %s versus log %s'%('ml','unemp'))
# plt.show()


'''seabron 散布图矩阵:多个分类变量的关系'''
# sns.pairplot(trans_data,diag_kind= 'kde',plot_kws= {'alpha':0.2})
# plt.show()


'''seaborn 分面网格 和 类型数据
   针对有多个分类变量的数据可视化的一种方式是使用小面网格
'''
# sns.factorplot(x = 'day' ,y = 'tip_pct',hue = 'time',col = 'smoker',kind = 'bar',data = tips[tips.tip_pct < 1])
# plt.show()

