import numpy as np
import matplotlib.pyplot as plt
# -*- coding:utf-8 -*-
#1 用astype将其转化成数值形式
from numpy.matlib import randn
import matplotlib.pyplot as mp
import seaborn as sns
sns.set(color_codes=reset_index)
import matplotlib

# a = np.array(['21.2','213.4','43.5'],dtype=np.string_)
# a.astype(float) #转换类型
# print(a)
# print(a.dtype)
#
# #2 用astype将其转化成数值形式(另一种形式）
# a = np.arange(10)
# b = np.array([.11,.324,.343,.5456,.546],dtype=np.float64)
# print(a)
# print(a.dtype) #int32
# print(b.dtype)
# print(a.astype(b.dtype)) #不能直接改变元素的数据类型
# c = a.astype(b.dtype)
# print(c.dtype)
#
# #3 数组和标量之间的运算
#
# arr = np.array([[1.,2.,3.],[4.,5.,6.]])
# print(arr.dtype) #float64
# print(arr*arr) #[[ 1.  4.  9.]
#                #[16. 25. 36.]]
# print(arr-arr)
# print(1/arr)
# '''
# [[1.         0.5        0.33333333]
#  [0.25       0.2        0.16666667]]
#
# '''
# print(arr*0.5)
# '''
# [[0.5 1.  1.5]
#  [2.  2.5 3. ]]
#
# '''
# #4 基本的索引和切片
# arr = np.arange(10) #[0 1 2 3 4 5 6 7 8 9]
# print(arr)
# arr[5:8] = 12
# print(arr)          #[ 0  1  2  3  4 12 12 12  8  9]
# '''
# 当你将标量赋值给一个切片时，该值会自动传播到整个选区。
# 和列表最重要的区别是数组切片时原始数组的视图。
# '''
# brr = arr[5:8] #[12 12 12]
# brr[1] = 123456
# print(arr) #[     0      1      2      3      4     12 123456     12      8      9]
# brr[:] =1314
# print(arr) #[   0    1    2    3    4 1314 1314 1314    8    9]
# '''如果你想得到一个副本而不是视图，可以使用arr[5:8].copy()'''
#
# arr2d =np.array([[1,2,3],
#         [4,5,6],
#         [7,8,9]])
# print('叮叮当当')
# print(arr2d[:,:2]) #多维数组切片第一个是从y轴切片，第二个是从x轴切片，：表示的是选取整个轴
#
#
# #5 numpy.where 函数
# xarr = np.array([1,2,3,4,5])
# yarr = np.array([6.,7.,8.,9.,10.])
# zarr = np.array([False,True,False,True,False])
#
# # 一般的python写法
# a =[x if z else y for x,y,z in zip(xarr,yarr,zarr) ]
#
# #用where函数来写
# a= np.where(zarr,yarr,xarr)
# print(a)
#
# #其中where的条件可以是 标量值
#
# #计算数组分位数最简单的方法是对其进行排序，然后选定特定位置的数
# large_arr = randn(1000)
# # large_arr.sort() #修改数组本身
# #np.sort()返回的是已排序副本
# large_arr.sort()
# # print(large_arr[int(0.05*len(large_arr))])
# # print(large_arr)
#
#
# # print(np.sort(large_arr)[int((0.05*len(large_arr)))])
#
#
#
# #将数组以二进制的形式保存在磁盘中
#
# wenjian = np.arange(155555)
# # np.save(r'C:\Users\lenovo\Desktop\erjinzhi_file',wenjian) #保存名字为erjinzhi_file的 .npy文件
# # message = np.load(r'C:\Users\lenovo\Desktop\erjinzhi_file.npy')
# # print(message)
#
# '''简单随机漫步'''
# import random
# position = 0
# walk = [position]
# steps = 1000
# for i in range(steps):
#     step = random.randint(-1,1)
#     position += step
#     walk.append(position)
# #
# # mp.plot(walk,label = 'walk')
# # mp.title('Random walk with +1/-1 steps')
# # mp.xlabel('steps')
# # mp.legend()
# # mp.show()
#
# #
# nsteps = 1000
# draws = np.random.randint(0,2,size = nsteps)
# steps = np.where(draws>0,1,-1)
# print(steps)
# walk = steps.cumsum() #计算累计合上的轴，None是计算平数组上的轴
#
# print(walk)
# print(walk.min(),walk.max())
#
# '''pandas练习'''
import pandas as pd
from pandas import Series,DataFrame
#
# obj  = Series([1,3,45,6])
# print(obj)
#
# obj.values
# obj.index
#
# obj1= Series([1,23,4,5,65],index=['q','34','dd','rr','rr'])
# print(obj1)
# print('#'*13)
# print(obj1[obj1>23])
#
# 'rr' in obj1 #True 可以把Series看成一个定长的有序字典
#
# a = {'name':'zhoujianhui','age':23,'love':'god'}
# Series(a) #name    zhoujianhui #产生一个Series对象
# #         age              23
# #        love            god
# #        dtype: object
#
#
# '''对许多应用而言，Series最重要的一个功能是她在算术运算
# 中会自动对齐不用索引的数据'''
# #构建DataFrame
# data ={ 'name':['周剑辉','苏焕','刘爽','马长岁','李兆'],'age':[25,30,21,31,32]}
# frame = DataFrame(data)
# print(frame)

#
#
# #重新索引
# obj = Series([1,2,3,4,5],index=['q','w','e','r','t'])
# print(obj)
# '''
# q    1
# w    2
# e    3
# r    4
# t    5
# dtype: int64
# '''
#
# obj2 = obj.reindex(['q','r','v'])
# #调用该Series的index将会根据索引进行重排，如果某个索引值当前不存在，就引入缺失值
# print(obj2)
# '''
# q    1.0
# r    4.0
# v    NaN
# dtype: float64
# '''
#
# #做一些插值处理
# obj2 = obj.reindex(['q','r','v'],fill_value=0)#填充零
# #调用该Series的index将会根据索引进行重排，如果某个索引值当前不存在，就引入0
# print(obj2)
# '''
# q    1
# r    4
# v    0
# dtype: int64
# '''
#
# #向后填充
# obj3 = Series(['name','age','god','time','man'],index=[i for i in range(5)])
# print(obj3)
# '''
# 0    name
# 1     age
# 2     god
# 3    time
# 4     man
# dtype: object
# '''
# obj4 = obj3.reindex([1,3,4,6,2],method='pad')#前向填充
# print(obj4)
# '''
# 0    name
# 1     age
# 2     god
# 3    time
# 4     man
# dtype: object
# 1     age
# 3    time
# 4     man
# 6     man #pad 前向填充
# 2     god
# dtype: object
# '''
#
# frame= DataFrame(np.arange(9).reshape(3,3),index=['xiao','mi','feng'],columns = ['周','剑','辉'])
# print(frame)
# '''
#    周  剑  辉
# a  0  1  2
# b  3  4  5
# c  6  7  8
# '''
#
# frame1 = frame.reindex(['a','b','v','c'])
# print(frame1)
# '''
#      周    剑    辉
# a  0.0  1.0  2.0
# b  3.0  4.0  5.0
# c  6.0  7.0  8.0
# f  NaN  NaN  NaN
# '''
#
# new_cloumns = ['周','剑','小','辉']
#  frame2 = frame.reindex(columns = new_cloumns)
# print(frame2)
# '''
#    周  剑   小  辉
# a  0  1 NaN  2
# b  3  4 NaN  5
# c  6  7 NaN  8
# '''
#
#
# #插值只能按行应用
# print(frame)
# print(frame)
# frame3 = frame.reindex(index = ['xiao','what','mi','feng'],method='backfill') #此时向前或者向后补充的话只能沿着0轴进行补充
# print(frame3)
#
#
# #丢弃指定轴上的值
#
# obj = Series(np.arange(6.),index=['a','b','c','d','e','f'])
# print(obj.drop('c'))
# '''
# a    0.0
# b    1.0
# d    3.0
# e    4.0
# f    5.0
# dtype: float64
# '''
#
# print(obj.drop(['a','e']))
# '''
# b    1.0
# c    2.0
# d    3.0
# f    5.0
# dtype: float64
# '''
#
# data = DataFrame(np.arange(16).reshape(4,4),columns=['北京','上海','神国','新加坡'],index=['one','two','three','four'])
# print(data.drop(['神国'],1))#删除1轴上的某个字段需要指出axis = 1
# '''
#        北京  上海  新加坡
# one     0   1    3
# two     4   5    7
# three   8   9   11
# four   12  13   15
# '''
# print(data.drop(['神国','上海'],axis = 1))#删除1轴上的某个字段需要指出axis = 1
# '''
#        北京  新加坡
# one     0    3
# two     4    7
# three   8   11
# four   12   15
# '''
# print(data.drop(['one'])) #删除0轴上的不需要指出
# '''
#        北京  上海  神国  新加坡
# two     4   5   6    7
# three   8   9  10   11
# four   12  13  14   15
# '''
#
#
#
# #索引、选取和过滤
# #Series索引的工作方式类似于Numpy数组的索引，只不过Series的索引值不只是整数
# obj = Series(np.arange(4),index= ['a','b','c','d'])
# print(obj['a'])
# #0
# print(obj[0])
# #0
#
# print(obj[1:3])
# '''
# b    1
# c    2
# dtype: int32
# '''
# print(obj[['b','c']]) #Series也可以通过标签进行选择，也可以通过下标进行选择,多选择标签时，选择的是一个列表
# '''
# b    1
# c    2
# dtype: int32
# '''
# print(obj[obj<2])
# '''
# dtype: int32
# a    0
# b    1
# dtype: int32
# '''
#
# #利用标签切片和普通的切片不同，其末端是宝包含的
# print(obj['a':'c'])
# '''
# dtype: int32
# a    0
# b    1
# c    2
# dtype: int32
# '''
#
#
# data = DataFrame(np.arange(16).reshape(4,4),index=['china','japan','American','Gussa'],columns=['one','two','three','four'])
# print(data)
# '''
#           one  two  three  four
# china       0    1      2     3
# japan       4    5      6     7
# American    8    9     10    11
# Gussa      12   13     14    15
# '''
# #以下都是选取列的
# print(data['one'])
# '''
# china        0
# japan        4
# American     8
# Gussa       12
# Name: one, dtype: int32
# '''
# print(data[['one','three']])
# '''
#           one  three
# china       0      2
# japan       4      6
# American    8     10
# Gussa      12     14
# '''
# #以下都是选取行的
# #切片或者布尔Series对行进行索引
# print(data[:2])
# '''
#        one  two  three  four
# china    0    1      2     3
# japan    4    5      6     7
# '''
# print(data[data['one']>10]) #只能对columns的选项进行 > 或者 < 处理 不能对col进行 < 或者 >处理
# #显示结果
# '''
#        one  two  three  four
# Gussa   12   13     14    15
# '''
#
# #因为data['one']>10等于 下面的
# '''
# china       False
# japan       False
# American    False
# Gussa        True
# '''
# #    print(data['china']) #注意：不可以直接使用下标对列进行索引，除非该columns当中包含该值。左边的这个操作是错误的
#
#
# print('*'*10)
# #loc主要运用于标签选取，选取行或者的列的标签，iloc主要运用于下标的选取，选取行或者列
#
# #算术运算和数据对齐
# #pandas最重要的一个功能是 他可以对不同索引的对象进行算术运算
# s1 = Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
# s2 = Series([7.3,-2.5,3.4,23,1.5],index=['a','c','3','e','f'])
# print(s1+s2) #如果数据重叠的话就会相加，如果数据不重叠的话则会在其索引处引入NaN值
# '''
# 3     NaN
# a    14.6
# c    -5.0
# d     NaN
# e    24.5
# f     NaN
# dtype: float64
# '''
#
# data = DataFrame(np.arange(16).reshape(4,4),columns=['时间','柜机数','投件数','值守姓名'],index=['one','two','three','four'])
# data1 = DataFrame(np.arange(9).reshape(3,3),columns=['时间','城市名字','值守姓名'],index=['one','two','four'])
#
# print(data+data1)
# '''
#        值守姓名  城市名字  投件数    时间  柜机数
# four   23.0   NaN  NaN  18.0  NaN
# one     5.0   NaN  NaN   0.0  NaN
# three   NaN   NaN  NaN   NaN  NaN
# two    12.0   NaN  NaN   7.0  NaN
# '''
#
#
# #当两个Datarame进行相加的时候
# data = DataFrame(np.arange(16).reshape(4,4),columns=['时间','柜机数','投件数','值守姓名'],index=['one','two','three','four'])
# data1 = DataFrame(np.arange(9).reshape(3,3),columns=['时间','城市名字','值守姓名'],index=['one','two','four'])
#
# print(data.add(data1,fill_value=200)) #为啥不是所有的行列不重叠的地方填补成 200呢
# '''
#          值守姓名   城市名字    投件数     时间    柜机数
# four    23.0  207.0  214.0   18.0  213.0
# one      5.0  201.0  202.0    0.0  201.0
# three  211.0    NaN  210.0  208.0  209.0
# two     12.0  204.0  206.0    7.0  205.0
# '''
#
#
# #Dataframe和Series之间的运算
# frame = DataFrame(np.arange(12.).reshape(4,3),columns=list('qwe'),index=['name','sex','homework','job'])
# print(frame)
# '''          q     w     e
# name      0.0   1.0   2.0
# sex       3.0   4.0   5.0
# homework  6.0   7.0   8.0
# job       9.0  10.0  11.0
# '''
# print(frame.iloc[0])
# '''
# q    0.0
# w    1.0
# e    2.0
# Name: name, dtype: float64
# '''
# print((frame-frame.iloc[0])) #默认情况下，Dataframe和Series之间的算术运算会将Series的索引匹配到DataFrame的列，然后沿着行一直向下广播
# '''
#             q    w    e
# name      0.0  0.0  0.0
# sex       3.0  3.0  3.0
# homework  6.0  6.0  6.0
# job       9.0  9.0  9.0
# '''
# frame1 = Series(range(3),index=list('qre')) #如果某个索引值在Dataframe的列或Series的索引中找不到，则参与运算的两个对象就会被重新索引以形成并集
# print(frame-frame1)
# '''
# e    q   r   w
# name      0.0  0.0 NaN NaN
# sex       3.0  3.0 NaN NaN
# homework  6.0  6.0 NaN NaN
# job       9.0  9.0 NaN NaN
# '''
#
# #你若希望匹配行并在列上传播，则必须使用算术运算方法
# series = frame['q']
# print(series)
# print(frame)
# print(frame.sub(series,axis = 0)) #匹配行并在列上传播
# '''
# name      0.0  1.0  2.0
# sex       0.0  1.0  2.0
# homework  0.0  1.0  2.0
# job       0.0  1.0  2.0
#
# '''
#
# #函数应用和映射
# #one Numpy的ufuns（元素级数组方法）也可用于操作pandas对象
# frame = DataFrame(np.random.randn(4,3),columns=list('qwe'),index=['name','sex','homework','job']) #randn产生正态分布的函数
# print(frame)
# '''
#                  q         w         e
# name     -1.371019 -0.443054  2.160663
# sex      -0.055386 -0.453091  1.649388
# homework  0.937876  0.959010  0.247819
# job       0.075764  0.210178  1.065421
# '''
# print(np.abs(frame))
# '''
#                  q         w         e
# name      0.044447  0.208530  0.030495
# sex       0.306406  1.618247  0.349330
# homework  0.839247  0.043582  1.275101
# job       0.230716  1.873716  0.136758
# '''
# #将函数应用到各列或各行形成的一维数组上 ，Dataframe的apply的方法即可实现此功能
# f = lambda x:x.max()-x.min()
# print(frame.apply(f)) #默认式作用在axis = 0（列）上的，从每列开始
# '''
# q    0.991385
# w    2.229679
# e    0.791939
# dtype: float64
# '''
# print(frame.apply(f,axis=1))
# '''
# name        4.441860
# sex         1.551519
# homework    1.203424
# job         1.593704
# dtype: float64
# '''
#
# #除标量外，传递给apply的函数还可以返回多个值组成的Series
# def f(x):
#     return Series([x.min(),x.max()],index=['min','max'])
# print(frame.apply(f))
# '''
#             q         w         e
# min -1.213960 -1.598532 -0.381062
# max  0.954835  0.868105  0.833397
# '''
# #元素级的python函数也是可以使用的
# #例子
# def_ = lambda x:'%.2f'%x
# print(frame.applymap(def_))
# '''
#               q      w      e
# name      -0.48  -1.60  -0.38
# sex       -1.14  -0.88   0.83
# homework  -1.21   0.87   0.08
# job        0.95  -0.55  -0.03
# '''
#
# #排序和排名
# obj = Series(range(4),index=['a','b','c','d'])
#
# print(obj.sort_index()) #从小到大进行排序
# '''
# a    0
# b    1
# c    2
# d    3
# dtype: int64
# '''
# #对于Dataframe，则可以根据任意一个轴上的索引进行排序
# frame = DataFrame(np.arange(8).reshape(2,4),index=['one','two'],columns=['a','b','d','c'])
# print(frame)
# '''
#      a  b  d  c
# one  0  1  2  3
# two  4  5  6  7
# '''
# print(frame.sort_index(axis=1)) #默认是 axis=0 从小到大进行排序（升序）
# '''
#      a  b  c  d
# one  0  1  3  2
# two  4  5  7  6
# '''
#
# print(frame.sort_index(axis=1,ascending=False)) #选择排序的方法是降序
# '''
#      d  c  b  a
# one  2  3  1  0
# two  6  7  5  4
# '''
#
#
# #按照值对Series进行排序
# obj = Series([3,4,-6,1])
# print(obj)
# '''
# 0    3
# 1    4
# 2   -6
# 3    1
# dtype: int64
# '''
# print(obj.sort_values()) #默认按照值进行升值排序（可以用于Dataframe，选择列或者行进行排序，取行或者列的名字就行）
# '''
# 2   -6
# 3    1
# 0    3
# 1    4
# dtype: int64
# '''
#
# #在Dataframe中，你可能希望根据一个列的值进行排序，将一个列传给by就行
# frame = DataFrame({'b':[2,3,1,6],'a':[4,5,1,2]})
# print(frame)
# '''
#    b  a
# 0  2  4
# 1  3  5
# 2  1  1
# 3  6  2
# '''
# print(frame.sort_index(by='b')) #将Dataframe中的数按照某列的值进行排序,这种用法不赞成，不好
# '''
#    b  a
# 2  1  1
# 0  2  4
# 1  3  5
# 3  6  2
# '''
# print('*'*20)
# print(frame.sort_values(by='a')) #将Dataframe中的数按照某列的值进行排序
#
# #rank函数
# obj  = Series([7,-5,7,4,2,0,4])
# print(obj)
# '''
# 0    7
# 1   -5
# 2    7
# 3    4
# 4    2
# 5    0
# 6    4
# dtype: int64
# '''
# print(obj.rank()) #将其中的元素按照从小到大进行排序，取其排名的作为值，如果它的排名的和它前一名的排名一样的话，取它们排名的平均值作为值
# '''
# 0    6.5
# 1    1.0
# 2    6.5
# 3    4.5
# 4    3.0
# 5    2.0
# 6    4.5
# dtype: float64
# '''
#
# print(obj.rank(method='first')) #根据数据在原数据中的出现的顺序进行排名
# '''
# 0    6.0 #第一次出现的名次是第六名吗
# 1    1.0
# 2    7.0 #第二次出现的名次是第七名吗
# 3    4.0
# 4    3.0
# 5    2.0
# 6    5.0
# dtype: float64
# '''
# print(obj.rank(method='max')) #根据数据在组中出现的最高排名（默认从小到大）升序
# '''
# 0    7.0 #最高排名是第七名
# 1    1.0
# 2    7.0
# 3    5.0
# 4    3.0
# 5    2.0
# 6    5.0
# dtype: float64
# '''
# print(obj.rank(method='min')) #根据数据在组中出现的最低排名（默认从小到大）升序
# '''
# 0    6.0 #
# 1    1.0
# 2    6.0
# 3    4.0
# 4    3.0
# 5    2.0
# 6    4.0
# dtype: float64
# '''
# print(obj.rank(ascending=False,method='min')) #根据数据在组中出现的最低排名（默认从大到小）降序
#
# #Dataframe可以在行或者列上计算排名
# frame = DataFrame({'a':[2.3,4,5,1],'b':[3,4,1,5.6],'c':[4,5,1,8.7]})
# print(frame)
# '''
#      a    b    c
# 0  2.3  3.0  4.0
# 1  4.0  4.0  5.0
# 2  5.0  1.0  1.0
# 3  1.0  5.6  8.7
# '''
# print(frame.rank(axis=1,method='max')) #默认升序
# '''
#      a    b    c
# 0  1.0  2.0  3.0
# 1  2.0  2.0  3.0
# 2  3.0  2.0  2.0
# 3  1.0  2.0  3.0
# '''
#
#
# #带有重复值的轴索引
# obj= Series(range(5),index=['2018/11/1','2018/11/1','2018/11/2','2018/11/2','2018/11/3'])
# print(obj)
# '''
# 2018/11/1    0
# 2018/11/1    1
# 2018/11/2    2
# 2018/11/2    3
# 2018/11/3    4
# dtype: int64
# '''
# print(obj.index.is_unique) #False
# print(obj.loc['2018/11/1'])
# '''
# 2018/11/1    0
# 2018/11/1    1
# dtype: int64
# '''
#
# #对Dataframe进行索引时也是如此
# df = DataFrame(np.random.randn(4,3),index=['2018/11/1','2018/11/1','2018/11/2','2018/11/2'])
# ''' 自动生成columns 号
#                   0         1         2
# 2018/11/1 -0.566791  1.016382  0.064721
# 2018/11/1  1.206169  1.025067  2.300873
# 2018/11/2 -0.769456 -0.396170 -0.730590
# 2018/11/2  1.361159 -0.104695  0.277078
# '''
# print(df.loc['2018/11/1'])
# '''
#                   0         1         2
# 2018/11/1  0.247117 -1.351215  0.853388
# 2018/11/1 -1.452018  0.794226  1.331388
# '''
#
# df = DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])
# '''
#     one  two
# a  1.40  NaN
# b  7.10 -4.5
# c   NaN  NaN
# d  0.75 -1.3
# '''
# print(df.sum()) #对axis = 0轴上的数进行求和计算
# '''
# one    9.25
# two   -5.80
# dtype: float64
# '''
# print(df.sum(axis= 1))#对axis=1轴上的数进行求和计算
# '''
# a    1.40
# b    2.60
# c    0.00
# d   -0.55
# dtype: float64
# '''
# print(df.sum(axis= 1,skipna = False))#对axis=1轴上的数进行求和计算,NAN的值会自动进行排除，除非整个切片都是NA
# #通过skipna选项可以禁用该功能,r如果参与计算的那一项中值存在NaN的情况的话则这一行不参与计算
# '''skipna 排除缺失值
# dtype: float64
# a     NaN
# b    2.60
# c     NaN
# d   -0.55
# dtype: float64
# '''
# print(df.idxmax()) #返回最大值得索引
# '''
# one    b
# two    d
# dtype: object
# '''
# print(df.cumsum()) #累进形
# '''
#     one  two
# a  1.40  NaN
# b  8.50 -4.5
# c   NaN  NaN
# d  9.25 -5.8
# '''

# '''相关系数和协方差'''
# import pandas_datareader.data as web
# # print('web:',web)
# #web: <module 'pandas_datareader.data' from 'D:\\Anaconda_chengxuwenjian\\lib\\site-packages\\pandas_datareader\\data.py'>
# all_data = {}
# print('执行到这里吗')
# for ticker in ['AAPL','IBM','MSFT','GOOG']:
#     all_data[ticker] = web.get_data_yahoo(ticker,'1/1/2000','1/1/2010')
# for i in all_data.values():
#
#     print('*'*30)
#     print(type(i),i.columns)
# #Index(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], dtype='object')
#
#
# price  =  DataFrame({tic:data['Adj Close'] for tic,data in all_data.items()})
# volume =  DataFrame({tic:data['Volume'] for tic,data in all_data.items()})
#
# print(price.index)
# '''
# DatetimeIndex(['1999-12-31', '2000-01-03', '2000-01-04', '2000-01-05',
#                '2000-01-06', '2000-01-07', '2000-01-10', '2000-01-11',
#                '2000-01-12', '2000-01-13',
#                ...
#                '2009-12-17', '2009-12-18', '2009-12-21', '2009-12-22',
#                '2009-12-23', '2009-12-24', '2009-12-28', '2009-12-29',
#                '2009-12-30', '2009-12-31']
# '''
# print(price.columns)
# '''
# Index(['AAPL', 'IBM', 'MSFT', 'GOOG'], dtype='object')
# '''
#
# returns = price.pct_change() #pct_change()这个函数的作用是返回当前元素和先前元素之间的百分比变化
# print(returns.tail()) #返回对象的最后n行，默认值为5
# '''
#                 AAPL       IBM      MSFT      GOOG
# Date
# 2009-12-24  0.034339  0.004385  0.002587  0.011117
# 2009-12-28  0.012295  0.013326  0.005484  0.007098
# 2009-12-29 -0.011862 -0.003477  0.007058 -0.005571
# 2009-12-30  0.012147  0.005461 -0.013698  0.005376
# 2009-12-31 -0.004300 -0.012597 -0.015504 -0.004416
# '''
#
# print(returns.MSFT.corr(returns.IBM)) #计算两列的相关系数 a.corr(b)
# #0.49253706494724375
# print(returns.MSFT.cov(returns.IBM)) #计算两列的协方差 a.corr(b)
# #0.00021557771540279465
#
# #Dataframe的corr和cov的方法将以DataFrame的形式返回完整的相关系数或协方差矩阵
# print(returns.corr())
# '''
#           AAPL       IBM      MSFT      GOOG
# AAPL  1.000000  0.412392  0.422852  0.470676
# IBM   0.412392  1.000000  0.492537  0.390688
# MSFT  0.422852  0.492537  1.000000  0.438313
# GOOG  0.470676  0.390688  0.438313  1.000000
# '''
#
# print(returns.corrwith(returns.IBM)) #计算returns每列和参数的相关系数,默认axis=0是按照
# '''
# AAPL    0.412392
# IBM     1.000000
# MSFT    0.492537
# GOOG    0.390688
# dtype: float64
# '''
# print(returns.corrwith(volume))
# '''
# AAPL   -0.057664
# IBM    -0.006592
# MSFT   -0.016101
# GOOG    0.062647
# dtype: float64
# '''
# #传入axis = 1则可按行进行计算，无论如何，在计算相关系数之前，所有的数据项都会按标签对齐



#唯一值、值计数以及成员资格
# obj = Series([1,2,3,4,4,5,5,2,1,3,45,5,1,2])
# print(obj.unique()) #求解唯一值
# '''
# [ 1  2  3  4  5 45]
# '''
# print(obj.value_counts()) #计算一个Series中各个值出现频率
#
# '''
# 5     3
# 2     3
# 1     3
# 4     2
# 3     2
# 45    1
# dtype: int64'''
# excl = pd.read_excel(r'E:\工作中的文件\设备大表数据\2017年11月设备大表1\11.xlsx')
# print(excl.iloc[:,2].value_counts())
# '''
# 成都市        3290
# 深圳市        2905
# 重庆市        2718
# 上海市        2638
#            ...
# 漳州市         227
# 蚌埠市         220
# 遂宁市         214
# 湖州市         200
# 九江市         184
# 太原市         147
# 银川市          86
# 澳门特别行政区      29
# 内江市          22
# 包头市           7
# 德州市           2
# 呼和浩特市         1
# Name: 运营城市, Length: 85, dtype: int64
# '''
# #value_count还是一个顶级的pandas方法，可用于任何数组或序列
# print(pd.value_counts(excl.iloc[:,2].values,sort=False))
# '''
# 江门市         233
# 昆明市        1334
# 泉州市         601
# ...
# 呼和浩特市         1
# 温州市         305
# 惠州市         697
# 柳州市         490
# 苏州市         736
# 潍坊市         664
# 青岛市        1434
# Length: 85, dtype: int64
#
# '''

#将pandas.value-counts传给apply函数
# print(excl.iloc[:,1:3].apply(pd.value_counts))
# '''
#        城市名称    运营城市
# 上海市    2638  2638.0
# 东莞市     735   735.0
# ...     ...     ...
# 银川市      86    86.0
# 马鞍山市     83     NaN
# 黄冈市      46     NaN
# 黄石市     188     NaN
#
# [99 rows x 2 columns]
# '''


# #处理缺失数据
# #pandas所有的描述性统计都排除了缺失数据
# #滤掉缺失数据
# from numpy import nan as NA
# data = Series([1,NA,3.5,NA,7])
# # drop 删除某行或者某列
# df = pd.DataFrame(np.arange(12).reshape(3, 4), columns=['A', 'B', 'C', 'D'])
#
# df
# # Drop columns,两种方法等价
#
# df.drop(['B', 'C'], axis=1)
# df.drop(columns=['B', 'C'])
# # 第一种方法下删除column一定要指定axis=1,否则会报错
# df.drop(['B', 'C'])
#
# '''ValueError: labels['B' 'C']
# not contained in axis'''
#
# # Drop rows
# df.drop([0, 1])
# df.drop(index=[0, 1])

# print(data.dropna()) #dropna返回一个仅含非空数据和索引值的Series,默认丢失任何含有缺失值的行
# '''
# 0    1.0
# 2    3.5
# 4    7.0
# dtype: float64
# '''
# print(data[data.notnull()])
# '''
# 0    1.0
# 2    3.5
# 4    7.0
# dtype: float64
#
# '''
# data1 = DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]]) #丢失任何含有缺失值的行
# print(data1.dropna())
# '''
#      0    1    2
# 0  1.0  6.5  3.0
# '''
# print(data1.dropna(how='all')) #丢失那些全为缺失值的行
# '''
#      0    1    2
# 0  1.0  6.5  3.0
# 1  1.0  NaN  NaN
# 3  NaN  6.5  3.0
# '''
# print(data1.dropna(how='all',axis = 1))#默认axis = 0
# '''
# 0  1.0  6.5  3.0
# 1  1.0  NaN  NaN
# 2  NaN  NaN  NaN
# 3  NaN  6.5  3.0
# '''
#
# #另一个滤除Dataframe行的问题涉及时间序列数据，假设你只想留下一部分观测数据，可以用thresh参数实现此目的
# df = DataFrame(np.random.randn(7,3))
# '''
#           0         1         2
# 0 -0.712890 -0.482502  1.687630
# 1 -1.511119  0.985896  1.712066
# 2 -1.019478  2.149830 -2.586221
# 3 -2.734125 -1.118476 -0.412005
# 4 -1.669053 -1.711741 -0.482341
# 5 -1.522972  0.093594  1.043770
# 6  1.891391 -2.137012  1.904018
# '''
# df.iloc[:1,0] = NA;df.iloc[:4,1] = NA;df.iloc[:2,2] = NA; #iloc的选取行或列的时候不包含右边的边界
# print(df)
# '''
# 0       NaN       NaN       NaN
# 1 -0.177411       NaN       NaN
# 2  1.196250       NaN -0.855532
# 3 -1.113623       NaN -1.475016
# 4  0.767188  0.782644 -1.712272
# 5 -0.284677  0.411226  1.128473
# 6 -1.605518 -0.157424 -0.083798
# '''
# print(df.dropna(thresh=1))  #thres的值是 至少这一行中有几个不是空值的才能被留下
# '''
#           0         1         2
# 1 -0.177411       NaN       NaN
# 2  1.196250       NaN -0.855532
# 3 -1.113623       NaN -1.475016
# 4  0.767188  0.782644 -1.712272
# 5 -0.284677  0.411226  1.128473
# 6 -1.605518 -0.157424 -0.083798
# '''
#
#
# #填充缺失值
# print(df.fillna('what')) #把缺失值填充为 'what'
# #fillna返回的是修改的新对象，并不修改原有的对象
# '''
#            0         1         2
# 0       what      what      what
# 1   -2.03418      what      what
# 2  -0.531639      what  -2.87678
# 3 -0.0742371      what  0.137053
# 4    2.20667 -0.343029   0.46567
# 5  -0.257569 -0.721593 -0.150986
# 6   0.986305 -0.207165 -0.780194
# '''
#
# #fillna如果要修改原对象，inplace = True,d但是不再返回新的修改对象
# print(df.fillna('修改了！',inplace=True)) #None
# print(df)
# '''
#            0         1         2
# 0       修改了！      修改了！      修改了！
# 1    0.60925      修改了！      修改了！
# 2  -0.608697      修改了！   1.00009
# 3    -1.0137      修改了！ -0.685157
# 4   0.141868  0.895669    -1.404
# 5  0.0533796  0.474335 -0.568944
# 6  -0.451627   1.26642   1.10978
# '''
# df = DataFrame(np.random.randn(6,3))
# df.iloc[2:,1]=NA;df.iloc[4:,2] = NA
# print(df)
# '''
#         0         1         2
# 0  1.917973  1.221890  0.619497
# 1  0.225257 -0.876099  0.275541
# 2  0.121836       NaN  1.536356
# 3 -0.140229       NaN  0.365261
# 4 -0.017938       NaN       NaN
# 5 -2.780607       NaN       NaN
# '''
# print(df.fillna(method='ffill'))
# '''
#         0         1         2
# 0  1.917973  1.221890  0.619497
# 1  0.225257 -0.876099  0.275541
# 2  0.121836 -0.876099  1.536356
# 3 -0.140229 -0.876099  0.365261
# 4 -0.017938 -0.876099  0.365261
# 5 -2.780607 -0.876099  0.365261
# '''
# print(df.fillna(method='ffill',limit=2))
# '''
#         0         1         2
# 0  1.917973  1.221890  0.619497
# 1  0.225257 -0.876099  0.275541
# 2  0.121836 -0.876099   1.536356
# 3 -0.140229 -0.876099   0.365261
# 4 -0.017938       NaN   0.365261
# 5 -2.780607       NaN   0.365261
# '''
#
#
#
# # 层次化索引
# data= Series(np.random.randn(10),index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])
# print(data) #带有multiindex索引的Series的格式化输出形式。索引之间的间隔表明直接使用上面的标签
# '''
# a  1    0.414566
#    2   -0.602992
#    3   -0.234485
# b  1    2.290100
#    2   -0.956544
#    3   -2.337945
# c  1   -1.308799
#    2   -0.864416
# d  2   -0.148119
#    3    0.618244
# dtype: float64
# '''
# print(data.index)
'''
MultiIndex(levels=[['a', 'b', 'c', 'd'], [1, 2, 3]],
           labels=[[0, 0, 0, 1, 1, 1, 2, 2, 3, 3], [0, 1, 2, 0, 1, 2, 0, 1, 1, 2]])
'''
# print(data['b'])
# '''
# 1   -0.212707
# 2    0.551499
# 3   -2.296885
# dtype: float64
# '''
# print(data['b':'c'])
# '''
# b  1   -0.644534
#    2   -1.850365
#    3   -1.980620
# c  1   -1.232501
#    2    1.102820
# dtype: float64
# '''
# print(data.loc[['a','b']])
# '''
# a  1   -0.322934
#    2    0.261313
#    3    0.128524
# b  1    0.680270
#    2    0.577322
#    3   -0.449675
# dtype: float64
# '''
#
# #内层中选取
# print(data[:,2])
# '''
# a   -1.203358
# b    0.709208
# c   -1.049297
# d    1.844196
# dtype: float64
# '''
#
# #层次化索引在在数据重塑和基于分组的操作（透视表生成）中起重要作用
# print(data.unstack())
# '''
#           1         2         3
# a -0.120211  0.259307  1.851326
# b -1.671440 -0.943115  0.737874
# c  1.067520  0.628492       NaN
# d       NaN  0.434530 -0.829924
# '''
# print(data.unstack().stack()) #unstack的逆运算
# '''
# a  1   -0.279937
#    2   -0.360961
#    3   -1.214816
# b  1    1.154832
#    2   -0.610438
#    3   -0.040759
# c  1   -0.874592
#    2    2.463101
# d  2    1.522671
#    3   -1.299753
# dtype: float64
#
# '''
# #对于一个Dataframe，每条轴都可以有分层索引
# frame = DataFrame(np.arange(16).reshape(4,4),index=[['a','a','b','b'],[1,3,1,3]],columns=[['河南省','河南省','山西省','山西省'],['郑州','焦作','太原','大同']])
# print(frame)
# '''
#     河南省     山西省
#      郑州  焦作  太原  大同
# a 1   0   1   2   3
#   3   4   5   6   7
# b 1   8   9  10  11
#   3  12  13  14  15
# '''
#
# #各层都可以有名称->索引名称 索引名称和轴标签不一样
# frame.index.names = ['key1','key2']
# frame.columns.names = ['省名','城市名']
# print(frame)
# '''
# 省名        河南省     山西省
# 城市名        郑州  焦作  太原  大同
# key1 key2
# a    1      0   1   2   3
#      3      4   5   6   7
# b    1      8   9  10  11
#      3     12  13  14  15
# '''
# print(frame['河南省'])
# '''
# 城市名        郑州  焦作
# key1 key2
# a    1      0   1
#      3      4   5
# b    1      8   9
#      3     12  13
# '''
# print(frame)
# '''
# 省名        河南省     山西省
# 城市名        郑州  焦作  太原  大同
# key1 key2
# a    1      0   1   2   3
#      3      4   5   6   7
# b    1      8   9  10  11
#      3     12  13  14  15
# '''
# print(frame.swaplevel('省名','城市名',axis=1)) #swaplevel中的参数既可以是字符串也可以是索引（int）
# #返回互换了级别i和j的新对象，不改变原数据
# '''
# 城市名        郑州  焦作  太原  大同
# 省名        河南省 河南省 山西省 山西省
# key1 key2
# a    1      0   1   2   3
#      3      4   5   6   7
# b    1      8   9  10  11
#      3     12  13  14  15
# '''
# print(frame.swaplevel(0,1).sort_index(level=0)) #将级别索引为0和1的两个级别进行互换，并且按照从小到大排列索引未0的级别
# '''
# 省名        河南省     山西省
# 城市名        郑州  焦作  太原  大同
# key2 key1
# 1    a      0   1   2   3
#      b      8   9  10  11
# 3    a      4   5   6   7
#      b     12  13  14  15
# '''
#
#
# #根据级别汇总数据
# print(frame.sum(level = 'key2')) #根据级别 'key2'来求和
# '''
# 省名   河南省     山西省
# 城市名   郑州  焦作  太原  大同
# key2
# 1      8  10  12  14
# 3     16  18  20  22
# '''
# print(frame.sum(level = '省名',axis = 1))
# '''
# 省名         河南省  山西省
# key1 key2
# a    1       1    5
#      3       9   13
# b    1      17   21
#      3      25   29
# '''
#
# frame = DataFrame({'a':range(7),'b':range(7,0,-1),'c':['one','one','one','two','two','two','two'],'d':[0,1,2,1,3,4,5]})
# print(frame)
# '''
#    a  b    c  d
# 0  0  7  one  0
# 1  1  6  one  1
# 2  2  5  one  2
# 3  3  4  two  1
# 4  4  3  two  3
# 5  5  2  two  4
# 6  6  1  two  5
# '''
# print(frame.set_index(['c','a']))#将a和c两列设立为行索引
# '''
# c   a
# one 0  7  0
#     1  6  1
#     2  5  2
# two 3  4  1
#     4  3  3
#     5  2  4
#     6  1  5
# '''
# print(frame.set_index(['c','a'],drop=False))#将a和c两列设立为行索引,并且保留这两列
# '''
# c   a
# one 0  0  7  one  0
#     1  1  6  one  1
#     2  2  5  one  2
# two 3  3  4  two  1
#     4  4  3  two  3
#     5  5  2  two  4
#     6  6  1  two  5
# '''
# frame2 = frame.set_index(['c','a'])
# print(frame2.()) #层次化索引会重新转移到列里面
# '''reindex
#      c  a  b  d
# 0  one  0  7  0
# 1  one  1  6  1
# 2  one  2  5  2
# 3  two  3  4  1
# 4  two  4  3  3
# 5  two  5  2  4
# 6  two  6  1  5
# '''
#
#
#
# #数据加载、存储与文件形式
# #read_csv
#
# # df = pd.read_csv(r'C:\Users\lenovo\Desktop\test.csv',index_col=1) #将日期作为索引
#
# # print(df)
#
#
#
# #逐块读取文件
# #只想读取几行
# df = pd.read_csv(r'C:\Users\lenovo\Desktop\test.csv',index_col=1,nrows=6) #将日期作为索引,只读取6行
# print(df)
#
# #逐块读取文件，只需要设置chunksize（行数）
# #用于文件很大时
# chunker = pd.read_csv(r'C:\Users\lenovo\Desktop\test.csv',chunksize = 1000)
# print(chunker)
# '''
# <pandas.io.parsers.TextFileReader object at 0x000000000551DCC0>
# '''
# #Dataframe.add函数 对两个Dataframe相加，计算之前使用此值填充现有缺失（NaN）值以及成功进行DataFrame对齐所需的任何新元素，相应位置只能有一个NaN，如果两个位置均为NaN，则无法
# city = Series([])
# for i in chunker:
#     city = city.add(i['运营城市名'].value_counts(),fill_value = 0)
# city = city.sort_values(ascending=False) #Python3.6之后的版本已经没有order属性了
# print(city[:10])
# '''
# 成都市    4632.0
# 上海市    3321.0
# 深圳市    3092.0
# 重庆市    2944.0
# 北京市    2401.0
# 武汉市    2217.0
# 杭州市    2021.0
# 广州市    1802.0
# 长沙市    1486.0
# 西安市    1462.0
# dtype: float64
# '''
#
#
# #缺失值在输出结果中会被表示为空字符串， 你可以将其表示为其他字符串
# chunker = city.to_csv(r'C:\Users\lenovo\Desktop\city.csv',na_rep='缺失值') #na_rep：将缺失值表示为自己想表示的对象
#
#
#
# #手工处理分隔符格式
# import csv
#
# with open(r'C:\Users\lenovo\Desktop\test_1.csv',encoding='UTF-8') as txt:
#     a = csv.reader(txt) #<_csv.re2ader object at 0x000000000675FAD8>   默认的dialect方法为excl，这个可以自己定义
#     object1 = list(a)[1:3]
#
#     a.writerows(object1) #保存



# # 合并，连接和连接
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                         'B': ['B0', 'B1', 'B2', 'B3'],
#                         'C': ['C0', 'C1', 'C2', 'C3'],
#                         'D': ['D0', 'D1', 'D2', 'D3']},index=[0, 1, 2, 3])
# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                         'B': ['B4', 'B5', 'B6', 'B7'],
#                         'C': ['C4', 'C5', 'C6', 'C7'],
#                         'D': ['D4', 'D5', 'D6', 'D7']},
#                          index=[4, 5, 6, 7])
# df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
#                         'B': ['B8', 'B9', 'B10', 'B11'],
#                         'C': ['C8', 'C9', 'C10', 'C11'],
#                         'D': ['D8', 'D9', 'D10', 'D11']},
#                         index=[8, 9, 10, 11])
# print(pd.concat([df1,df2,df3],keys=['one','two','three']))
# '''
#             A    B    C    D
# one   0    A0   B0   C0   D0
#       1    A1   B1   C1   D1
#       2    A2   B2   C2   D2
#       3    A3   B3   C3   D3
# two   4    A4   B4   C4   D4
#       5    A5   B5   C5   D5
#       6    A6   B6   C6   D6
#       7    A7   B7   C7   D7
# three 8    A8   B8   C8   D8
#       9    A9   B9   C9   D9
#       10  A10  B10  C10  D10
#       11  A11  B11  C11  D11
# '''

# '''
# 数据规整化：清理、转换、合并、重塑
# '''
#
# df1 = DataFrame({'key':['b','b','a','c','a','a','b'],'data1':range(7)})
# df2 = DataFrame({'key':['a','b','d'],'data2':range(3)})
# print(df1)
# '''
#   key  data1
# 0   b      0
# 1   b      1
# 2   a      2
# 3   c      3
# 4   a      4
# 5   a      5
# 6   b      6
# '''
# print(df2)
# '''
# key  data2
# 0   a      0
# 1   b      1
# 2   d      2
# '''
#
# #使用merge进行一对一合并
# print(pd.merge(df1,df2)) #并没有指明要用哪个列进行连接。如果没有指明，merge就会重叠列的列名作为键
# '''重叠列的 列名作为键（最好显式指明一下）
#   key  data1  data2
# 0   b      0      1
# 1   b      1      1
# 2   b      6      1
# 3   a      2      0
# 4   a      4      0
# 5   a      5      0
# '''
#
# #如果两个对象的列名不同，也可以分别进行指定
# df3 = DataFrame({'1key':['b','b','a','c','a','a','b'],'data1':range(7)})
# df4 = DataFrame({'rkey':['a','b','d'],'data2':range(3)})
# '''
# 只要是两个表的key值相同，都会产生笛卡尔积，不管是内、外、左和右连接
# 笛卡尔积：
# 因为我们是要的结果是从第一张表里面取出一条记录跟第二张表里面的一条记录综合成为一条新的记录；
# 假如表1有m条记录，表2有n条记录，那么对于第一张表而言有m种选择，而对于第二张表来说有n种选择。
# 结果就是m*n种选择
# 所以，在merge时，一定要避免相同的key值，可以分批次merge，最后再concat。
# '''
# #默认连接（inner）
# print(pd.merge(df3,df4,left_on='1key',right_on='rkey'))#默认情况下，merge做的是'inner'连接
# '''默认情况下，merge做的是'inner'连接
#   1key  data1 rkey  data2
# 0    b      0    b      1
# 1    b      1    b      1
# 2    b      6    b      1
# 3    a      2    a      0
# 4    a      4    a      0
# 5    a      5    a      0
# '''
#
# #外连接
# print(pd.merge(df3,df4,left_on='1key',right_on='rkey',how = 'outer'))#默认情况下，merge做的是'inner'连接
# '''
# 1key  data1 rkey  data2
# 0    b    0.0    b    1.0
# 1    b    1.0    b    1.0
# 2    b    6.0    b    1.0
# 3    a    2.0    a    0.0
# 4    a    4.0    a    0.0
# 5    a    5.0    a    0.0
# 6    c    3.0  NaN    NaN
# 7  NaN    NaN    d    2.0
# '''
#
#
# print(pd.merge(df1,df2,how = 'left')) #外连接，求的是并集
# '''
#   key  data1  data2
# 0   b    0.0    1.0
# 1   b    1.0    1.0
# 2   b    6.0    1.0
# 3   a    2.0    0.0
# 4   a    4.0    0.0
# 5   a    5.0    0.0
# 6   c    3.0    NaN
# 7   d    NaN    2.0
# '''
#
# #要根据多个键进行合并，传入一个由列名组成的列表即可
# left = DataFrame({'设备ID':['12','334','345'],'运营城市':['郑州','东京','北京'],'收入额':[1,2,3]})
# right = DataFrame({'设备ID':['12','345','334','666'],'运营城市':['郑州','北京','东京','上海'],'收入额':[i for i in range(4,8)]})
#
# print(pd.merge(left,right,on=['设备ID','运营城市'],how='outer'))
#
# '''
#    设备ID 运营城市  收入额_x  收入额_y
# 0   12   郑州    1.0      4
# 1  334   东京    2.0      6
# 2  345   北京    3.0      5
# 3  666   上海    NaN      7
# '''
# print(pd.merge(left,right,on='设备ID'))
# '''
# 设备ID 运营城市_x  收入额_x 运营城市_y  收入额_y
# 0   12     郑州      1     郑州      4
# 1  334     东京      2     东京      6
# 2  345     北京      3     北京      5
# '''
# print(pd.merge(left,right,on='设备ID',suffixes=('_表1','_表2'))) #为左右两个DataFrame对象的重叠列名上的字符串
# '''
#   设备ID 运营城市_表1  收入额_表1 运营城市_表2  收入额_表2
# 0   12      郑州       1      郑州       4
# 1  334      东京       2      东京       6
# 2  345      北京       3      北京       5
# '''
#
#
# #索引上的合并
# '''
# Dataframe中的连接键位于其索引中，在这种情况下，你可以传入left_index = True 或 right_index= True 来说明索引应该被用于连接键
# '''
# left1 = DataFrame({'key':['a','b','a','a','b','c'],'value':range(6)})
# '''
#   key  value
# 0   a      0
# 1   b      1
# 2   a      2
# 3   a      3
# 4   b      4
# 5   c      5
# '''
# right1 = DataFrame({'group_val':[3.5,7]},index=['a','b'])
# '''
# group_val
# a        3.5
# b        7.0
# '''
# print(pd.merge(left1,right1,left_on = 'key',right_index = True)) #允许右边使用index和左边进行匹配(默认连接方法 内连接）
# '''
#  key  value  group_val
# 0   a      0        3.5
# 2   a      2        3.5
# 3   a      3        3.5
# 1   b      1        7.0
# 4   b      4        7.0
# '''
#
#
# #轴向连接
# s1 = Series([0,1],index=['a','b'])
# s2 = Series([2,3,4],index=['c','d','e'])
# s3 = Series([5,6],index=['f','g'])
# print(pd.concat([s1,s2,s3])) #默认axis = 0进行连接
# '''
# a    0
# b    1
# c    2
# d    3
# e    4
# f    5
# g    6
# dtype: int64
# '''
# print(pd.concat([s1,s2,s3],axis = 1,sort=False))
# '''
#    0    1    2
# a  0.0  NaN  NaN
# b  1.0  NaN  NaN
# c  NaN  2.0  NaN
# d  NaN  3.0  NaN
# e  NaN  4.0  NaN
# f  NaN  NaN  5.0
# g  NaN  NaN  6.0
# '''
# s4 = pd.concat([s1*5,s3])
# print(s4)
# '''
# a    0
# b    5
# f    5
# g    6
# dtype: int64
# '''
# print(pd.concat([s1,s4],axis=1,sort=False)) #默认连接方式为outer
# '''
#  0  1
# a  0.0  0
# b  1.0  5
# f  NaN  5
# g  NaN  6
# '''
# print(pd.concat([s1,s4],axis=1,join='inner'))
# '''
#    0  1
# a  0  0
# b  1  5
# '''
#
# #在连接轴上创建一个层次化索引
# print(pd.concat([s1,s2,s3],keys=['one','two','three']))
# '''
# one    a    0
#        b    1
# two    c    2
#        d    3
#        e    4
# three  f    5
#        g    6
# dtype: int64
# '''
#
# #join函数 ,可以将 多个列进行连接
# import pandas as pd
#
# caller = pd.DataFrame({'key':['K0', 'K1', 'K2', 'K3', 'K4', 'K5'], 'A':['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
# print(caller)
# '''
#   key   A
# 0  K0  A0
# 1  K1  A1
# 2  K2  A2
# 3  K3  A3
# 4  K4  A4
# 5  K5  A5
# '''
# other = pd.DataFrame({'key':['K0', 'K1', 'K2','K99'], 'B':['B0', 'B1', 'B2', 'B99']})
# print(other)
# '''
#    key    B
# 0   K0   B0
# 1   K1   B1
# 2   K2   B2
# 3  K99  B99
# '''
# print(caller.join(other,lsuffix='左边的',rsuffix='右边的')) #默认使用索引进行连接
# '''
#   key左边的   A key右边的    B
# 0     K0  A0     K0   B0
# 1     K1  A1     K1   B1
# 2     K2  A2     K2   B2
# 3     K3  A3    K99  B99
# 4     K4  A4    NaN  NaN
# 5     K5  A5    NaN  NaN
# '''
#
# #通过指定列进行连接
# print(caller.set_index('key').join(other.set_index('key'))) #左右两个Dataframe分别将某列设成index
# '''
# key
# K0   A0   B0
# K1   A1   B1
# K2   A2   B2
# K3   A3  NaN
# K4   A4  NaN
# K5   A5  NaN
# '''
# #通过on参数指定连接的列，DataFrame.join总是使用other的索引去连接caller，因此我们可以把指定的列设置为other的索引，然后用on去指定caller的连接列，这样可以让连接结果的索引和caller一致
# print(caller.join(other.set_index('key'), on='key'))
# '''
#   key   A    B
# 0  K0  A0   B0
# 1  K1  A1   B1
# 2  K2  A2   B2
# 3  K3  A3  NaN
# 4  K4  A4  NaN
# 5  K5  A5  NaN
# '''

# 通过isnull 和 notnull来选择数据
# df = pd.DataFrame(np.random.randn(10,6),columns = ['one','two','three','four','five','six'])
# # Make a few areas have NaN values
# df.iloc[1:3,1] = np.nan
# df.iloc[5,3] = np.nan
# df.iloc[7:9,5] = np.nan
# '''
# dOut[21]:
#         one       two     three      four      five       six
# 0  1.633504 -0.862918  0.576555  0.058510  0.185410  1.407019
# 1  0.489295       NaN  1.225558 -0.965359  0.532709 -1.230544
# 2  0.484203       NaN  1.463121 -0.865273  0.292813 -1.036304
# 3 -1.854974  0.001866  0.932175 -0.612564 -1.241936  1.148582
# 4 -0.908277  1.617018 -1.299057  0.214503  0.343650 -1.647180
# 5 -1.418701  1.844799 -0.137709       NaN -0.727171  1.102250
# 6  0.864575  1.161839 -1.467584  0.193953  0.312534 -2.107013
# 7  1.125387 -0.476579  0.262594  1.750284  0.410969       NaN
# 8  0.147630  0.699069  0.218889 -0.204260 -1.195298       NaN
# 9  2.077085 -0.879059  0.647961 -3.003325  0.847344 -1.654472
# '''
#
# df.isnull()
# '''
# Out[23]:
#      one    two  three   four   five    six
# 0  False  False  False  False  False  False
# 1  False   True  False  False  False  False
# 2  False   True  False  False  False  False
# 3  False  False  False  False  False  False
# 4  False  False  False  False  False  False
# 5  False  False  False   True  False  False
# 6  False  False  False  False  False  False
# 7  False  False  False  False  False   True
# 8  False  False  False  False  False   True
# 9  False  False  False  False  False  False
# '''
#
# df.isnull().any()
# '''
# Out[24]:
# one      False
# two       True
# three    False
# four      True
# five     False
# six       True
# dtype: bool
# '''
# #df[df.two.isnull()]等同于df[df.two.isnull()==True]
# df[df.two.isnull()]
# '''
# Out[25]:
#         one  two     three      four      five       six
# 1  0.489295  NaN  1.225558 -0.965359  0.532709 -1.230544
# 2  0.484203  NaN  1.463121 -0.865273  0.292813 -1.036304
# '''
# df[df.two.isnull()==True]
# '''
# Out[26]:
#         one  two     three      four      five       six
# 1  0.489295  NaN  1.225558 -0.965359  0.532709 -1.230544
# 2  0.484203  NaN  1.463121 -0.865273  0.292813 -1.036304
# '''
# df[df.two.isnull()==False]
# '''
# Out[28]:
#         one       two     three      four      five       six
# 0  1.633504 -0.862918  0.576555  0.058510  0.185410  1.407019
# 3 -1.854974  0.001866  0.932175 -0.612564 -1.241936  1.148582
# 4 -0.908277  1.617018 -1.299057  0.214503  0.343650 -1.647180
# 5 -1.418701  1.844799 -0.137709       NaN -0.727171  1.102250
# 6  0.864575  1.161839 -1.467584  0.193953  0.312534 -2.107013
# 7  1.125387 -0.476579  0.262594  1.750284  0.410969       NaN
# 8  0.147630  0.699069  0.218889 -0.204260 -1.195298       NaN
# 9  2.077085 -0.879059  0.647961 -3.003325  0.847344 -1.654472
# '''
# df[df.two.notnull()]
# '''
# Out[29]:
#         one       two     three      four      five       six
# 0  1.633504 -0.862918  0.576555  0.058510  0.185410  1.407019
# 3 -1.854974  0.001866  0.932175 -0.612564 -1.241936  1.148582
# 4 -0.908277  1.617018 -1.299057  0.214503  0.343650 -1.647180
# 5 -1.418701  1.844799 -0.137709       NaN -0.727171  1.102250
# 6  0.864575  1.161839 -1.467584  0.193953  0.312534 -2.107013
# 7  1.125387 -0.476579  0.262594  1.750284  0.410969       NaN
# 8  0.147630  0.699069  0.218889 -0.204260 -1.195298       NaN
# 9  2.077085 -0.879059  0.647961 -3.003325  0.847344 -1.654472
#
# '''
#
#
#
# pop = pd.read_csv('state-population.csv')
# # print(pop.head())
# '''
#   state/region     ages  year  population
# 0           AL  under18  2012   1117489.0
# 1           AL    total  2012   4817528.0
# 2           AL  under18  2010   1130966.0
# ...
# '''
# areas = pd.read_csv(r'state-areas.csv')
# # print(areas.head())
# '''
#         state  area (sq. mi)
# 0     Alabama          52423
# 1      Alaska         656425
# 2     Arizona         114006
# ...
# '''
# abbrevs = pd.read_csv(r'state-abbrevs.csv')
# # print(abbrevs.head())
# '''
#         state abbreviation
# 0     Alabama           AL
# 1      Alaska           AK
# 2     Arizona           AZ
# ...
# '''
# merged = pd.merge(pop,abbrevs,left_on='state/region',right_on='abbreviation',how = 'outer') #用outer进行连接 不会丢失数据
# print(merged.head().iloc[:3,])
# '''
#   state/region     ages  year  population    state abbreviation
# 0           AL  under18  2012   1117489.0  Alabama           AL
# 1           AL    total  2012   4817528.0  Alabama           AL
# 2           AL  under18  2010   1130966.0  Alabama           AL
# '''
# merged = merged.drop('abbreviation',axis = 1) #axis = 1 删除某列 因为某列重复了
# print(merged.isnull().any()) #这一句可以看下哪一列有缺失值
# '''
# state/region    False
# ages            False
# year            False
# population       True
# state            True
# dtype: bool
# '''
# print(merged[merged['population'].isnull()==True].head()) #和后一句功能相同
# print('--------------------------------------------')
# print(merged[merged['population'].isnull()]) #和前一句功能相同
# '''
#    state/region     ages  year  population state
# 2448           PR  under18  1990         NaN   NaN
# 2449           PR    total  1990         NaN   NaN
# '''
# print(merged['state'].isnull().any()) #
# '''0    False
# 1    False
# 2    False
# 3    False
# 4    False'''
# print(merged.loc[merged['state'].isnull(),'state/region'].unique())#.loc[] is primarily label based, but may also be used with a boolean array
# #loc函数允许一个bool数组,只允许在index处的放置一个bool数组，不允许在columns处放置一个数组
# '''['PR' 'USA']'''
#
# merged.loc[merged['state/region']== "PR",'state'] = 'Puerto Rico' #给State/region为PR时的state填充上 Puerto Rico
# merged.loc[merged['state/region']== "USA",'state'] = 'United States'
# print(merged.isnull().any())
#
# final = pd.merge(merged,areas,on = 'state',how='left')
# print(final.head())
# print(final.isnull().any()) #看看新的Dataframe有哪些列有缺失值
# '''
# state/region     False
# ages             False
# year             False
# population        True
# state            False
# area (sq. mi)     True
# dtype: bool
# '''
# #看看面积有缺失值的地区是哪些
# print(final.loc[final['area (sq. mi)'].isnull(),'state'].unique())
# '''['United States']'''
#
# #去掉state为['United States']的area
# final.dropna()
#
#
# data2010 = final.query("year == 2000 & ages == 'total'") #适用于多个条件选择数据进行选择
# print(data2010.head())
#
# print(data2010.columns)
# '''Index(['state/region', 'ages', 'year', 'population', 'state', 'area (sq. mi)'], dtype='object')'''
#
# data2010.set_index('state',inplace = True) #将columns为state设置为列名
# density = data2010['population']/data2010['area (sq. mi)']


# #合并重叠数据
# #对索引部分重叠的两个数据集进行
# a = Series([np.nan,2.5,np.nan,3.5,4.5,np.nan],index=['f','e','d','c','b','a'])
# '''
# f    NaN
# e    2.5
# d    NaN
# c    3.5
# b    4.5
# a    NaN
# dtype: float64
# '''
# b = Series(np.arange(len(a)),dtype=np.float64,index=['f','e','d','c','b','a'])
# b[-1] = np.nan
# '''
# f    0.0
# e    1.0
# d    2.0
# c    3.0
# b    4.0
# a    NaN
# dtype: float64
# '''
# a.combine_first(b)
# '''
# f    0.0
# e    2.5
# d    2.0
# c    3.5
# b    4.5
# a    NaN
# dtype: float64
# '''
#
# df1 =DataFrame({'a':[1,np.nan,3,4,np.nan],'b':[3,4,np.nan,5,4]})
# print(df1)
# df2 = DataFrame({'a':[np.nan,2,4,np.nan],'b':[np.nan,4,np.nan,5]})
# print(df2)
# print(df1.combine_first(df2)) #index以左边为主，左边如果有的话就用左边的，左边如果没有的话就用右边的填充
# '''
#      a    b
# 0  1.0  3.0
# 1  2.0  4.0
# 2  3.0  NaN
# 3  4.0  5.0
# 4  NaN  4.0
# '''
#
#
# #重塑和轴向旋转
# '''stack:将数据的列"旋转"为行
#    unstack"：将数据的行"旋转"为列
# '''
# data  = DataFrame(np.arange(6).reshape(2,3),index=['周剑辉','黄燕玲'],columns=['年龄','性别','爱好'])
# '''
#      年龄  性别  爱好
# 周剑辉   0   1   2
# 黄燕玲   3   4   5
# '''
# result= data.stack()  #将数据的列（列名）转化成行（名）
# '''
# 周剑辉  年龄    0
#         性别    1
#         爱好    2
# 黄燕玲  年龄    3
#         性别    4
#         爱好    5
# dtype: int32
# '''
# print(result.unstack()) #将数据的行（行名）转化成列（列名)__默认情况下，unstack操作的是内层
# '''
#      年龄  性别  爱好
# 周剑辉   0   1   2
# 黄燕玲   3   4   5
# '''
# print(result.unstack([0]))#将数据的行（行名）转化成列（列名)__传入分层级别号（0），unstack操作的是外层'''
# '''
# 周剑辉  黄燕玲
# 年龄    0    3
# 性别    1    4
# 爱好    2    5
# '''
# df = DataFrame({'left':result,'right':result+100})
# df.columns.names = ['one']
# df.index.names = ['two','three']
# '''
#            one   left  right
# two three
# 周剑辉 年龄        0    100
#        性别        1    101
#        爱好        2    102
# 黄燕玲 年龄        3    103
#        性别        4    104
#        爱好        5    105
# '''
# print(df.unstack('two')) #将two旋转轴，旋转之后它作为columns的最低级别
# '''
#
# one   left     right
# two    周剑辉 黄燕玲   周剑辉  黄燕玲
# three
# 年龄       0   3   100  103
# 性别       1   4   101  104
# 爱好       2   5   102  105
# '''


# #如果不是所有级别值都能在各分组中找到的话，则unstack操作可能会引入缺失数值
# s1 = Series(['周剑辉','男','25','读书'],index=['name','sex','age','hobby'])
# s2 = Series(['34','男','工程师'],index=['age','sex','job'])
# s3 = pd.concat([s1,s2])
# '''
#     age hobby  job name sex
# one  25    读书  NaN  周剑辉   男
# two  34   NaN  工程师  NaN   男
# '''
# a = DataFrame({'name':['周剑辉','小波'],'age':[26,45]})
# pd.concat([a,DataFrame({'name':['小飞','小吴','小玲'],'age':[88,22,100]})],axis = 1)
# '''
#   name   age name  age
# 0  周剑辉  26.0   小飞   88
# 1   小波  45.0   小吴   22
# 2  NaN   NaN   小玲  100
# '''
#
# '''累计与分组'''
#
import seaborn as sns
# planets = sns.load_dataset('planets')
# print(planets.shape)
# print(planets.head())
# print(planets.isnull().any())
# print(planets.dropna().describe())
# df = DataFrame({'key':['A','B','C',"A",'B','C'],'data':range(6)})
# '''
# 0   A     0
# 1   B     1
# 2   C     2
# 3   A     3
# 4   B     4
# 5   C     5
# '''
#
#
# df.groupby('key') #一个DataFrameGroupBy对象
# '''<pandas.core.groupby.groupby.DataFrameGroupBy object at 0x0000000007B93160>
# '''
#
#
# #(1)按列取值
# planets.groupby('method')['orbital_period'].median()
# '''
# method
# Astrometry                         631.180000
# Eclipse Timing Variations         4343.500000
# Imaging                          27500.000000
# Microlensing                      3300.000000
# ...
# '''
#
# #(2)按组迭代 Groupby对象支持直接按组进行迭代，返回的每一组都是Series或DataFrame
# for (a,b) in planets.groupby('method'):
#     print("%s shape=%s"%(a,b.shape))
# '''Astrometry shape=(2, 6)
# Eclipse Timing Variations shape=(9, 6)
# Imaging shape=(38, 6)
# Microlensing shape=(23, 6)
# Orbital Brightness Modulation shape=(3, 6)
# Pulsar Timing shape=(5, 6)
# Pulsation Timing Variations shape=(1, 6)
# Radial Velocity shape=(553, 6)
# Transit shape=(397, 6)
# Transit Timing Variations shape=(4, 6)
# '''
#
# #(3)调用方法,可以让任何不由groupby对象直接实现的方法直接应用到每一组
# #ag
# print(planets.groupby('method')['year'].describe())
# '''
#                               count         mean   ...        75%     max
# method                                              ...
# Astrometry                       2.0  2011.500000   ...    2012.25  2013.0
# Eclipse Timing Variations        9.0  2010.000000   ...    2011.00  2012.0
# Imaging                         38.0  2009.131579   ...    2011.00  2013.0
# Microlensing                    23.0  2009.782609   ...    2012.00  2013.0
# Orbital Brightness Modulation    3.0  2011.666667   ...    2012.00  2013.0
# Pulsar Timing                    5.0  1998.400000   ...    2003.00  2011.0
# Pulsation Timing Variations      1.0  2007.000000   ...    2007.00  2007.0
# Radial Velocity                553.0  2007.518987   ...    2011.00  2014.0
# Transit                        397.0  2011.236776   ...    2013.00  2014.0
# Transit Timing Variations        4.0  2012.500000   ...    2013.25  2014.0
#
# [10 rows x 8 columns]
# '''
#
# #累计、过滤、转换和应用
# #累计
# #filter和apply传过来的都是一个分好组数据
# rng = np.random.RandomState(0)
# df = pd.DataFrame({'key':['A','B','C','A','B','C'],'data1':range(6),'data2':rng.randint(0,10,6)})
# '''
#   key  data1  data2
# 0   A      0      5
# 1   B      1      0
# 2   C      2      3
# 3   A      3      3
# 4   B      4      7
# 5   C      5      9
# '''
# df.groupby('key').aggregate(['min',np.median,max])  #方法1 指定不同列使用一些函数（这些函数可以使用字符串也可以不使用字符串）
# '''
# data1            data2
#       min median max   min median max
# key
# A       0    1.5   3     3    4.0   5
# '''
# df.groupby('key').aggregate({'data1':'min','data2':max}) #针对不同的列使用不同的函数
# '''                         这些函数也可以写成字符串形式也可以写成非字符串的形式
#      data1  data2
# key
# A        0      5
# B        1      7
# C        2      9
# '''
#
# #过滤
# def filter_func(x):
#     return x['data2'].std() >4
# df.groupby('key').std()
# '''
#        data1     data2
# key
# A    2.12132  1.414214
# B    2.12132  4.949747
# '''
# df.groupby('key').filter(filter_func) #filter函数会返回一个布尔值，表示每个组是否通过过滤
# '''
#   key  data1  data2
# 1   B      1      0
# 2   C      2      3
# 4   B      4      7
# 5   C      5      9
# '''
# #转换
# df.groupby('key').transform(lambda x:x - x.mean()) #每组的每一个值减去每组的平均值，实现数据的标准化
# '''
# key  data1  data2
# 0   A      0      5
# 1   B      1      0
# 2   C      2      3
# 3   A      3      3
# 4   B      4      7
# '''
#
# #apply函数
# def norm_by_data2(x): #输入进去的是一个分好组的数
#     #x是一个分组数据的Dataframe
#     print('one %s'%x)
#     x['data1'] /= x['data2'].sum()
#     print('two %s' % x)
#     return x
#
# df.groupby('key').apply(norm_by_data2)
# '''
# one:  key  data1  data2   ---输入进去的
# 0   A      0      5
# 3   A      3      3
# two:  key  data1  data2   ---输出的
# 0   A  0.000      5
# 3   A  0.375      3
# one:  key  data1  data2
# 1   B      1      0
# 4   B      4      7
# two:  key     data1  data2
# 1   B  0.142857      0
# 4   B  0.571429      7
# one:  key  data1  data2
# 2   C      2      3
# 5   C      5      9
# two:  key     data1  data2
# 2   C  0.166667      3
# 5   C  0.416667      9
#
# Out[101]:
#   key     data1  data2
# 0   A  0.000000      5
# 1   B  0.142857      0
# 2   C  0.166667      3
# 3   A  0.375000      3
# 4   B  0.571429      7
# 5   C  0.416667      9
#
# '''
#
# #设置分割的键
# #1 将列表、数组、Series或索引作为分组键
# L = [0,1,0,1,2,0]
# df.groupby(L).sum()
# '''
#
# '''
#
#
# #分组案例
# decade = 10*(planets['year']//10) #将年份转化为年代
# decade = decade.astype(str) + 's' #将年份转化为年代
# first = planets.groupby(['method',decade])['number'].sum() #形成一个method在前 decade(年代)在后的数据
# first.unstack().fillna(0) #将内层的decade(年代)转化成columns(l列名)
# '''
# year                           1980s  1990s  2000s  2010s
# method
# Astrometry                       0.0    0.0    0.0    2.0
# Eclipse Timing Variations        0.0    0.0    5.0   10.0
# Imaging                          0.0    0.0   29.0   21.0
# Microlensing                     0.0    0.0   12.0   15.0
# Orbital Brightness Modulation    0.0    0.0    0.0    5.0
# Pulsar Timing                    0.0    9.0    1.0    1.0
# Pulsation Timing Variations      0.0    0.0    1.0    0.0
# Radial Velocity                  1.0   52.0  475.0  424.0
# Transit                          0.0    0.0   64.0  712.0
# Transit Timing Variations        0.0    0.0    0.0    9.0
# '''
#
#
# #pandas reset_index 函数和set_index函数
# frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
#                       'c': ['one', 'one', 'one', 'two', 'two',
#                             'two', 'two'],
#                       'd': [0, 1, 2, 0, 1, 2, 3]})
# '''
#    a  b    c  d
# 0  0  7  one  0
# 1  1  6  one  1
# 2  2  5  one  2
# 3  3  4  two  0
# 4  4  3  two  1
# 5  5  2  two  2
# 6  6  1  two  3
# '''
# frame1 = frame.set_index(['c','d'])  #将列变成index
# '''
#        a  b
# c   d
# one 0  0  7
#     1  1  6
#     2  2  5
# two 0  3  4
#     1  4  3
#     2  5  2
#     3  6  1
# '''
# frame1 = frame1.reset_index() #将index变为列
# '''
# c  d  a  b
# 0  one  0  0  7
# 1  one  1  1  6
# 2  one  2  2  5
# 3  two  0  3  4
# 4  two  1  4  3
# 5  two  2  5  2
# 6  two  3  6  1
#
# '''
#
#
# # apply map 和 applymap三者之间的不同
'''这三个方法
   如果是作用于每一个元素的话 推荐使用applymap
   如果是作用于单列（也是一个Series）的话推荐使用 apply 或 map
   如果是作用于列于列之间的运算的话 推荐使用 apply
    '''
#
#
#
# #map 主要作用于Series
# #map() 是一个Series的函数，DataFrame结构中没有map()。map()将一个自定义函数应用于Series结构中的每个元素(elements)。
# df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
#                    'key2' : ['one', 'two', 'one', 'two', 'one'],
#                    'data1' : np.arange(5),
#                    'data2' : np.arange(5,10)})
# df['data1'] = df['data1'].map(lambda x : "%.3f"%x) #lambda在这里其实是在定义一个简单的函数，一个没有函数名的函数。
# '''
#   key1 key2  data1  data2
# 0    a  one  0.000      5
# 1    a  two  1.000      6
# 2    b  one  2.000      7
# 3    b  two  3.000      8
# 4    a  one  4.000      9
# '''
# #apply
# #apply()将一个函数作用于DataFrame中的每个行或者列(可以作用到列或者行上) map一般是作用到单列上
# df['total'] = df[['data1','data2']].apply(lambda x : x.sum(),axis=1 )
# df
# '''
# key1 key2  data1  data2  total
# 0    c  one  0.000      5    5.0
# 1    c  two  1.000      6    7.0
# 2    d  one  2.000      7    9.0
# 3    d  two  3.000      8   11.0
# 4    c  one  4.000      9   13.0
# '''
# # applymap（）
# #将函数做用于DataFrame中的所有元素(elements)
# def  addA(x):
#     return "A" + str(x )
# df.applymap(addA)
# '''
#   key1  key2   data1 data2  total
# 0   Ac  Aone  A0.000    A5   A5.0
# 1   Ac  Atwo  A1.000    A6   A7.0
# 2   Ad  Aone  A2.000    A7   A9.0
# 3   Ad  Atwo  A3.000    A8  A11.0
# 4   Ac  Aone  A4.000    A9  A13.0
# '''

# #isin 选择某列中含有某个值的 列 刻印取不是没有这个值得
# data = DataFrame([{'name':'周剑辉','age':25,'sex':'man','长相':'normal'},{'name':'张彩','age':23,'sex':'women','长相':'good'},{'name':'小菲菲','age':28,'sex':'women','长相':'good'},{'name':'周剑辉','age':28,'sex':'women','长相':'good'}])
# data[data['name'].isin(['周剑辉'])]
# '''
#    age name  sex      长相
# 0   25  周剑辉  man  normal
# '''
# data[~data['name'].isin(['周剑辉'])]
# '''
#    age name    sex    长相
# 1   23   张彩  women  good
# 2   28  小菲菲  women  good
# '''
# data[~(data['name'].isin(['周剑辉']) & data['sex'].isin(['women']))] #选择 name 不等于 ‘周剑辉’ 和 sex 不等于 ‘women’的(只能使用 & 不能使用 and)
# '''
#    age name    sex      长相
# 0   25  周剑辉    man  normal
# 1   23   张彩  women    good
# '''
# data[~(data['name'].isin(['周剑辉']) | data['sex'].isin(['men']))]
# '''
#    age name    sex    长相
# 1   23   张彩  women  good
# 2   28  小菲菲  women  good
# '''
# #数据透视表
# titanic = sns.load_dataset('titanic')
# titanic.head()
# '''
#    survived  pclass     sex   age  ...    deck  embark_town  alive  alone
# 0         0       3    male  22.0  ...     NaN  Southampton     no  False
# 1         1       1  female  38.0  ...       C    Cherbourg    yes  False
# 2         1       3  female  26.0  ...     NaN  Southampton    yes   True
# 3         1       1  female  35.0  ...       C  Southampton    yes  False
# 4         0       3    male  35.0  ...     NaN  Southampton     no   True
# '''
# titanic.groupby('sex')['survived'].mean()  #看下性别对幸存率的影响(mean()之后的幸存率的平均值）
# '''
# sex
# female    0.742038
# male      0.188908
# Name: survived, dtype: float64
# '''
# titanic.groupby(['class','sex'])['survived'].mean().unstack(0)  #看下船舱、性别对幸存率的影响(mean()之后的幸存率的平均值）
# '''
# class      First    Second     Third
# sex
# female  0.968085  0.921053  0.500000
# male    0.368852  0.157407  0.135447
# '''
#groupby
# #用DataFrame的pivot_table实现的效果等同于上一节的管道命令的代码
# titanic.pivot_table('survived',index='sex',columns='class')
# '''DataFrame.pivot_table（values = None，index = None，columns = None，aggfunc ='mean'，fill_value = None，margin = False，dropna = True，margins_name ='All' ）'''
# '''
# class      First    Second     Third
# sex
# female  0.968085  0.921053  0.500000
# male    0.368852  0.157407  0.135447
# '''
# #多级数据透视表
# #一个小插曲 离散化和面元划分
# ages = [20,22,25,27,21,23,37,31,61,45,41,32] #将这些数划分为几个等级
# bins = [18,25,35,60,100]
# cats = pd.cut(ages,bins) #系统默认左闭右开
# cats
# '''
# [(18, 25], (18, 25], (18, 25], (25, 35], (18, 25], ..., (25, 35], (60, 100], (35, 60], (35, 60], (25, 35]]
# Length: 12
# Categories (4, interval[int64]): [(18, 25] < (25, 35] < (35, 60] < (60, 100]]
# '''
# cats.value_counts()
# '''
# Out[26]:
# (18, 25]     5
# (25, 35]     3
# (35, 60]     3
# (60, 100]    1
# dtype: int64
# '''
# cats = pd.cut(ages,bins,right=False) #系统默认左闭右开，可以设置右边是闭端
# cats
# '''
# [[18, 25), [18, 25), [25, 35), [25, 35), [18, 25), ..., [25, 35), [60, 100), [35, 60), [35, 60), [25, 35)]
# Length: 12
# Categories (4, interval[int64]): [[18, 25) < [25, 35) < [35, 60) < [60, 100)]
# '''
# #可以设置组的名字
# cats = pd.cut(ages,bins,right=False,labels=['18到25岁','25到35岁','35到60岁','60到100岁']) #系统默认左闭右开，可以设置右边是闭端
# cats
# '''
# [18到25岁, 18到25岁, 25到35岁, 25到35岁, 18到25岁, ..., 25到35岁, 60到100岁, 35到60岁, 35到60岁, 25到35岁]
# Length: 12
# Categories (4, object): [18到25岁 < 25到35岁 < 35到60岁 < 60到100岁]
# '''
# #cut和qcut的区别
# #cut
# #如果向cut传入组的数量而不是组的确切的组边界，它会根据数据的最小值和最大值计算等长组
# a = [i for i in range(50)]+[i for i in range(50,75)]*2 #1-50有50个 50-75有50个
# pd.cut(a,2,precision =3) #将最大值（75）和最小值（0）取平均值计算组分界
# '''
# [(-0.074, 37.0], (-0.074, 37.0], (-0.074, 37.0], (-0.074, 37.0], (-0.074, 37.0], ..., (37.0, 74.0], (37.0, 74.0], (37.0, 74.0], (37.0, 74.0], (37.0, 74.0]]
# Length: 100
# Categories (2, interval[float64]): [(-0.074, 37.0] < (37.0, 74.0]]
# '''
# #qcut
# #根据样本分位数进行划分，根据数据的分布情况，可以得到大小基本一致分组
# pd.qcut(a,2,precision =3,duplicates='drop') #1-50有50个 50-75有50个 分两组 ，分界在50
# '''
# [(-0.001, 49.5], (-0.001, 49.5], (-0.001, 49.5], (-0.001, 49.5], (-0.001, 49.5], ..., (49.5, 74.0], (49.5, 74.0], (49.5, 74.0], (49.5, 74.0], (49.5, 74.0]]
# Length: 100
# Categories (2, interval[float64]): [(-0.001, 49.5] < (49.5, 74.0]]
# '''
# #多级数据透视表
# age = pd.cut(titanic['age'],[0,18,80]) #使年龄分组并做为index
# #把年龄加进去作为第三个维度
# titanic.pivot_table('survived',['sex',age],'class') #groupby 的分组标签既可以用字符串也可以是数组
# '''
# class               First    Second     Third
# sex    age
# female (0, 18]   0.909091  1.000000  0.511628
#        (18, 80]  0.972973  0.900000  0.423729
# male   (0, 18]   0.800000  0.600000  0.215686
#        (18, 80]  0.375000  0.071429  0.133663
# '''
#
# #用 pd.qcut将船票价格按照计数项等分为两份（qcut函数按照分位数进行划分）
# fare = pd.qcut(titanic['fare'],2) #按照价格分成两个 船票数量大致相同的部分
# titanic.pivot_table('survived',['sex',age],[fare,'class'])
# '''
# fare            (-0.001, 14.454]              ...    (14.454, 512.329]
# class                      First    Second    ...               Second     Third
# sex    age                                    ...
# female (0, 18]               NaN  1.000000    ...             1.000000  0.318182
#        (18, 80]              NaN  0.880000    ...             0.914286  0.391304
# male   (0, 18]               NaN  0.000000    ...             0.818182  0.178571
#        (18, 80]              0.0  0.098039    ...             0.030303  0.192308
# [4 rows x 6 columns]
# '''
#
# #显示不同性别 年龄段 船舱 的人数
# # one
# age = pd.cut(titanic['age'],[0,18,80])
# titanic.groupby(['sex',age,'class'])['who'].count().unstack()
# '''
# class            First  Second  Third
# sex    age
# female (0, 18]      11      14     43
#        (18, 80]     74      60     59
# male   (0, 18]       5      15     51
#        (18, 80]     96      84    202
# '''
#
# # two(效果等同上面）
# age = pd.cut(titanic['age'],[0,18,80])
# titanic.pivot_table('who',['sex',age],'class',aggfunc=np.count_nonzero)
#
# #其他的数据透视表选项
# titanic.pivot_table(index = 'sex',columns = 'class',aggfunc={'survived':sum,'fare':'mean'})
# #aggfunc指定映射关系时，透视表的数值就已经确定了 aggfunc后面的函数可以写成字符串形式的 如 ‘mean’ 也可以写成 mean()
# '''
#               fare                       survived
# class        First     Second      Third    First Second Third
# sex
# female  106.125798  21.970121  16.118810       91     70    72
# male     67.226127  19.741782  12.661633       45     17    47
# '''
#
# #当需要计算每一组的总数时，可以通过margins参数进行设置
# titanic.pivot_table('survived',index='sex',columns='class',margins=True)
# '''
# class      First    Second     Third       All
# sex
# female  0.968085  0.921053  0.500000  0.742038
# male    0.368852  0.157407  0.135447  0.188908
# All     0.629630  0.472826  0.242363  0.383838
# '''
#
# #案例 USDA食品数据库
# import json
# db = json.load(open(r'E:\学习书籍\python-for-data-analysis-master\python-for-data-analysis-master\datasets\usda_food'))
# db #是一个有着6636个字典的列表.列表中的元素是由字典组成,每个字典有五个键值对组成
# db[0].keys()
#
# '''
# dict_keys(['id', 'description', 'tags', 'manufacturer', 'group', 'portions', 'nutrients'])
# '''
# db[0]['nutrients'][0]
# '''
# {'value': 25.18,
#  'units': 'g',
#  'description': 'Protein',
#  'group': 'Composition'}
# '''
# nutrients = DataFrame(db[0]['nutrients'])
# nutrients[:7]
# '''
#                    description        group units    value
# 0                      Protein  Composition     g    25.18
# 1            Total lipid (fat)  Composition     g    29.20
# 2  Carbohydrate, by difference  Composition     g     3.06
# 3                          Ash        Other     g     3.28
# 4                       Energy       Energy  kcal   376.00
# 5                        Water  Composition     g    39.28
# 6                       Energy       Energy    kJ  1573.00
# '''
# info_keys = ['description','group','id','manufacturer']
# info = DataFrame(db,columns = info_keys)
# #将各实物的营养成分连接起来
# list_ = []
# for i in db:
#  aa = DataFrame(i['nutrients'])
#  aa['id'] = i['id'] #增加一列不用 行数 直接赋一个值就行
#  list_.append(aa)
#
# list_ = pd.concat(list_,ignore_index=True)
# list_.duplicated().sum() #计算重复项有多少项
# list_.drop_duplicates(inplace=True)
# list_.rename(columns = {'description':'food','group':'fgroup'},inplace =True)
# list_.head()
#
#
#
#
#
#
#
#
#案例 2012联邦选举委员会数据库
fec = pd.read_csv(r'E:\P00000001-ALL.csv',low_memory=False)
print(fec.head())
print(fec.describe())
print(fec.columns)





# #案例 美国人的生日
# '''
#    year  month  day gender  births
# 0  1969      1  1.0      F    4046
# 1  1969      1  1.0      M    4440
# 2  1969      1  2.0      F    4454
# 3  1969      1  2.0      M    4548
# 4  1969      1  3.0      F    4548
#
# '''
# births = pd.read_csv(r'D:\data-CDCbirths-master\data-CDCbirths-master\births.csv')
#
# year = (births['year']//10*10).astype(str) +'s' #将年数变成年代
# # 或者 births['decade'] = births['year']//10*10
# births.pivot_table('births',year,'gender',aggfunc='sum')
# '''
# gender         F         M
# year
# 1960s    1753634   1846572
# 1970s   16263075  17121550
# 1980s   18310351  19243452
# 1990s   19479454  20420553
# 2000s   18229309  19106428
# '''
# births.dropna(inplace = True)
# births['date'] = births['year'].astype(str)+'-'+births['month'].astype(str)+'-'+births['day'].astype(str)
#
# # births.pivot_table('births',year,'gender',aggfunc='sum').plot()
# births.pivot_table('births','month').plot()
#
# plt.ylabel('total births per day') #设置y轴标签名字
# plt.show()
#
#
# #向量化字符串操作
# #处理（清洗）现实数据时不可或缺的功能
# #Pandas字符串方法列表
# monte =pd.Series(['zhoujianhui','lizhao','machangsui','gaofeng','chenrui'])
# #1、与Python字符串方法相似的方法(这些方法的返回值不同)
# # ag:返回字符串Series（lower）
# monte.str.lower()
# '''
# 0    zhoujianhui
# 1         lizhao
# 2     machangsui
# 3        gaofeng
# 4        chenrui
# dtype: object
# '''
# # ag:返回字符串数值（len）
# monte.str.len()
# '''
# 0    11
# 1     6
# 2    10
# 3     7
# 4     7
# dtype: int64
# '''
# monte.str.split()
# '''
# 0    [zhou, jianhui]
# 1         [li, zhao]
# 2     [ma, changsui]
# 3        [gao, feng]
# 4        [chen, rui]
# dtype: object
# '''
# #向量化取值和切片操作
# monte.str.slice(0,3)
# '''
# 0    zho
# 1    li
# 2    ma
# 3    gao
# 4    che
# dtype: object
# '''
# #get()与slice()操作还可以在split（）操作之后使用
# monte.str.split().str.slice(-1)
# '''
# 0     [jianhui]
# 1        [zhao]
# 2    [changsui]
# 3        [feng]
# 4         [rui]
# dtype: object
# '''
#
# #计算指标变量
# #将分类变量转换成“指标变量”
# df = DataFrame({'key':['b','b','a','c','a','b'],'data1':range(6)})
# pd.get_dummies(df)
# '''
#    data1  key_a  key_b  key_c
# 0      0      0      1      0
# 1      1      0      1      0
# 2      2      1      0      0
# 3      3      0      0      1
# 4      4      1      0      0
# 5      5      0      1      0
# '''
# pd.get_dummies(df['key'])
# '''
#    a  b  c
# 0  0  1  0
# 1  0  1  0
# 2  1  0  0
# 3  0  0  1
# 4  1  0  0
# 5  0  1  0
# '''
# df1 = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
#  'C': ['d', 'e', 'f']})
# pd.get_dummies(df, prefix=['col1', 'col2','col3']) #如果 填写prefix的话，prefix中的元素个数必须和data中可以分开（不能是数字）的列数相同
# '''
#  col1_a  col1_b  col2_a  col2_b  col2_c  col3_d  col3_e  col3_f
# 0       1       0       0       1       0       1       0       0
# 1       0       1       1       0       0       0       1       0
# 2       1       0       0       0       1       0       0       1
# '''
# #处理时间序列
# #原生Python的日期与时间工具:datetime 与 dateutil
from datetime import datetime
datetime(year=2019,month=1,day=5)
# '''
datetime.datetime(2019, 1, 5, 0, 0)
# '''
# from dateutil import  parser
# date = parser.parse("4th of July,2015")
# '''
# datetime.datetime(2015, 7, 4, 0, 0)
# '''
# date.strftime('%A') #打印出星期几
# '''
# 'Saturday'
# '''
#
# date = np.array('2015-07-04',dtype=np.datetime64) #datetime64类型将日期编码为64位整数,这样可以让日期数组非常紧凑
# '''array('2015-07-04', dtype='datetime64[D]')'''
# date + np.arange(12)
# '''
# array(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
#        '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
#        '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'],
#       dtype='datetime64[D]')
# '''
#
#
#
# '''python：利用pandas进行绘图（总结）基础篇
#     https://blog.csdn.net/genome_denovo/article/details/78322628
#     https://www.2cto.com/kf/201803/730093.html
# '''
# #Pandas的日期与时间工具：理想与现实的最佳解决方案
# date = pd.to_datetime("4th of July,2015")
# '''
# Timestamp('2015-07-04 00:00:00') #Timestamp类型数据
# '''
# date + pd.to_timedelta(np.arange(12),'D')
# '''
# DatetimeIndex(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
#                '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
#                '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'],
#               dtype='datetime64[ns]', freq=None)
# '''
#
# #时间戳数据
# dates = pd.to_datetime([datetime(2019,1,8),'4th of July,2015','2015-Jul-6','07-07-2015','20150708'])
# ''' 是一个DatetimeIndex类型的数据 -----
# DatetimeIndex(['2019-01-08', '2015-07-04', '2015-07-06', '2015-07-07',
#                '2015-07-08'],
#               dtype='datetime64[ns]', freq=None)
# '''
#
# #DatetimeIndex类型数据通过to_period（）方法 和 一个频率代码转换成PeriodIndex(时间周期数据）类型
# dates.to_period('D')
# '''
# PeriodIndex(['2019-01-08', '2015-07-04', '2015-07-06', '2015-07-07',
#              '2015-07-08'],
#             dtype='period[D]', freq='D')
# '''
#
# #当用一个日期减去另一个日期时，返回的结果是TimedeltaIndex类型
# dates - dates[0]
# '''
# TimedeltaIndex(['0 days', '-1284 days', '-1282 days', '-1281 days',
#                 '-1280 days'],
#                dtype='timedelta64[ns]', freq=None)
# '''
#
#
# #为了更简便的创建有规律的时间序列，Pandas提供了一些方法
# #pd.date_range()可以处理时间戳
# pd.date_range('2019-01-01','2019-01-08') #默认时间频率是 天
# '''DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
#                '2019-01-05', '2019-01-06', '2019-01-07', '2019-01-08'],
#               dtype='datetime64[ns]', freq='D')
# '''
# pd.date_range('2019-01-01',periods=8) #默认时间频率是天  开始时间 + 周期数 periods
# '''
# DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
#                '2019-01-05', '2019-01-06', '2019-01-07', '2019-01-08'],
#               dtype='datetime64[ns]', freq='D')
# '''
# pd.date_range('2019-01-01',periods=8,freq='H') #默认时间频率是天  开始时间 + 周期数 periods    freq参数改变时间间隔（默认时间频率是天 ）
# '''DatetimeIndex(['2019-01-01 00:00:00', '2019-01-01 01:00:00',
#                '2019-01-01 02:00:00', '2019-01-01 03:00:00',
#                '2019-01-01 04:00:00', '2019-01-01 05:00:00',
#                '2019-01-01 06:00:00', '2019-01-01 07:00:00'],
#               dtype='datetime64[ns]', freq='H')
#             '''
#
# ##pd.period_range()可以处理周期
# pd.period_range('2015-07',periods=8,freq='H')
# '''
# PeriodIndex(['2019-01-01 00:00', '2019-01-01 01:00', '2019-01-01 02:00',
#              '2019-01-01 03:00', '2019-01-01 04:00', '2019-01-01 05:00',
#              '2019-01-01 06:00', '2019-01-01 07:00'],
#             dtype='period[H]', freq='H')
# '''
#
# # pd.timedelta_range() 可以处理时间间隔
# pd.timedelta_range(0,periods=10,freq='H')
# '''
# TimedeltaIndex(['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00',
#                 '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00'],
#                dtype='timedelta64[ns]', freq='H')
# '''


# #时间频率与偏移量
# from pandas_datareader import  data
# goog = data.DataReader('GOOG',start = '2004',end = '2016',data_source = 'yahoo')
# goog.head()
# '''
#                  Open       High    ...     Adj Close    Volume
# Date                                ...
# 2004-08-19  49.676899  51.693783    ...     49.845802  44994500
# 2004-08-20  50.178635  54.187561    ...     53.805050  23005800
# 2004-08-23  55.017166  56.373344    ...     54.346527  18393200
# 2004-08-24  55.260582  55.439419    ...     52.096165  15361800
# 2004-08-25  52.140873  53.651051    ...     52.657513   9257400
# [5 rows x 6 columns]
# '''
# print(goog.columns)
# goog = goog['Close']
# goog.plot()
# plt.show()
# goog.plot(alpha = 0.5,style = '-')
# #这个方法可以用来 显示某个固定的时间段内 值 得 计算方式
# goog.resample('BA').mean().plot(style = ':')  #The default is ‘left’ for all frequency offsets except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’, and ‘W’ which all have a default of ‘right’.
# '''
# goog.resample('BA').mean()  #反映的是这一年的均值
# 2004-12-31     75.410168    #2014年的均值
# 2005-12-30    137.982103
# 2006-12-29    204.264109
# 2007-12-31    267.634514
# 2008-12-31    230.924069
# 2009-12-31    218.423533
# 2010-12-31    266.080947
# 2011-12-30    282.648726
# 2012-12-31    319.331297
# 2013-12-31    439.264377
# 2014-12-31    558.270558
# 2015-12-31    601.550547
# 2016-12-30           NaN
# '''
# goog.asfreq('BA').plot(style = '-') #上一年最后一个工作日得到收盘价
# '''
# 2004-12-31     95.772095
# 2005-12-30    206.089584
# 2006-12-29    228.752182
# 2007-12-31    343.505829
# 2008-12-31    152.830978
# 2009-12-31    307.986847
# 2010-12-31    295.065887
# 2011-12-30    320.863098
# 2012-12-31    351.404449
# 2013-12-31    556.734009
# 2014-12-31    523.521423
# 2015-12-31    758.880005
# Freq: BA-DEC, Name: Close, dtype: float64
# '''
# plt.legend(['input','resample','asfreq'])
# plt.show()
#
#
# fig,ax = plt.subplot(2,sharex = True)
# data = goog.iloc[:10]



#shfit函数的运用,用在比较DataFrame之间行与行

a = pd.DataFrame([{'姓名':'周剑辉','age':25,'work':45},{'姓名':'马向阳','age':27,'work':50},{'姓名':'罗军','age':36,'work':85},{'姓名':'小吴','age':29,'work':89}])
#选择年龄比上一行小但是能力比上一行的人的姓名
a[(a['age'] < a['age'].shift(1)) & (a['work'] > a['work'].shift(1))]['姓名']
'''
3    小吴
Name: 姓名, dtype: object
'''












