'''日期和时间数据类型及工具'''
from datetime import datetime
import pandas as pd
# now = datetime.now()
# print(now)
# print(now.year,now.month,now.day)
#timedelat表示两个datetime对象之间的时间差
# delta = datetime(2011,1,7) - datetime(2011,1,17)
# print(delta)
# print(delta.days)
# print(delta.seconds)

#给datetime对象加上（或减去）一个或多个timedelta,这样会产生一个新对象
# from datetime import  timedelta
# start = datetime(2011,1,7)
# print(start)
# print(start + timedelta(12))
# print(start - 2 * timedelta(12))


#字符串和datetime的相互转换
#利用str或strftime方法，datetime对象 和 pandas的Timestamp对象可以被格式化为字符串
# stamp = datetime(2011,1,3)
# print(str(stamp))
# print(stamp.strftime('%Y-%m-%d'))
# value = '2011-01-03'
#date.time.strptime可以用这些格式化编码将字符串转换成日期
# print(datetime.strptime(value,'%Y-%m-%d'))

# datestrs = ['7/6/2011','8/6/2011']
# print([datetime.strptime(x,'%m/%d/%Y') for  x in datestrs])


#datetime.strptime是通过已知格式进行日期解析的最佳方式，但是每次编写格式定义是很麻烦的事情，可以用deteutil这个第三方包中的parser.parse方法
# from dateutil.parser import parse
# print(parse('2019-05-25'))
# print(parse('Jan 31,1997 10:45 PM'))
# print(parse('6/12/2011',dayfirst = True))


#to_datetime方法可以解析多种不同的日期表示格式
# datestrs = ['2011-07-06 12:00:00','2011-08-06 00:00:00']
# print(pd.to_datetime(datestrs))
#它还可以处理缺失值（None,空字符串等）
# print(pd.to_datetime(datestrs + [None]))



'''时间序列基础'''
from datetime import datetime
import numpy as np
dates = [datetime(2011,1,2),datetime(2011,1,5),
		 datetime(2011,1,7),datetime(2011,1,8),
		 datetime(2011,1,10),datetime(2011,1,12)]

ts = pd.Series(np.random.randn(6),index = dates)
# print(ts.index)
#跟其他Series一样，不同索引的时间序列之间的算术运算会自动按日期对其
# print(ts + ts[::2])


#索引、选取、子集构造
# stamp = ts.index[2]
# print(ts[stamp])
# print(ts['2011-01-10'])
#对于较长的时间序列，只需传入'年'或'年月'即可轻松选取数据的切片
# longer_ts = pd.Series(np.random.randn(1000),index = pd.date_range('1/1/2000',periods= 1000))
#字符串“2001”被解释成年，并根据它选取时间区间，指定月也同样奏效
# print(longer_ts['2001'])
# print(longer_ts['2001-05'])
# print(ts[datetime(2011,1,7)])
#因为大部分时间序列数据都是按照时间先后排序的，因此你也可以用不存在于该时间序列中的时间戳对其进行切片
# print(ts['1/6/2011':'1/11/2011'])
# print(longer_ts['2001':'2006'])
#此外还有一个等价的实例方法也可以截取两个日期之间TimeSeries
# print(ts.truncate(after= '1/9/2011')) #在某个索引值之前和之后截断一个Series或DataFrame。


#这些操作对DataFrame也有效，对DataFrame的行进行索引
# dates = pd.date_range('1/1/2000',periods=100,freq = 'W-WED')
# long_df = pd.DataFrame(np.random.randn(100,4),index=dates,columns = ['Colorado','Texas','New York','Ohio'])
# print(long_df['2000-05'])
#带有重复索引的时间序列
# dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000','1/3/2000'])
# dup_ts = pd.Series(np.arange(5),index= dates)
# print(dup_ts)
#通过检查索引的is_unique属性，我们就知道它是不是唯一的
# print(dup_ts.index.is_unique)
#对这个时间序列进行索引，要么产生标量值，要不产生切片，具体要看所选的时间点是否重复
# print(dup_ts['1/3/2000'])
# print(dup_ts['1/2/2000'])
#假设你想要对具有非唯一时间戳的数据进行聚合，一个办法是使用groupby,并传入level = 0
# grouped = dup_ts.groupby(level = 0)
# print(grouped.mean())



'''日期的范围、频率以及移动'''
# print(ts)
#将之前那个时间序列转换为一个固定频率（每日）的时间序列，只需调用resample即可
# print(ts.resample("D"))


#生成日期范围
# index = pd.date_range('2012-04-01','2012-06-01')
# print(index)
#如果只传入起始或结束日期，那就得传入一个表示一段时间的数字
# print(pd.date_range(start= '2012-04-01',periods= 20))
# print(pd.date_range(end= '2012-04-01',periods= 20))
#起始和结束日期定义了日期索引的严格边界：如果你想生成一个由每月最后一个工作日组成的日期索引，可以传入“BM”频率
# print(pd.date_range('2000-01-01','2010-01-01',freq = 'BM'))
#date_range默认会保留起始和结束时间戳的时间信息
# print(pd.date_range('2012-05-02 12:56:31',periods=5))
#虽然起始和结束日期带有时间信息，但你希望产生一组被规范化到午夜的时间戳
# print(pd.date_range('2012-05-02 12:56:31',periods= 5,normalize= True))


#频率和日期偏移量
#传入一个整数可定义偏移量的倍数
# print(pd.date_range('2000-01-01','2000-01-03',freq = '4h'))
#大部分偏移量对象都可通过加法进行连接
# from pandas.tseries.offsets import  Hour,Minute
# a = Hour(2) + Minute(30)
# print(a)
#也可传入频率字符串，这种字符串可以被高效地解析为的等效的表达式
# print(pd.date_range('2000-01-01',periods=10,freq='1h20min'))


#WOM日期
#WOM（week of Month)是一种非常实用频率类，它以WOM开头
# rng = pd.date_range('2012-01-01','2012-09-01',freq= 'WOM-3FRI') #每月的第三个星期五
# print(list(rng))


#移动（超前和滞后）数据
#移动（shifting)指的是沿着时间轴将数据前移或后移。Series 和 DataFrame都有一个shift方法用于执行单纯的前移或后移操作，保持索引不变
# ts = pd.Series(np.random.randn(4),index = pd.date_range('1/1/2000',periods=4,freq='M'))
# print(ts)
# print(ts.shift(2))
# print(ts.shift(-2))
#由于单纯移位操作不会修改索引，所以部分数据会被丢弃，因此，如果频率已知，则可以将其传给shift以便实现对时间戳进行位移而不是对数据进行简单位移。
# print(ts.shift(2,freq= 'M')) #将日期向前移动两个每月最后一个工作日
#使用其他频率
# print(ts.shift(3,freq = 'D'))


#通过偏移量对日期进行位移
from pandas.tseries.offsets import   Day,MonthEnd
# now = datetime(2011,11,29)
# print(now + 3* Day())
#如果加的是锚点偏移量（MonthEnd,不均匀数据)，第一次增量会将原日期向前滚动到符合频率规则的下一个日期
# print(MonthEnd(2))
# print(now + MonthEnd())
# print(now + MonthEnd(2))
#通过锚点偏移量的rollforward 和 rollback方法，可明确地将日期向前或向后“滚动”
# print(MonthEnd().rollforward(now))
# print(MonthEnd().rollback(now))



'''时区处理'''
#时区本地化和转换
# rng = pd.date_range('3/9/2012 9:30',periods=6,freq='D')
# ts = pd.Series(np.random.randn(len(rng)),index=rng)
# print(ts)
#索引的tz字段为None
# print(ts.index.tz)
#用时区集生成日期范围
# print(pd.date_range('3/9/2012 9:30',periods= 10,freq= 'D' ,tz='UTC'))
#本地化到UTC时区
# ts_utc = ts.tz_localize('UTC')
# print(ts_utc.index)
# print(ts_utc)
#一旦时间序列被本地化到某个特定时区，就可以用tz_convert将其转换到别的时区
# print(ts_utc.tz_convert('America/New_York'))
#时间序列本地化到America/New_York
# ts_eastern = ts.tz_localize('America/New_York')
#将其转化成UTC或柏林时间
# ts_eastern.tz_convent('UTC')
# ts_eastern.tz_convent('Europe/Berlin')


#操作时区意识型Timestamp对象
#跟时间序列和日期范围差不多，独立的Timestamp对象也能被从单纯型本地化为时区意识型，并从一个时区转换到另一个时区

# stamp = pd.Timestamp('2011-03-12 04:00')
# stamp_utcz = stamp.tz_localize('utc')
# print(stamp_utcz.tz_convert('America/New_York'))
#创建Timestamp时，还可以传入一个时区信息
# stamp_moscow = pd.Timestamp('2011-03-12 04:00',tz = 'Europe/Moscow')
# print(stamp_moscow)
#时区意识型Timestamp对象在内部保存了一个UTC时间戳值，这个UTC值在时区转换过程中是不会发生变化的
# print(stamp_utcz.value)
# print(stamp_utcz.tz_convert('America/New_York').value)\



'''不同时区之间的运算'''
#如果两个时间序列的时区不同，在将她们合并在一起时，最终的结果就会是UTC，由于时间戳其实是以UTC存储的
# rng = pd.date_range('3/7/2012 9:30',periods= 10 ,freq= 'B')
# ts =pd.Series(np.random.randn(len(rng)),index=rng)
# print(ts)
# ts1 = ts[:7].tz_localize("Europe/London")
# ts2 = ts1[2:].tz_convert('Europe/Moscow')
#
# reuslt = ts1 + ts2
# print(reuslt.index)



'''时期及其算术运算'''
p = pd.Period(2007,freq = 'A-DEC')
#只需对Period对象加上或减去一个整数即可整数即可达到根据其频率进行位移的效果
print(p + 5)
print(p - 2)
#如果两个Period对象拥有相同的频率，则它们的差就是它们之间的单位数量
print(pd.Period('2014',freq = 'A-DEC') - p)
#period_range函数可用于创建规则的时期范围
rng = pd.period_range('2000-01-01','2000-06-30',freq= 'M')
print(rng)
#PeriodIndex类保存了一组Period,它可以在任何pandas数据结构中被用作轴索引
pd.Series(np.random.randn(6),index = rng)

