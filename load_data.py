from sklearn import preprocessing
import pandas
import numpy
import oandapy

# OANDA
ACCESS_TOKEN = "71ce71cc491ed6761f62c91529797a42-5f71bdfe7ea17c20a8b72a7a1f125707"
#ACCESS_TOKEN = "e2d515e8591ad375131f73b4d00fa046-dbcc42f596456f1562792f3639259b7f"

#FILE_NAME_MEN_SDT = "const_mean_std.csv"
#MEAN_STD = pandas.read_csv(FILE_NAME_MEN_SDT)

class ShigureLoadData:
	def __init__(self) :
		self.oanda = oandapy.API(environment="live", access_token=ACCESS_TOKEN)

	def load_data(self, shoken_code=None, date_from=None) :
		df = pandas.read_csv("../Rnn/csv/" + shoken_code + ".CSV", encoding="SHIFT_JIS", sep = ",", dtype="object", header=None)
		df = df.drop([0,1])
		df = df.dropna()
		columns = ['date','time','open','high','low','close','volume','oi','macd','signal','osci','rsi','ema_kairi_short','ema_kairi_long','ema_short','ema_long']
		df.columns = columns

		# 日付でソート
		df['date'] = pandas.to_datetime(df['date'], format='%Y-%m-%d')
		df = df.sort_values(by='date')
		df = df.reset_index(drop=True)

		# 数値列を数値化
		target_columns = ['open','high','low','close','volume','ema_short','ema_long']
		df[target_columns] = df[target_columns].astype("float32")

		df = self.common(df, target_columns, date_from=date_from)

		return df, target_columns

	def load_data_fx(self, file, date_from=None) :
		df = pandas.read_csv(file, encoding="SHIFT_JIS", sep = ",")
		df = df.dropna()

		df['date_time'] = pandas.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
		df['time'] = df['date_time'].apply(lambda d: d.timestamp())
		df = df.sort_values(by='date_time')
		df = df.set_index("date_time", drop=True)
		if "Unnamed: 0" in df.columns :
			df = df.drop(["Unnamed: 0"],axis=1)
		target_columns = df.columns.values
		return df, target_columns

	def load_data_oanda(self, date_from=None, load_count=10, granularity="M5") :
		res_hist = self.oanda.get_history(instrument="USD_JPY", granularity=granularity, count=load_count)
		df = pandas.DataFrame(res_hist['candles'])
		print ("load_data_oanda:")
		print (df.iloc[len(df)-1])

		df.to_csv("load_data_oanda.csv")
		columns = ["USD_JPY_closeAsk", "USD_JPY_closeBid", "USD_JPY_complete", "USD_JPY_highAsk", "USD_JPY_highBid", "USD_JPY_lowAsk", "USD_JPY_lowBid", "USD_JPY_openAsk", "USD_JPY_openBid", "time", "USD_JPY_volume"]
		df.columns = columns
		df.to_csv("load_data_oanda_edit_columns.csv")

		df['date_time'] = pandas.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
		df['time'] = df['date_time'].apply(lambda d: d.timestamp())
		df = df.sort_values(by='date_time')
		df = df.set_index("date_time", drop=True)
		if "Unnamed: 0" in df.columns :
			df = df.drop(["Unnamed: 0"],axis=1)
		target_columns = df.columns.values
		return df, target_columns

	def common(self, df, target_columns, date_from=None):
		# 日付で対象データを絞り込み
		if date_from != None :
			df = df[df.date_time >= date_from]
			df = df.reset_index(drop=True)

		# データを標準化
		#for column_name in target_columns:
			#df[column_name + "_before"] = df[column_name]
			#df[column_name] = preprocessing.scale(df[column_name])
			#mean_std_target = MEAN_STD[MEAN_STD["column_name"] == column_name].reset_index(drop=True)
			#print("column_name: " + column_name)
			#mean = mean_std_target["mean"][0]
			#print("mean: " + str(mean))
			#std = mean_std_target["std"][0]
			#print("std: " + str(std))
			#df[column_name] = (df[column_name] - mean) / std
		return df

	def add_avg_column(self, df, avg_count) :
		#print("add_avg_column: " + str(avg_count))
		add_column_name = "avg_" + str(avg_count)
		df[add_column_name] = df['close'].rolling(window=avg_count, min_periods=avg_count).mean()
		return df, add_column_name

	def convert_train_dataset(self, df, target_columns, look_back):
		dataX, dataY = [], []
		for i in range(look_back, len(df)-1):
			xset = numpy.array(df[target_columns][i + 1 - look_back : i + 1])
			dataX.append(xset)
			dataY.append([df["close"][i+1]]) # 答えも配列にいれておかないといけないので、データ数1の配列を作る
		return numpy.array(dataX), numpy.array(dataY)

	def convert_predict_dataset(self, df, target_columns, look_back):
		dataX = []
		for i in range(look_back, len(df)):
			xset = numpy.array(df[target_columns][i + 1 - look_back : i + 1])
			dataX.append(xset)
		return numpy.array(dataX)

	def log(self, msg) :
		f = open('log.txt','a')
		f.write(str(msg) + '\n')
		f.close()
