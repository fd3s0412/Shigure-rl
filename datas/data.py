import pandas
import oandapy
import time
import glob

def download(target_year="2018", target=None) :
	access_token = "e2d515e8591ad375131f73b4d00fa046-dbcc42f596456f1562792f3639259b7f"
	start_time = "T00:00:00"
	end_time = "T23:59:59"
	granularity="H1"

	oanda = oandapy.API(environment="practice", access_token=access_token)

	for target in target_list :
		res_hist = oanda.get_history(instrument=target, granularity=granularity, start=target_year + "-01-01" + start_time, end=target_year + "-06-30" + end_time)
		rate1 = pandas.DataFrame(res_hist['candles'])

		time.sleep(1)

		res_hist = oanda.get_history(instrument=target, granularity=granularity, start=target_year + "-07-01" + start_time, end=target_year + "-12-31" + end_time)
		rate2 = pandas.DataFrame(res_hist['candles'])

		pandas.concat([rate1, rate2]).to_csv(target + "-" + granularity + "-" + target_year + ".csv", sep=",")
		print(target_year + ": " + str(target))

def concat_all(target_list, year_list, file_name) :
	df = pandas.DataFrame()
	for year in year_list :
		line = pandas.DataFrame()
		for target in target_list :
			tmp = pandas.read_csv(target + "-H1-" + year + ".csv")
			tmp = tmp.set_index("time", drop=True)
			tmp = tmp.drop(["Unnamed: 0"],axis=1) 
			tmp = tmp.add_prefix(target + "_")
			line = pandas.concat([line, tmp], axis=1)
		#print(line.tail())
		df = pandas.concat([df, line])
	df.to_csv(file_name + ".csv")

def build_parser():
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("--year",dest="year",help="2016, 2017", default="2018")
	parser.add_argument("--target",dest="target",help="USD_JPY,AUD_JPY")
	parser.add_argument("--mode",dest="mode", default="")
	return parser

if __name__ == "__main__":
	parser = build_parser()
	op = parser.parse_args()

	#target_list = ["USD_JPY", "EUR_USD", "EUR_JPY", "ZAR_JPY", "TRY_JPY", "AUD_JPY", "GBP_USD", "GBP_JPY", "AUD_USD"]
	target_list = ["USD_JPY", "EUR_USD", "EUR_JPY", "GBP_USD", "AUD_USD"]

	#year_list = ["2016", "2017", "2018"]
	year_list = ["2018"]

	if op.mode == "concat" :
		year_list = ["2011","2012","2013","2014","2015","2016","2017"]
		concat_all(target_list, year_list, "train")
		concat_all(target_list, ["2018"], "forward")
	else :
		year_list = ["2011","2012","2013","2014","2015","2016","2017","2018"]
		for year in year_list :
			time.sleep(1)
			download(year, target_list)
