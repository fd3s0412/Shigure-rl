import pandas as pd
import oandapy
import time

access_token = "e2d515e8591ad375131f73b4d00fa046-dbcc42f596456f1562792f3639259b7f"
target_year = "2016"
start_time = "T00:00:00"
end_time = "T23:59:59"
granularity="H1"

oanda = oandapy.API(environment="practice", access_token=access_token)

res_hist = oanda.get_history(instrument="USD_JPY", granularity=granularity, start=target_year + "-01-01" + start_time, end=target_year + "-06-30" + end_time)
rate1 = pd.DataFrame(res_hist['candles'])

time.sleep(1)

res_hist = oanda.get_history(instrument="USD_JPY", granularity=granularity, start=target_year + "-07-01" + start_time, end=target_year + "-12-31" + end_time)
rate2 = pd.DataFrame(res_hist['candles'])

pd.concat([rate1, rate2]).to_csv("USDJPY_" + granularity + "_" + target_year + ".csv", sep=",")
print(target_year)
