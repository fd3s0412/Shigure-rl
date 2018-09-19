import pandas as pd
import oandapy
import time

access_token = "e2d515e8591ad375131f73b4d00fa046-dbcc42f596456f1562792f3639259b7f"
target_date = "2018-08-"
start_time = "T00:00:00"
end_time = "T23:59:59"

oanda = oandapy.API(environment="practice", access_token=access_token)
for i in range(10) :
    start_day = '{0:02d}'.format(1 + 3 * i)
    end_day = '{0:02d}'.format(3 + 3 * i)
    res_hist = oanda.get_history(instrument="USD_JPY", granularity="M1", start=target_date + start_day + start_time, end=target_date + end_day + end_time)
    rate = pd.DataFrame(res_hist['candles'])
    rate.to_csv("USDJPY_5m_" + target_date + start_day + ".csv", sep=",")
    print(target_date + start_day)
    time.sleep(1)

