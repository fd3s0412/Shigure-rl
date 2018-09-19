# -*- coding: utf-8 -*-
import sys
import pandas
import matplotlib.pyplot as plt
import glob

class GraphUtil :
	def show(self, year="2017", ep="*") :
		list = glob.glob("./forward_" + year + "/*" + ep + ".hdf5.csv")
		for path in list :
			print(path)
			result = pandas.read_csv(path, delimiter=',')
			result1 = result[['reward', 'ruiseki']]

			# 一日未来を予測しているため、予測と実際を比較するには、
			# 実際の値を１日過去方向にずらす必要がある
			#result['hendoryo'] = result['hendoryo'].shift(-1)

			result1.plot()
			plt.grid()
			plt.show()

def build_parser():
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("--year",dest="year",help="2016, 2017", default="2017")
	parser.add_argument("--ep",dest="ep",help="1, 2, 3...", default="*")
	return parser

if __name__ == "__main__":
	parser = build_parser()
	op = parser.parse_args()

	util = GraphUtil()
	util.show(op.year, op.ep)
