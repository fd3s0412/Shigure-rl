from datetime import datetime
# from keras import regularizers
from keras.callbacks import Callback
from keras.layers import Dense,Activation,Flatten,Input,concatenate,Dropout,LSTM,Reshape,Conv2D,Conv1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from sklearn import preprocessing
from load_data import ShigureLoadData
from threading import Thread
import gym.spaces
import json
import math
import numpy
import os
import pandas
import time
import timeit
import warnings
warnings.filterwarnings('ignore')

LOOK_BACK = 48
MAX_HOUR_CHART_NUM=12
TARGET_COLUMNS = ["time","USD_JPY_closeAsk","USD_JPY_closeBid","USD_JPY_highAsk","USD_JPY_highBid","USD_JPY_lowAsk","USD_JPY_lowBid","USD_JPY_openAsk","USD_JPY_openBid","USD_JPY_volume"]
TARGET_COLUMNS_FOR_TRAIN = ["time","USD_JPY_closeAsk","USD_JPY_closeBid","USD_JPY_highAsk","USD_JPY_highBid","USD_JPY_lowAsk","USD_JPY_lowBid","USD_JPY_openAsk","USD_JPY_openBid","USD_JPY_volume"
						,"USD_JPY_closeAsk_h4","USD_JPY_closeBid_h4","USD_JPY_highAsk_h4","USD_JPY_highBid_h4","USD_JPY_lowAsk_h4","USD_JPY_lowBid_h4","USD_JPY_openAsk_h4","USD_JPY_openBid_h4","USD_JPY_volume_h4"
						,"USD_JPY_closeAsk_h8","USD_JPY_closeBid_h8","USD_JPY_highAsk_h8","USD_JPY_highBid_h8","USD_JPY_lowAsk_h8","USD_JPY_lowBid_h8","USD_JPY_openAsk_h8","USD_JPY_openBid_h8","USD_JPY_volume_h8"
						,"USD_JPY_closeAsk_h12","USD_JPY_closeBid_h12","USD_JPY_highAsk_h12","USD_JPY_highBid_h12","USD_JPY_lowAsk_h12","USD_JPY_lowBid_h12","USD_JPY_openAsk_h12","USD_JPY_openBid_h12","USD_JPY_volume_h12"]
#target_model_update=5e-2
#lr=1e-4
TARGET_MODEL_UPDATE=5e-2
LR=1e-4

CONST_BEFORE = "_before"

TRAIN_COUNT = 10000
N_ACTION = 21 # ポジションを持てる段階の数×2 + 1
GRANULARITY = "H1"
LOAD_DATA_FX_FILE = "./datas/train.csv"
LOAD_DATA_FX_FILE_FORWARD = "./datas/forward.csv"
# OANDA
#ACCOUNT_ID = "7291359"
#ACCESS_TOKEN = "71ce71cc491ed6761f62c91529797a42-5f71bdfe7ea17c20a8b72a7a1f125707"
ACCOUNT_ID = "4062442"
ACCESS_TOKEN = "e2d515e8591ad375131f73b4d00fa046-dbcc42f596456f1562792f3639259b7f"

def calc_observation(df, index, columns):
	# columnsを上書きする(暫定対応)
	columns = TARGET_COLUMNS
	tmp = df[columns][index + 1 - LOOK_BACK:index + 1]
	tmp = tmp.reset_index(drop=True)
	tmp = pandas.concat([tmp, create_hour_chart(4, df, index)], axis=1)
	tmp = pandas.concat([tmp, create_hour_chart(8, df, index)], axis=1)
	tmp = pandas.concat([tmp, create_hour_chart(MAX_HOUR_CHART_NUM, df, index)], axis=1)
	#tmp.to_csv("debug.csv")

	# 標準化
	columns = tmp.columns
	for column_name in columns :
		if column_name == "time" :
			tmp[column_name] = tmp[column_name] % (3600 * 24)
		tmp[column_name] = preprocessing.scale(tmp[column_name])

	return numpy.array(tmp[columns][0:LOOK_BACK])

def create_hour_chart(target_hour, df, index) :
	tmp = df[TARGET_COLUMNS][index + 1 - LOOK_BACK * target_hour:index + 1]
	tmp = tmp.reset_index(drop=True)
	tmp = tmp.reset_index(drop=False)
	tmp["index"] = tmp["index"] - (tmp["index"] % target_hour)
	g = tmp.groupby("index")
	result = pandas.DataFrame()
	result["USD_JPY_closeAsk_h" + str(target_hour)] = tmp[~tmp["index"].duplicated(keep='last')].reset_index()["USD_JPY_closeAsk"]
	result["USD_JPY_closeBid_h" + str(target_hour)] = tmp[~tmp["index"].duplicated(keep='last')].reset_index()["USD_JPY_closeBid"]
	result["USD_JPY_highAsk_h" + str(target_hour)] = g.max().reset_index()["USD_JPY_highAsk"]
	result["USD_JPY_highBid_h" + str(target_hour)] = g.max().reset_index()["USD_JPY_highBid"]
	result["USD_JPY_lowAsk_h" + str(target_hour)] = g.min().reset_index()["USD_JPY_lowAsk"]
	result["USD_JPY_lowBid_h" + str(target_hour)] = g.min().reset_index()["USD_JPY_lowBid"]
	result["USD_JPY_openAsk_h" + str(target_hour)] = tmp[~tmp["index"].duplicated()].reset_index()["USD_JPY_openAsk"]
	result["USD_JPY_openBid_h" + str(target_hour)] = tmp[~tmp["index"].duplicated()].reset_index()["USD_JPY_openBid"]
	result["USD_JPY_volume_h" + str(target_hour)] = g.sum().reset_index()["USD_JPY_volume"]
	return result

def calc_reward(action, df, index, sum_reward, is_print=True):
	# actionに応じたrewardを即時計算（翌日まで待たない）
	reward = 0.0
	position = 0.0
	amount = 0.0
	if action > 0.0 :
		amount = math.ceil(action / 2.0)
		if action % 2 != 0 :  # 買
			position = df["USD_JPY_closeAsk"].iloc[index]
			reward = df["USD_JPY_closeBid"].iloc[index + 1] - position
		else :  # 売
			position = df["USD_JPY_closeBid"][index] * -1.0
			reward = -1.0 * position - df["USD_JPY_closeAsk"].iloc[index + 1]
	reward = reward * amount

	win_kbn = "○"
	if reward < 0 :
		win_kbn = "▲"
	sum_reward += reward
	if is_print :
		print(win_kbn + " sum_reward: " + str(sum_reward) + ", index: " + str(index) + ", action:" + str(action) + ", reward:" + str(reward) + ", position:" + str(position) + ", amount:" + str(amount))
	return reward, position, amount, sum_reward

# ------------------------------------------------------------
# 現在日時取得.
# ------------------------------------------------------------
def get_now() :
	return datetime.now().strftime("%Y%m%d_%H%M%S")

def log(msg, file="log.txt") :
	f = open(file,'a')
	f.write(msg + '\n')
	f.close()

class Game(gym.core.Env):
	def __init__(self, df, columns):
		self.df = df.reset_index(drop=True)
		self.columns = columns
		self.action_space = gym.spaces.Discrete(N_ACTION)
		self.observation_space = gym.spaces.Box(0, 999, shape=(LOOK_BACK, len(columns)), dtype=numpy.float32)
		self.time = LOOK_BACK * MAX_HOUR_CHART_NUM - 1 # TODO: 開始インデックス設定
		self.profit = 0
		self.sum_reward = 0.0

	def step(self, action):
		reward, self.position, self.amount, self.sum_reward = calc_reward(action, self.df, self.time, self.sum_reward)
		
		self.time += 1
		self.profit += reward	   
		#done = self.time >= (len(self.df) - 1)
		done = (self.time >= (len(self.df) - 2)) or (self.sum_reward <= -5.0) # 最後のindexは翌日の結果が取れないため飛ばす
		if done:
			print("[Episode End]----------profit: {}".format(self.profit))
		info = {}
		observation = calc_observation(self.df, self.time, self.columns)
		return observation, reward, done, info

	def reset(self):
		self.time = LOOK_BACK * MAX_HOUR_CHART_NUM - 1 # TODO: 開始インデックス設定
		self.profit = 0
		self.sum_reward = 0.0
		return calc_observation(self.df, self.time, self.columns)

	def render(self, mode):
		pass

	def close(self):
		pass

	def seed(self):
		pass

class MyCallback(Callback):
	def __init__(self, output_path="."):
		# Some algorithms compute multiple episodes at once since they are multi-threaded.
		# We therefore use a dictionary that is indexed by the episode to separate episodes
		# from each other.
		self.episode_start = {}
		self.observations = {}
		self.rewards = {}
		self.actions = {}
		self.metrics = {}
		self.step = 0
		self.lastreward = -99999999
		self.output_path = output_path

	def on_train_begin(self, logs):
		self.train_start = timeit.default_timer()
		self.metrics_names = self.model.metrics_names
		print('Training for {} steps ...'.format(self.params['nb_steps']))
		
	def on_train_end(self, logs):
		duration = timeit.default_timer() - self.train_start
		print('done, took {:.3f} seconds'.format(duration))

	def on_episode_begin(self, episode, logs):
		self.episode_start[episode] = timeit.default_timer()
		self.observations[episode] = []
		self.rewards[episode] = []
		self.actions[episode] = []
		self.metrics[episode] = []
		

	def on_episode_end(self, episode, logs):
		duration = timeit.default_timer() - self.episode_start[episode]
		episode_steps = len(self.observations[episode])

		# Format all metrics.
		metrics = numpy.array(self.metrics[episode])
		metrics_template = ''
		metrics_variables = []
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			for idx, name in enumerate(self.metrics_names):
				if idx > 0:
					metrics_template += ', '
				try:
					value = numpy.nanmean(metrics[:, idx])
					metrics_template += '{}: {:f}'
				except Warning:
					value = '--'
					metrics_template += '{}: {}'
				metrics_variables += [name, value]		  
		metrics_text = metrics_template.format(*metrics_variables)

		nb_step_digits = str(int(numpy.ceil(numpy.log10(self.params['nb_steps']))) + 1)
		template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], {metrics}'
		variables = {
			'step': self.step,
			'nb_steps': self.params['nb_steps'],
			'episode': episode + 1,
			'duration': duration,
			'episode_steps': episode_steps,
			'sps': float(episode_steps) / duration,
			'episode_reward': numpy.sum(self.rewards[episode]),
			'reward_mean': numpy.mean(self.rewards[episode]),
			'reward_min': numpy.min(self.rewards[episode]),
			'reward_max': numpy.max(self.rewards[episode]),
			'action_mean': numpy.mean(self.actions[episode]),
			'action_min': numpy.min(self.actions[episode]),
			'action_max': numpy.max(self.actions[episode]),
			'metrics': metrics_text,
			'output_path': self.output_path,
		}
		
		print(template.format(**variables))
		'''
		Code for saving up weights if the episode reward is higher than the last one
		'''
		
		# とりあえず学習結果を毎回保存
		template = '{output_path}/{episode}_{episode_reward}.hdf5'
		newWeights = template.format(**variables)
		self.model.save_weights(newWeights, overwrite=True)
		log(get_now() + ":\t" + newWeights, file="log_train.txt")

		last_forward=0.0
		#last_forward = ShigureRl().forward(file=LOAD_DATA_FX_FILE_FORWARD, weights=newWeights)
		print(get_now() + " last_forward: " + str(last_forward))
		log(get_now() + " last_forward: " + str(last_forward), file="log_train.txt")
		if last_forward > self.lastreward :
			self.lastreward = last_forward
			previousWeights = "{}/best_weight.hdf5".format(self.output_path)
			if os.path.exists(previousWeights): os.remove(previousWeights)
			print("The reward is higher than the best one, saving checkpoint weights")
			newWeights = "{}/best_weight.hdf5".format(self.output_path)
			self.model.save_weights(newWeights, overwrite=True)
		else:
			print("The reward is lower than the best one, checkpoint weights not updated")

		# Free up resources.
		del self.episode_start[episode]
		del self.observations[episode]
		del self.rewards[episode]
		del self.actions[episode]
		del self.metrics[episode]

	def on_step_end(self, step, logs):
		episode = logs['episode']
		self.observations[episode].append(logs['observation'])
		self.rewards[episode].append(logs['reward'])
		self.actions[episode].append(logs['action'])
		self.metrics[episode].append(logs['metrics'])
		self.step += 1

class ShigureRl:
	def __init__(self):
		self.sld = ShigureLoadData()

		# トレードで使用するRLエージェントをコンストラクタで作っておく。毎回作ると重いため。
		df, target_columns = self.sld.load_data_oanda(load_count=LOOK_BACK * MAX_HOUR_CHART_NUM, granularity=GRANULARITY)
		target_columns = TARGET_COLUMNS_FOR_TRAIN
		obs = calc_observation(df, len(df)-1, target_columns)
		df = pandas.DataFrame(data=obs, columns=target_columns)
		print(get_now() + ": get_rl_agent")
		self.agent = self.get_rl_agent(df, target_columns)

	def train_fx_rl(self):
		folder = "./fx_rl"
		os.makedirs(folder, exist_ok=True)
		df, target_columns = self.sld.load_data_fx(file=LOAD_DATA_FX_FILE)
		# columnsを上書きする(暫定対応)
		target_columns = TARGET_COLUMNS_FOR_TRAIN
		print(get_now() + ": Game")
		env = Game(df, target_columns)
		agent = self.get_rl_agent(df, target_columns, env=env)
		agent.load_weights("fx_rl/11_-5.424999999999699.hdf5")
		callback = MyCallback(folder)
		agent.fit(env, nb_steps=(len(df)-LOOK_BACK) * TRAIN_COUNT,visualize=False,verbose=2,callbacks=[callback])

	def forward(self, file=None, weights="./fx_rl/best_weight.hdf5"):
		print(get_now() + ": forward")
		df = None
		target_columns = None
		if file != None :
			df, target_columns = self.sld.load_data_fx(file=file)
		else : # ファイルが指定されていない場合はoandaからロード
			load_count = LOOK_BACK * MAX_HOUR_CHART_NUM + (24 * 30) # 直近約1か月分
			df, target_columns = self.sld.load_data_oanda(load_count=load_count, granularity=GRANULARITY)
		action_list, reward_list, reward_sum_list, sum_reward = self.get_action_list(df, target_columns, self.agent, weights=weights)
		df["action"] = action_list
		df["reward_list"] = reward_list
		df["reward_sum_list"] = reward_sum_list
		#df.to_csv("forward_oanda_" + get_now() + ".csv", sep=",")
		log(get_now() + " forward: " + str(sum_reward) + ", file: " + str(file) + ", weights: " + weights)
		return sum_reward

	def get_action_list(self, df, target_columns, agent, weights=None) :
		action_list = [None] * (LOOK_BACK * MAX_HOUR_CHART_NUM - 1)
		reward_list = [0] * (LOOK_BACK * MAX_HOUR_CHART_NUM - 1)
		sum_reward = 0.0
		for i in range(LOOK_BACK * MAX_HOUR_CHART_NUM -1, len(df)) :
			obs = calc_observation(df, i, target_columns)
			self.agent.load_weights(weights)
			action = agent.forward(obs)
			action_list.append(action)
			if i == len(df)-1 : # 最終行はrewardの計算不可
				reward_list.append(0)
			else :
				reward, position, amount, sum_reward = calc_reward(action, df, i, sum_reward, is_print=False)
				reward_list.append(reward)
		return action_list, reward_list, numpy.cumsum(reward_list), sum_reward

	def forward_weights(self) :
		import glob
		import re
		pattern = re.compile(r'^.*(\\|/)(.*)$')
		df, target_columns = self.sld.load_data_fx(file= LOAD_DATA_FX_FILE_FORWARD)
		file_list = glob.glob("fx_rl/*.hdf5")
		print(get_now() + ": file_list size: " + str(len(file_list)))
		for file_path in file_list :
			matchObj = pattern.match(file_path)
			weights = matchObj.group(2)
			self.forward_fx_rl(df=df, target_columns=target_columns, weights=weights)

	def tradestart_rl(self) :
		self.tradestart(ACCOUNT_ID)

	def tradestart(self, account_id) :
		import oandapy
		oanda = oandapy.API(environment="practice",access_token=ACCESS_TOKEN)
		unixtime_before = 0
		order_1 = None
		before_kbn = None
		before_amount = None
		shorizumi_flg = False
		while True :
			unixtime = int(time.mktime(datetime.now().timetuple()))
			second = unixtime % (60 * 60)

			if unixtime - unixtime_before >= 30 :
				shorizumi_flg = False

			target_minutes = 59 # 毎時59分55秒以降に処理実施
			if (shorizumi_flg == False and second >= 60 * target_minutes + 50 and second <= 60 * (target_minutes + 1)) :
				unixtime_before = unixtime
				shorizumi_flg = True
				buy_sell_kbn, amount = self.get_buy_sell_kbn_rl()
				print(get_now() + ": " + str(buy_sell_kbn) + ", amount: " + str(amount))
				log(get_now() + ": " + str(buy_sell_kbn) + ", amount: " + str(amount), file="log_trade.txt")
				if (before_kbn != buy_sell_kbn) or (before_amount != amount) :
					before_kbn = buy_sell_kbn
					before_amount = amount
					if order_1 != None :
						try:
							close_pos = oanda.close_trade(account_id, order_1)
							order_1 = None
							print(get_now() + ": close_trade:")
							print(close_pos)
						except:
							import traceback
							traceback.print_exc()
							print(get_now() + ": close_trade error: " + str(order_1))
					if buy_sell_kbn != None :
						try :
							units = amount * 3000
							order_1 = oanda.create_order(account_id,instrument="USD_JPY",units=units,side=buy_sell_kbn,type="market")
							order_1 = order_1['tradeOpened']['id']
							print(get_now() + " " + buy_sell_kbn + ": " + str(order_1))
						except:
							import traceback
							traceback.print_exc()
							print(get_now() + ": error: " + buy_sell_kbn)
							before_kbn = None
			time.sleep(1)

	def get_buy_sell_kbn_rl(self) :
		try:
			print(get_now() + ": get_buy_sell_kbn_rl")
			df, target_columns = self.sld.load_data_oanda(load_count=LOOK_BACK * MAX_HOUR_CHART_NUM, granularity=GRANULARITY)
			print(get_now() + ": load_weights")
			self.agent.load_weights("fx_rl/best_weight.hdf5")
			print(get_now() + ": forward")
			action = self.agent.forward(calc_observation(df, len(df)-1, target_columns))
			print(get_now() + ": action: " + str(action))
			amount = 0.0
			if action > 0.0 :
				amount = math.ceil(action / 2.0)
				if action % 2 != 0 :  # 買
					return "buy", amount
				else :  # 売
					return "sell", amount
			else :
				return None, None
		except:
			import traceback
			traceback.print_exc()
			print(get_now() + ": weights file not found.")
			return None, None

	def get_rl_agent(self, df, target_columns, env=None):
		if env == None :
			print(get_now() + ": Game")
			env = Game(df, target_columns)
		print(get_now() + ": model_rl")
		model = self.model_rl(env.observation_space, target_columns, N_ACTION)
		print(get_now() + ": SequentialMemory")
		memory = SequentialMemory(limit=100000, window_length=1)
		print(get_now() + ": EpsGreedyQPolicy")
		policy = EpsGreedyQPolicy(eps=0.1)
		print(get_now() + ": DQNAgent")
		agent = DQNAgent(model=model, nb_actions=N_ACTION, memory=memory, nb_steps_warmup=100, target_model_update=TARGET_MODEL_UPDATE, policy=policy)
		print(get_now() + ": compile")
		agent.compile(Adam(lr=LR), metrics=['mae'])
		return agent

	def model_rl_old(self, observation_space, target_columns, n_action):
		model = Sequential()
		model.add(Reshape(observation_space.shape, input_shape=(1,) + observation_space.shape))
		model.add(LSTM(observation_space.shape[0] * observation_space.shape[1], return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(observation_space.shape[0] * observation_space.shape[1], return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(n_action))
		model.add(Activation('softmax'))
		self.model = model
		return model

	def model_rl(self, observation_space, target_columns, n_action):
		item_count = LOOK_BACK * len(target_columns)
		dense_count = n_action
		print("item_count:" + str(item_count))
		model = Sequential()
		model.add(Flatten(input_shape=(1, ) + observation_space.shape))
		model.add(Dropout(0.2))
		model.add(Dense(dense_count))
		model.add(Dropout(0.2))
		model.add(Dense(dense_count))
		model.add(Dense(n_action, activation="softmax"))
		self.model = model
		return model

	def model_rnn(self, input_shape):
		model = Sequential()
		model.add(LSTM(4, input_shape=input_shape, return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(4, return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(output_dim=1))
		model.add(Activation('linear'))
		self.model = model
		return model

def build_parser():
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("--mode",dest="mode",help="train, forward, tradestart", default="forward")
	parser.add_argument("--model",dest="model",help="rl, rnn",default="rl")
	parser.add_argument("--target",dest="target",help="kabu, fx, fx_all, oanda", default="oanda")
	parser.add_argument("--shoken_code",dest="shoken_code",help="all, etc 3656", default="3656")
	parser.add_argument("--date_from",dest="date_from",help="etc 2018-09-18", default=None)
	return parser

def main():
	parser = build_parser()
	op = parser.parse_args()

	tmp = ShigureRl()
	if op.mode == "train" :
		tmp.train_fx_rl()
	elif op.mode == "forward":
		tmp.forward()
	elif op.mode == "tradestart" :
		tmp.tradestart_rl()

main()
