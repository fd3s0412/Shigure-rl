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
import numpy
import os
import pandas
import time
import timeit
import warnings
warnings.filterwarnings('ignore')

LOOK_BACK = 40
TRAIN_COUNT = 10000
N_ACTION = 3
GRANULARITY = "H1"
SPREAD = 0.01
LOAD_DATA_FX_FILE = "../Rnn/data_fx/USDJPY_H1.csv"
LOAD_DATA_FX_FILE_FORWARD = "../Rnn/data_fx/USDJPY_H1_2015.csv"
# OANDA
#ACCOUNT_ID = "7291359"
#ACCESS_TOKEN = "71ce71cc491ed6761f62c91529797a42-5f71bdfe7ea17c20a8b72a7a1f125707"
ACCOUNT_ID = "4062442"
ACCESS_TOKEN = "e2d515e8591ad375131f73b4d00fa046-dbcc42f596456f1562792f3639259b7f"

def calc_observation(df, index, columns):
	return numpy.array(df[columns][index + 1 - LOOK_BACK:index + 1])

def calc_reward(action, df, index, columns, position):
	reward = 0.0
	if position > 0.0 : # 買ポジション
		reward = df["close_before"][index] - SPREAD - position
	elif position < 0.0 : # 売ポジションを持っている場合
		reward = -1.0 * position - df["close_before"][index] + SPREAD

	position = 0.0
	if action == 0:  # 買
		position = df["close_before"][index]
	elif action == 1:  # 売
		position = df["close_before"][index] * -1.0

	return reward, position

def calc_reward_forward(action, df, index, columns, position):
	reward = 0.0
	if action == 0:  # 買
		if position < 0.0 : # 売ポジションを持っている場合、決済する(売った時の金額 - 現在価格 = 決済額）
			reward = -1.0 * position - df["close_before"][index]
			position = 0.0
		if position == 0.0 :
			position = df["close_before"][index]
	elif action == 1:  # 売
		if position > 0.0 : # 買ポジションを持っている場合、決済する(現在価格 - 買った金額 = 決済額）
			reward = df["close_before"][index] - position
			position = 0.0
		if position == 0.0 :
			position = df["close_before"][index] * -1.0
	else:  # ポジションを持たない
		if position < 0.0 : # 売ポジションを持っている場合、決済する(売った時の金額 - 現在価格 = 決済額）
			reward = -1.0 * position - df["close_before"][index]
		if position > 0.0 : # 買ポジションを持っている場合、決済する(現在価格 - 買った金額 = 決済額）
			reward = df["close_before"][index] - position
		position = 0.0
	return reward, position

# ------------------------------------------------------------
# 現在日時取得.
# ------------------------------------------------------------
def get_now() :
	return datetime.now().strftime("%Y%m%d_%H%M%S")

class Game(gym.core.Env):
	def __init__(self, df, columns):
		self.df = df.reset_index(drop=True)
		self.columns = columns
		self.action_space = gym.spaces.Discrete(N_ACTION)
		self.observation_space = gym.spaces.Box(0, 999, shape=(LOOK_BACK, len(columns)), dtype=numpy.float32)
		self.time = LOOK_BACK - 1
		self.profit = 0
		self.position = 0

	def step(self, action):
		reward, self.position = calc_reward(action, self.df, self.time, self.columns, self.position)
		
		self.time += 1
		self.profit += reward	   
		done = self.time >= (len(self.df) - 1)
		if done:
			print("[Episode End]----------profit: {}".format(self.profit))
		info = {}
		observation = calc_observation(self.df, self.time, self.columns)
		return observation, reward, done, info

	def reset(self):
		self.time = LOOK_BACK - 1
		self.profit = 0
		self.position = 0
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
		
		#tmp = ShigureRl("fx", weights_file=newWeights)
		#action, rieki_goukei = tmp.forward()
		#print("fit rieki_goukei: " + str(numpy.sum(self.rewards[episode])))
		#print("callback forward rieki_goukei: " + str(rieki_goukei))
		#print("rieki_goukei_max: " + str(self.lastreward))
		
		if numpy.sum(self.rewards[episode]) > self.lastreward :
		#if numpy.sum(self.rewards[episode]) > self.lastreward:
			
			previousWeights = "{}/best_weight.hdf5".format(self.output_path)
			if os.path.exists(previousWeights): os.remove(previousWeights)
			#self.lastreward = numpy.sum(self.rewards[episode])
			self.lastreward = numpy.sum(self.rewards[episode])
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
		df, target_columns = self.sld.load_data_oanda(look_back=LOOK_BACK, granularity=GRANULARITY)
		print(get_now() + ": get_buy_sell_kbn_rl")
		print (df.iloc[len(df)-1])
		print(get_now() + ": get_rl_agent")
		self.agent = self.get_rl_agent(df, target_columns)

	def train_fx_rl(self):
		folder = "./fx_rl"
		os.makedirs(folder, exist_ok=True)
		df, target_columns = self.sld.load_data_fx(file=LOAD_DATA_FX_FILE)
		print(get_now() + ": Game")
		env = Game(df, target_columns)
		agent = self.get_rl_agent(df, target_columns, env=env)
		#agent.load_weights("fx_rl/-0.034999999999939746_382.hdf5")
		callback = MyCallback(folder)
		agent.fit(env, nb_steps=(len(df)-LOOK_BACK) * TRAIN_COUNT,visualize=False,verbose=2,callbacks=[callback])

	def train_fx_rnn(self):
		folder = "./fx_rnn"
		os.makedirs(folder, exist_ok=True)
		df, target_columns = self.sld.load_data_fx(file=LOAD_DATA_FX_FILE)
		x, y = self.sld.convert_train_dataset(df, target_columns, LOOK_BACK)
		xDf = pandas.DataFrame(x[0])
		model = self.model_rnn((LOOK_BACK, len(target_columns)))
		model.compile(loss='mse', optimizer='rmsprop')
		callback = MyCallback(folder)
		model.fit(x, y, epochs=TRAIN_COUNT, verbose=2)
		model.save(folder + "/best_weight.hdf5")

	def forward_fx_rl(self, df=None, target_columns=None, weights="best_weight.hdf5"):
		if df is None :
			df, target_columns = self.sld.load_data_fx(file= LOAD_DATA_FX_FILE_FORWARD)
		self.agent.load_weights("./fx_rl/" + weights)
		df = self.forward_rl(df, target_columns, self.agent)
		df.to_csv("forward_fx_rl-" + weights + ".csv", sep=",")
		print(get_now() + ": forward_fx_rl " + weights)
		self.log(get_now() + ": forward_fx_rl " + weights + " " + str(df["date"][len(df)-1]) + " " + str(df["ruiseki"][len(df)-1]))

	def forward_oanda_rl(self, weights="best_weight.hdf5"):
		df, target_columns = self.sld.load_data_oanda(look_back=LOOK_BACK, granularity=GRANULARITY)
		self.agent.load_weights("./fx_rl/" + weights)
		df = self.forward_rl(df, target_columns, self.agent)
		df.to_csv("forward_oanda_rl.csv", sep=",")
		print(get_now() + ": forward_oanda_rl")
		self.log(get_now() + ": forward_fx_rl " + weights + " " + str(df["date"][len(df)-1]) + " " + str(df["ruiseki"][len(df)-1]))

	def forward_rl(self, df, target_columns, agent) :
		position = 0.0
		df['position'] = 0.0
		df['reward'] = 0.0
		df['ruiseki'] = 0.0
		for i in range(LOOK_BACK-1, len(df)) :
			obs = calc_observation(df, i, target_columns)
			action = agent.forward(obs)
			reward, position = calc_reward_forward(action, df, i, target_columns, position)
			df['position'][i] = position
			df['reward'][i] = reward
			df['ruiseki'][i] = df['reward'].sum()
		return df

	def forward_oanda_rnn(self):
		df, target_columns = self.sld.load_data_oanda(look_back=LOOK_BACK)
		x = self.sld.convert_predict_dataset(df, target_columns, LOOK_BACK)
		model = load_model("./fx_rnn/best_weight.hdf5")
		predict_list = model.predict(x)
		df = df.drop(range(LOOK_BACK))
		df = df.reset_index(drop=True)
		df['future_price'] = predict_list
		# 予測価格列を追加
		closes = numpy.array(df[['close']].values.flatten().tolist(), dtype='float64')
		mean = closes.mean()
		std = closes.std()
		df['future_price_before'] = df['future_price'] * std + mean
		position = 0.0
		df['position'] = 0.0
		df['reward'] = 0.0
		df['ruiseki'] = 0.0
		for i in range(len(df)):
			action = 2.0
			now_price = df['close'][i]
			future_price = df['future_price'][i]
			if now_price < future_price :
				action = 0.0
			elif now_price > future_price :
				action = 1.0
			reward, position = calc_reward(action, df, i, target_columns, position)
			df['position'][i] = position
			df['reward'][i] = reward
			df['ruiseki'][i] = df['reward'].sum()
		df.to_csv("forward_oanda_rnn.csv", sep=",")
		print("forward_oanda_rnn")

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
		self.tradestart(ACCOUNT_ID, self.get_buy_sell_kbn_rl)

	def tradestart_rnn(self) :
		self.tradestart(ACCOUNT_ID, self.get_buy_sell_kbn_rnn)

	def tradestart(self, account_id, get_buy_sell_kbn, seccond=45) :
		import oandapy
		oanda = oandapy.API(environment="practice",access_token=ACCESS_TOKEN)
		unixtime_before = 0
		order_1 = None
		before_kbn = None
		shorizumi_flg = False
		while True :
			unixtime = int(time.mktime(datetime.now().timetuple()))
			second = unixtime % (60 * 60)

			if unixtime - unixtime_before >= 30 :
				shorizumi_flg = False

			# 4分xx秒～4分50秒の間に実施
			if (shorizumi_flg == False and second >= 60 * 59 + 55) :
				unixtime_before = unixtime
				shorizumi_flg = True
				buy_sell_kbn = get_buy_sell_kbn()
				print(get_now() + ": " + str(buy_sell_kbn))
				self.log(get_now() + ": " + str(buy_sell_kbn))
				if before_kbn != buy_sell_kbn :
					before_kbn = buy_sell_kbn
					if order_1 != None :
						try:
							close_pos = oanda.close_trade(account_id, order_1)
							order_1 = None
							print(get_now() + ": close_trade:")
							print(close_pos)
						except:
							print(get_now() + ": close_trade error: " + str(order_1))
					if buy_sell_kbn != None :
						try :
							order_1 = oanda.create_order(account_id,instrument="USD_JPY",units=30000,side=buy_sell_kbn,type="market")
							order_1 = order_1['tradeOpened']['id']
							print(get_now() + " " + buy_sell_kbn + ": " + str(order_1))
						except:
							print(get_now() + ": error: " + buy_sell_kbn)
							before_kbn = None
			time.sleep(1)

	def get_buy_sell_kbn_rl(self) :
		try:
			df, target_columns = self.sld.load_data_oanda(look_back=LOOK_BACK, granularity=GRANULARITY)
			print(get_now() + ": get_buy_sell_kbn_rl")
			print (df.iloc[len(df)-1])
			print(get_now() + ": load_weights")
			self.agent.load_weights("fx_rl/best_weight.hdf5")
			print(get_now() + ": forward")
			action = self.agent.forward(calc_observation(df, len(df)-1, target_columns))
			if action == 0 :
				return "buy"
			elif action == 1 :
				return "sell"
			else :
				return None
		except:
			print(get_now() + ": weights file not found.")
			return None

	def get_buy_sell_kbn_rnn(self) :
		df, target_columns = self.sld.load_data_oanda(look_back=LOOK_BACK)
		x = self.sld.convert_predict_dataset(df, target_columns, LOOK_BACK)
		try:
			model = load_model("fx_rnn/best_weight.hdf5")
			predict_list = model.predict(x)
			future_price = predict_list[len(predict_list)-1]
			now_price = df["close"][len(df)-1]
			if now_price < future_price :
				return "buy"
			elif now_price > future_price :
				return "sell"
			else :
				return None
		except:
			print("weights file not found.")
			return None

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
		agent = DQNAgent(model=model, nb_actions=N_ACTION, memory=memory, nb_steps_warmup=100, target_model_update=5e-2, policy=policy)
		print(get_now() + ": compile")
		agent.compile(Adam(lr=1e-4), metrics=['mae'])
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
		model = Sequential()
		model.add(Flatten(input_shape=(1, ) + observation_space.shape))
		model.add(Dense(item_count))
		model.add(Dropout(0.2))
		model.add(Dense(item_count))
		model.add(Dropout(0.2))
		model.add(Dense(item_count))
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

	def log(self, msg) :
		f = open('log.txt','a')
		f.write(msg + '\n')
		f.close()

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
		if op.target == "fx" :
			if op.model == "rl" :
				tmp.train_fx_rl()
			elif op.model == "rnn" :
				tmp.train_fx_rnn()
	elif op.mode == "forward":
		if op.target == "fx" :
			if op.model == "rl" :
				tmp.forward_fx_rl()
			elif op.model == "rnn" :
				tmp.forward_fx_rnn()
		elif op.target == "fx_all" :
			tmp.forward_weights()
		elif op.target == "oanda" :
			if op.model == "rl" :
				tmp.forward_oanda_rl()
			elif op.model == "rnn" :
				tmp.forward_oanda_rnn()
	elif op.mode == "tradestart" :
		if op.model == "rl" :
			tmp.tradestart_rl()
		elif op.model == "rnn" :
			tmp.tradestart_rnn()

main()
