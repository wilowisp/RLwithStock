import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print("Usage: python evaluate.py [stock] [model] [tgtposi]")
	exit()

stock_name, model_name, tgtposi = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
print("MODEL({}) is loaded.".format(model_name))
data = getStockDataVec(stock_name, tgtposi)
l = len(data) - 1
batch_size = 32
verbose = True


state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []
actioncnt = {"buy":0, "sell":0, "hold":0, "0sell":0}

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0

	if action == 1: # buy
		agent.inventory.append(data[t]) # inventory = [] 구입한 가격이 들어있는 저장소
		actioncnt['buy'] += 1
		if verbose:
			print("{}/{} {}".format(t,l-1, len(agent.inventory)) + " Buy: " + formatPrice(data[t]))

	elif action == 2 and len(agent.inventory) > 0: # sell
		actioncnt['sell'] += 1
		bought_price = agent.inventory.pop(0)
		# reward의 최소값을 0으로 고정하고 있지만 손실에 대한 penalty를 주는 편이 낫지 않을까?
		#reward = max(data[t] - bought_price, 0) 
		reward = data[t] - bought_price 
		total_profit += data[t] - bought_price
		if verbose:
			print("{}/{} {}".format(t,l-1, len(agent.inventory)) + " Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
	elif action == 0:
		reward = -.1 # 매수/매도보다 그대로 있으려는 경향이 커져서 매우 작은 penalty 부여
		actioncnt['hold'] += 1
		if verbose:
			print("{}/{} {}".format(t,l-1, len(agent.inventory)) + " Sit:")
	else:
		actioncnt['0sell'] += 1
		reward = -1000 # 없는데 Sell을 시도할 때는 penalty 부여
		if verbose:
			print("{}/{} {} {}".format(t,l-1, len(agent.inventory), action))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print(model_name + "--------------------------------")
		print(stock_name + " Total Profit: " + formatPrice(total_profit))
		print(actioncnt)
		print("--------------------------------")