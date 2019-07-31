from agent.agent import Agent
from functions import *
import sys

if len(sys.argv) != 5:
	print("Usage: python train.py [stock] [window] [episodes] [close_position]")
	exit()
	
# stock은 파일이름의 확장자 제외부분, 종목에 대한 데이터 파일 
# Date,Open,High,Low,Close,Adj Close,Volume
# window는 이전 내용을 보는 길이
# episode는 학습하는 횟수
# ^GSPC 50 20 4
# A069500 50 20 1
stock_name, window_size, episode_count, posi= sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

tgtposi=posi # ^GSPC.csv는 4, A069500은 1
agent = Agent(window_size)
data = getStockDataVec(stock_name, tgtposi)
# 데이터의 크기? 
l = len(data) - 1
# TODO 확인
batch_size = 32
train_period = 100
verbose = False

# episode 횟수만큼 반복 학습
historicalmax = 0
for e in range(1, episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []
	actioncnt = {"buy":0, "sell":0, "hold":0, "0sell":0}

	for t in range(l):
		action = agent.act(state)

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
			reward = -10 # 없는데 Sell을 시도할 때는 penalty 부여
			if verbose:
				print("{}/{} {} {}".format(t,l-1, len(agent.inventory), action))
			
			

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print(actioncnt)
			print("--------------------------------")
			
			if total_profit > historicalmax:
				historicalmax = total_profit
				agent.model.save("models/model_ep" + str(e) + "_" + stock_name + "({0})".format(total_profit))

		# memory에 저장된 것이 크면 사례마다 학습하던 것을 일정 빈도마다 학습하도록 변경
		if len(agent.memory) > batch_size and t % train_period == 0:
			#agent.expReplay(batch_size)
			agent.fitnetwork(batch_size)

	if e % 50 == 0:
		agent.model.save("models/model_ep" + str(e) + "_" + stock_name)

