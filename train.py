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
verbose = False

# episode 횟수만큼 반복 학습
for e in range(1, episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0, window_size + 1)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		# sit
		next_state = getState(data, t + 1, window_size + 1)
		reward = 0

		if action == 1: # buy
			agent.inventory.append(data[t]) # inventory = [] 구입한 가격이 들어있는 저장소
			if verbose:
				print("{}/{} {}".format(t,l-1, len(agent.inventory)) + " Buy: " + formatPrice(data[t]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			# reward의 최소값을 0으로 고정하고 있지만 손실에 대한 penalty를 주는 편이 낫지 않을까?
			#reward = max(data[t] - bought_price, 0) 
			reward = data[t] - bought_price 
			total_profit += data[t] - bought_price
			if verbose:
				print("{}/{} {}".format(t,l-1, len(agent.inventory)) + " Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
		elif action == 0:
			if verbose:
				print("{}/{} {}".format(t,l-1, len(agent.inventory)) + " Sit:")
		else:
			reward = -1 # 없는데 Sell을 시도할 때는 penalty로 1원을 차감
			if verbose:
				print("{}/{} {} {}".format(t,l-1, len(agent.inventory), action))
			
			

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			#agent.expReplay(batch_size)
			agent.fitnetwork(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e) + "_" + stock_name)

agent.model.save("models/model_ep" + str(e) + "_" + stock_name)
