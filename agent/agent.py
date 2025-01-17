import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000) # 2013년 논문 구현된 것으로 보임
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model("models/" + model_name) if is_eval else self._model()

	def _model(self): # 모델 정의
		# full connected hidden layer 3개 input > 64 > 32 > 8 > 1 
		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state): # 다음 state의 action을 받아오는 함수
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])

	# 학습을 수행하는 코드 minibatch만큼을 가져와서 학습을 하는데 dqn에 따라 처리하려면 random으로 select해야하는데
	# 순차 선택하도록 코드가 설계된 것으로 보임. 수정후 다시 수행할 필요가 있음.
	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 


	def fitnetwork(self, batch_size):
	    x_stack = np.empty(0).reshape(0, self.state_size)
	    y_stack = np.empty(0).reshape(0, self.action_size) # action의 개수
	    
	    mini_batch = random.sample(self.memory, batch_size)
	    
	    for state, action, reward, next_state, done in mini_batch:
	        Q = self.model.predict(state)
	        
	        if done:
	            Q[0, action] = reward
	        else:
	            Q[0, action] = reward + self.gamma*np.max(self.model.predict(next_state))
	            
	        y_stack = np.vstack([y_stack, Q])
	        x_stack = np.vstack([x_stack, state])
	        
	    self.model.fit(x_stack, y_stack, epochs=1, verbose=0)

	    if self.epsilon > self.epsilon_min:
		    self.epsilon *= self.epsilon_decay 
			