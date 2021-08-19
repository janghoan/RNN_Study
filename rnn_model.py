import numpy as np

# 아래의 코드는 의사 코드

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이
input_dim = 4  # 입력의 차원. NMP에서는 보통 단어 벡터의 차원
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량

inputs = np.random.random((timesteps, input_dim)) # 입력에 해당되는 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0으로 초기화

print(hidden_state_t)

Wx = np.random.random((hidden_size, input_dim)) # 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size)) # 은닉 상태에 대한 가중치
b = np.random.random((hidden_size,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t)+ b)
    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states))
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis = 0)

print(total_hidden_states)

