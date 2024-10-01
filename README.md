# Cellworld_Offline
 Offline RL on cellworld_gym

# Mouse data

We have 100,000:

![image](https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/c0e6801a-02ee-48a1-b117-e9d515763a48)

![image](https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/691eeadd-7123-4743-93ae-de58ef212501)

# BC/ CQL /IQL

evaluate for 100 epochs when captured -1, when success 1
The mean reward can be shown as follows: 

- BC: -17.42

- CQL: -150.08

- IQL: -29.06


It seems I have identified an intriguing case where Behavior Cloning (BC) is outperforming other offline RL methods, like CQL and Implicit Q-learning, which are designed to handle suboptimal datasets. This suggests that something unusual may be happening with the offline RL algorithms in your setting, potentially due to issues with how pessimistic they are when evaluating and learning from the dataset.

To investigate this further, comparing the Q-values (which represent the expected future reward from a given state-action pair) between your offline RL policies and a less pessimistic baseline (like BC) could shed light on why this is happening. Here's a detailed plan for how to approach this:

<img width="900" alt="image" src="https://github.com/user-attachments/assets/f4a43257-6409-43af-8243-e0debb5fe577">


<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/343b4406-f617-4ffd-a1e8-d7892f7fdc38">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/a107acf6-4fdb-4627-ba8f-2e3f2082e3e8">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/02d8b339-e903-46bf-a484-7e6d981d4f0b">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/bb9c8853-9610-4bd6-8cc0-fae8bbca7c2f">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/48a74248-c6a2-4f78-a7a3-8fdeb49ebd97">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/43c0821b-4fe4-43b7-94a1-6956ffd70425">




KL divergence()
- 7.569894318469576e-07 for IQL
- 1.4248751999871712e-07 for BC
- DQN: 1.1685386452255908
- QRDQN: 1.1869605695309853
- Dreamer-v3: 1.1931324363223705



## Improving Offline Learning

<img width="1000" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/4935fb55-250f-4e6f-a4e6-2b4a5a17986f">


<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/b2aeea05-a0bc-4b06-b8b6-daae5bbea07c">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/45864462-5483-4373-bd1d-d0fd5a8582ee">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/bbfc7678-ba84-4b33-ab0b-a8bf374b46ca">
<img width="500" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/c19a948d-81e1-423c-a29f-6e6ef821b0e3">

<img width="1000" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/2eec2f26-90d1-41ad-9136-2f519fbcf631">


Train for 100 epch, and evaluate for 20 epsolid after each epch:

<img width="1200" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/bd556d47-b540-48b4-9b52-4240fe24e226">

After training plot:

<img width="1200" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/3306f23f-b254-4dac-9e3d-b35539db5f3a">

## Discrete to continous


<img width="500" alt="image" src="https://github.com/user-attachments/assets/819019a9-5f6f-473e-954d-d833ca98c199">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/8cd14b4b-5420-4e17-850b-4d3ef126fb7e">


<img width="500" alt="image" src="https://github.com/user-attachments/assets/698f4675-f686-432a-aa5f-b5e325537ab0">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/d5700675-cf54-4d89-9dd3-09b4de618540">




<img width="500" alt="image" src="https://github.com/user-attachments/assets/b57851e9-a226-485d-8906-56cdbd715afa">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/6f20a855-2ade-452d-8f3b-90e5ba82d956">


<img width="500" alt="image" src="https://github.com/user-attachments/assets/40fcf0b1-e908-40c9-b1db-84565e60f066">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/e126c0db-7ab3-421d-9e68-e15ece209c46">

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/2f0ffa66-066d-420c-aef9-9fa9335b70ca">
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/8fafcd63-1e95-429e-9233-c866b10f62cc">


## Offline RL Implementation with Planner-Greedy Exploration

- There is a planning rate and RL rate. At the starting point, the planning rate is 100% and the RL rate is 0%. 

- And we are using the offline RL methods such as BC or Implict Q learning. During the training process, we are gradually reducing the planing rate and move the agent completely to RL based.

- A mechnism to reduce the planning rate: Most simple one--linear decay, exponential decay. Performance based decay.

```
initial_planning_rate = 1.0
final_planning_rate = 0.0
planning_rate: form initial_planning_rate -> final_planning_rate

## initilized the buffer with tlppo tractories
replay_buffer = collet_start_data(data_points = 10000)

# train
for episode in range(total_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < planning_rate:
            action = tlppo.(state)
        else:
            action = offline_rl(state)
        next_state, reward, done, _ = env.step(action)
        # here, when saving all transitions, should I save all transitions or only the planner's transition?
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
    offline_rl_agent.train(replay_buffer)
```

Planner still not ready, and I used a DQN agent with a success rate about 60%.

## expert start
<img width="815" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/ebb282e6-ec3e-4b24-a3dd-90fa20498ab5">

## random start
<img width="815" alt="image" src="https://github.com/hanshuo-shuo/Cellworld_Offline/assets/80494218/3717f1cf-5d33-4998-aacf-0877c3b879bb">

