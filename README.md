# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## POLICY ITERATION ALGORITHM
The policy iteration algorithm is a method for finding the optimal policy in a Markov Decision Process (MDP). It alternates between policy evaluation (finding the value function for a fixed policy) and policy improvement (updating the policy using the new value function).

## VALUE ITERATION ALGORITHM
- Value iteration is a method of computing an optimal MDP policy  and its value.
- It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
- The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:
1. Initialize the value function `V(s)` arbitrarily for all states `s`.
2. Repeat until convergence:
    - Initialize aaction-value function `Q(s, a)` arbitrarily for all states `s` and actions `a`.
    - For all the states s and all the action a of every state:
        - Update the action-value function `Q(s, a)` using the Bellman equation.
        - Take the value function `V(s)` to be the maximum of `Q(s, a)` over all actions `a`.
        - Check if the maximum difference between `Old V` and `new V` is less than `theta`, where theta is a **small positive number** that determines the **accuracy of estimation**.
3. If the maximum difference between Old V and new V is greater than theta, then
    - Update the value function `V` with the **maximum action-value** from `Q`.
    - Go to **step 2**.
4. The optimal policy can be constructed by taking the **argmax** of the action-value function `Q(s, a)` over all actions `a`.
5. Return the optimal policy and the optimal value function.

## VALUE ITERATION FUNCTION
#### Name: Ajay Aswin M
#### Register Number: 212222240005
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, s_prime, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[s_prime] * (1.0 - done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
        pi= lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi

```

## OUTPUT:

<img width="553" height="206" alt="image" src="https://github.com/user-attachments/assets/886c5cfd-00e9-4687-8b2d-9455ea73497d" />

<img width="841" height="80" alt="image" src="https://github.com/user-attachments/assets/ee1ed8f3-6128-4251-ab52-2817d98b50b8" />


<img width="641" height="174" alt="image" src="https://github.com/user-attachments/assets/7e7a1be0-8708-48b6-b2a4-f90b3405501d" />


## RESULT:

Thus we successfully developed Python program to find the optimal policy for the given MDP using the value iteration algorithm.
