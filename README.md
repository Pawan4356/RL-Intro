# **Reinforcement Learning — Intro**

### **Core Concepts**

#### **1. The ARO Framework**

Every Reinforcement Learning (RL) problem consists of:

* **Agent** → Learns and makes decisions
* **Environment** → The world the agent interacts with
* **Reward** → Feedback signal guiding behavior

#### **2. Types of Tasks**

* **Episodic** → Has a start and end (e.g., games)
* **Continuing** → Continuous process (e.g., trading, robotics)

#### **3. Key RL Components**

| Concept        | Description                        |
| -- | - |
| **Policy (π)** | Strategy mapping states to actions |
| **Value (V)**  | Expected long-term reward          |
| **Model**      | Predicts environment behavior      |
| **Reward**     | Immediate feedback signal          |

#### **4. Return and Discount Factor**

Total expected return:
**Gt = Rt+1 + γRt+2 + γ²Rt+3 + ...**
where **γ (0 < γ < 1)** discounts future rewards.

#### **5. Environment Types**

* **Fully Observable** → Complete state known
* **Partially Observable** → Incomplete or noisy information
* **Deterministic** → Fixed next state for each action
* **Stochastic** → Randomness in transitions



### **Markov Decision Process (MDP)**

Formal framework for RL, defined by **(S, A, P, R, γ)**:

* **S:** States
* **A:** Actions
* **P(s′|s,a):** Transition probability
* **R(s,a):** Reward
* **γ:** Discount factor

**Markov Property:** Future depends only on the present, not past history.



### **RL Approaches**

| Type            | Description                      | Examples                           |
|  | -- | - |
| **Model-Based** | Uses or learns environment model | Dynamic Programming, Dyna-Q        |
| **Model-Free**  | Learns directly from experience  | Monte Carlo, TD, SARSA, Q-Learning |



### **Value Functions**

* **Vπ(s):** Expected return from state `s` under policy `π`
* **Qπ(s,a):** Expected return from state–action pair under `π`



### **Exploration vs Exploitation**

* **Exploration:** Trying new actions to gain information
* **Exploitation:** Choosing the best-known action
  **ε-Greedy Policy:**
  → With probability **ε**, explore
  → With probability **1−ε**, exploit



### **Temporal Difference (TD) Learning**

Blends Monte Carlo and Dynamic Programming ideas:
**V(s) ← V(s) + α [R + γV(s′) − V(s)]**



### **SARSA (On-Policy)**

Learns using the same policy it follows:
**Q(s,a) ← Q(s,a) + α [R + γQ(s′,a′) − Q(s,a)]**



### **Q-Learning (Off-Policy)**

Learns optimal policy independent of the agent’s behavior:
**Q(s,a) ← Q(s,a) + α [R + γ maxₐ Q(s′,a) − Q(s,a)]**



**In essence:**

* **SARSA** → Safer, more stable (learns what it does)
* **Q-Learning** → Faster, more optimal (learns what it *should* do)

### References

* [Reinforcement Learning INtro.](https://youtu.be/zdIQkjtFX_I?si=slQA3ODrVy3v_Rp9)
* [How do RL Agents Learn?](https://youtu.be/DLcBjo5gIxs?si=0M8ou1fs68wfHBWr)
* [How to Train your first RL Agent.](https://youtu.be/tbpBW5Yr44k?si=kI7AVOrtPD7VGacZ)
* [Gymnasium Documentation](https://gymnasium.farama.org/)


**Author - Pawankumar Navinchandra**
