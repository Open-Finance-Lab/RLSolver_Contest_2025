"""
Ising Model starter kit – train on an instance

Dependencies
------------
pip install numpy matplotlib networkx torch
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass, asdict
from typing import Optional, List

# --------------------------------------------------------------------------- #
#  Config                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    graph_path: str = ""
    graph: nx = None
    n_nodes: int = 0
    n_edges: int = 0
    num_episodes: int = 200
    hidden_size: int = 64
    lr: float = 1e-2
    max_steps: Optional[int] = None
    save_logs: str = "./logs"
    seed: int = 42
    
    def __init__(self, G: str):
        self.graph_path = G
        f = open(G)
        self.n_nodes, self.n_edges = map(int, f.readline().split()[:2])
        self.graph = nx.Graph()
        for line in f.readlines():
            nodeA, nodeB = map(int, line.split()[:2])
            weight = float(line.split()[2])
            self.graph.add_edge(nodeA, nodeB, weight=weight)
            
# --------------------------------------------------------------------------- #
#  Environment                                                                #
# --------------------------------------------------------------------------- #
class IsingEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.graph = nx.to_numpy_array(cfg.graph, dtype=np.float32)

    def _cut(self) -> float:
        return float(np.sum((self.spins[:, None] != self.spins[None, :]) * self.graph) / 2)

    def _obs(self) -> np.ndarray:
        return self.spins.astype(np.float32)

    def reset(self, graph: Optional[np.ndarray] = None) -> np.ndarray:
        if graph is not None:
            self.graph = graph
        else:
            self.graph = nx.to_numpy_array(self.cfg.graph, dtype=np.float32)
        self.n = self.graph.shape[0]
        self.spins = np.random.choice([-1, 1], size=self.n)
        self.prev_cut = self._cut()
        self.step_count = 0
        self.max_steps = self.cfg.max_steps or self.n
        return self._obs()

    def step(self, action: int):
        self.spins[action] *= -1
        new_cut = self._cut()
        reward = self.prev_cut - new_cut 
        self.prev_cut = new_cut
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._obs(), reward, done, {"energy": new_cut}

# --------------------------------------------------------------------------- #
#  Policy                                                                     #
# --------------------------------------------------------------------------- #
class MLPPolicy(nn.Module):
    """
    Simple 2-layer MLP producing logits over actions.
    You could replace this policy with a GNN or any other neural architecture
    suited for graph inputs (e.g., GraphSAGE, GAT, or Transformer).
    """
    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)

# --------------------------------------------------------------------------- #
#  Train                                                                      #
# --------------------------------------------------------------------------- #
def train(cfg: Config):
    os.makedirs(cfg.save_logs, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = IsingEnv(cfg)
    policy = MLPPolicy(cfg.n_nodes, cfg.n_nodes, cfg.hidden_size)
    opt = optim.Adam(policy.parameters(), lr=cfg.lr)

    episode_cuts = []

    for ep in range(1, cfg.num_episodes + 1):
        obs = env.reset()
        log_probs, rewards = [], []
        done = False

        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            dist = policy(obs_t)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, done, info = env.step(action.item())
            rewards.append(reward)

        returns = torch.tensor([sum(rewards[i:]) for i in range(len(rewards))], dtype=torch.float32)
        log_probs_tensor = torch.stack(log_probs)
        loss = -(log_probs_tensor * returns).sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

        episode_cuts.append(info["energy"])
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep:4d} | Energy = {info['energy']}")

    plt.plot(episode_cuts)
    plt.xlabel("Episode")
    plt.ylabel("Final energy value")
    plt.title(f"Training on {cfg.graph_path} graphs (n={cfg.n_nodes})")
    plt.savefig(os.path.join(cfg.save_logs, "train_curve.png"))
    plt.close()

    torch.save(policy.state_dict(), os.path.join(cfg.save_logs, "policy.pth"))
    return policy

# --------------------------------------------------------------------------- #
#  Test                                                                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test(policy: MLPPolicy, cfg: Config):
    """
    Evaluate policy generalization on held-out graphs.
    This function can be extended to include other metrics,
    such as approximation ratio or gap to optimal.
    """
    policy.eval()

    env = IsingEnv(cfg)
    obs = env.reset()
    done = False
    energies = []
    while not done:
        dist = policy(torch.from_numpy(obs).float().unsqueeze(0))
        action = dist.sample().item()
        obs, _, done, info = env.step(action)
        print(f"Sampled Energy = {info['energy']}")
        energies.append(info['energy'])

    plt.hist(energies, bins=10)
    plt.title("Energy distribution on test set")
    plt.xlabel("Energy value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(cfg.save_logs, "test_hist.png"))
    plt.close()

# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    graph = "" # path to graph txt
    cfg = Config(graph)
    print("Config:", asdict(cfg))
    policy = train(cfg)
    print("\n--- Sampling from trained Ising model instance ---")
    test(policy, cfg)

if __name__ == "__main__":
    main()