"""
MaxCut starter kit – train on a graph distribution (same size, random topology)

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
    graph_type: str = "ER"      # 'ER', 'BA', or 'PL'
    n_nodes: int = 20
    p: float = 0.15             # ER edge probability / PL triangle prob
    m: int = 2                  # BA & PL: edges per new node
    num_episodes: int = 200
    hidden_size: int = 64
    lr: float = 1e-2
    max_steps: Optional[int] = None
    save_logs: str = "./logs"
    num_test_graphs: int = 50
    test_graphs_path: Optional[str] = None
    seed: int = 42

# --------------------------------------------------------------------------- #
#  Graph generator                                                            #
# --------------------------------------------------------------------------- #
class RandomGraphGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get(self) -> np.ndarray:
        t = self.cfg.graph_type.upper()
        n = self.cfg.n_nodes
        if t == "ER":
            g = nx.erdos_renyi_graph(n, self.cfg.p)
        elif t == "BA":
            g = nx.barabasi_albert_graph(n, self.cfg.m)
        elif t == "PL":
            g = nx.powerlaw_cluster_graph(n, self.cfg.m, self.cfg.p)
        else:
            raise ValueError(f"Unsupported graph type: {t}")
        return nx.to_numpy_array(g, dtype=np.float32)

# --------------------------------------------------------------------------- #
#  Environment                                                                #
# --------------------------------------------------------------------------- #
class MaxCutEnv:
    def __init__(self, generator: Optional[RandomGraphGenerator], cfg: Config):
        self.gen = generator
        self.cfg = cfg

    def _cut(self) -> int:
        return int(np.sum((self.spins[:, None] != self.spins[None, :]) * self.graph) // 2)

    def _obs(self) -> np.ndarray:
        return self.spins.astype(np.float32)

    def reset(self, graph: Optional[np.ndarray] = None) -> np.ndarray:
        if graph is not None:
            self.graph = graph
        else:
            self.graph = self.gen.get()
        self.n = self.graph.shape[0]
        self.spins = np.random.choice([-1, 1], size=self.n)
        self.prev_cut = self._cut()
        self.step_count = 0
        self.max_steps = self.cfg.max_steps or self.n
        return self._obs()

    def step(self, action: int):
        self.spins[action] *= -1
        new_cut = self._cut()
        reward = new_cut - self.prev_cut
        self.prev_cut = new_cut
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self._obs(), reward, done, {"cut": new_cut}

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

    gen = RandomGraphGenerator(cfg)
    env = MaxCutEnv(gen, cfg)
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

        episode_cuts.append(info["cut"])
        if ep % 10 == 0 or ep == 1:
            print(f"Episode {ep:4d} | Cut = {info['cut']}")

    plt.plot(episode_cuts)
    plt.xlabel("Episode")
    plt.ylabel("Final cut value")
    plt.title(f"Training on {cfg.graph_type} graphs (n={cfg.n_nodes})")
    plt.savefig(os.path.join(cfg.save_logs, "train_curve.png"))
    plt.close()

    torch.save(policy.state_dict(), os.path.join(cfg.save_logs, "policy.pth"))
    return policy

# --------------------------------------------------------------------------- #
#  Test                                                                       #
# --------------------------------------------------------------------------- #
def load_test_graphs(cfg: Config) -> List[np.ndarray]:
    if cfg.test_graphs_path and os.path.exists(cfg.test_graphs_path):
        with open(cfg.test_graphs_path, "rb") as f:
            return pickle.load(f)
    gen = RandomGraphGenerator(cfg)
    return [gen.get() for _ in range(cfg.num_test_graphs)]

@torch.no_grad()
def test(policy: MLPPolicy, cfg: Config):
    """
    Evaluate policy generalization on held-out graphs.
    This function can be extended to include other metrics,
    such as approximation ratio or gap to optimal.
    """
    policy.eval()
    test_graphs = load_test_graphs(cfg)
    cuts = []

    env = MaxCutEnv(None, cfg)
    for idx, adj in enumerate(test_graphs):
        obs = env.reset(graph=adj)
        done = False
        while not done:
            dist = policy(torch.from_numpy(obs).float().unsqueeze(0))
            action = dist.sample().item()
            obs, _, done, info = env.step(action)
        cuts.append(info["cut"])
        print(f"Test graph {idx+1:3d} | Cut = {info['cut']}")

    avg_cut = np.mean(cuts)
    print(f"\nAverage cut over {len(cuts)} test graphs: {avg_cut:.2f}")

    plt.hist(cuts, bins=10)
    plt.title("Cut distribution on test set")
    plt.xlabel("Cut value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(cfg.save_logs, "test_hist.png"))
    plt.close()

# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    cfg = Config()
    print("Config:", asdict(cfg))
    policy = train(cfg)
    print("\n--- Testing on held-out graphs ---")
    test(policy, cfg)

if __name__ == "__main__":
    main()