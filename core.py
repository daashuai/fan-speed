import numpy as np
import math
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.n
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()

class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于向输入序列添加位置信息。
    """
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播逻辑。
        - x: 输入张量 (seq_len, batch_size, embed_dim)
        """
        # x:(7,100,32)
        # pe(1,5000,32)
        x = x + self.pe[:, :x.size(0), :]
        return x

class TransformerBlock(nn.Module):
    """
    单层标准 Transformer 块（无门控机制）。
    """
    # (6,256,256,2)
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)


    def forward(self, x, memory=None):
        """
        前向传播逻辑。
        - x: 当前时间步的输入 (seq_len, batch_size, embed_dim)。
        - memory: 可选的跨时间步记忆 (seq_len_mem, batch_size, embed_dim)。
        """
        if memory is not None:
            # 将历史记忆拼接到输入
            x = torch.cat([memory, x], dim=0)

        # 自注意力机制
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + attn_output)  # 残差连接

        # 前馈网络
        ff_output = self.feedforward(x)
        x = self.layernorm2(x + ff_output)  # 残差连接

        # x = self.feedforward(x)
        return x

class Transformer(nn.Module):
    """
    多层标准 Transformer，用于 Actor 和 Critic 的骨干网络。
    """
    def __init__(self, obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(obs_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(500, embed_dim))
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory=None):
        """
        前向传播逻辑。
        - x.permute: 当前时间步的输入 (seq_len, batch_size, embed_dim)。
        - memory: 跨时间步的记忆。
        """
        x = x.permute(1, 0, 2)
        seq_len, batch_size, _ = x.size()
        x = self.embedding(x)
        pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(1).repeat(1, batch_size, 1)
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x, memory)
        x = x.permute(1, 0, 2)
        return x

class ITrXLBlock(nn.Module):
    """
    Identity TrXL。参考论文"Stabilizing Transformers for Reinforcement Learning"
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )


    def forward(self, x, memory=None):
        """
        前向传播逻辑。
        - x: 当前时间步的输入 (seq_len, batch_size, embed_dim)。
        - memory: 可选的跨时间步记忆 (seq_len_mem, batch_size, embed_dim)。
        """
        # 拼接历史记忆
        if memory is not None:
            x = torch.cat([memory, x], dim=0)

        # 自注意力模块
        x_norm = self.layernorm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_output  # 残差连接

        # 前馈模块
        y_norm = self.layernorm2(x)
        ff_output = self.feedforward(y_norm)
        x = x + ff_output  # 残差连接
        

        return x

# ===== ITrXL ===== #
class ITrXL(nn.Module):
    """
    没有门控机制的 GTrXL，多层堆叠，重命名为 ITrXL。
    """
    def __init__(self, obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(obs_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(500, embed_dim))
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory=None):
        """
        前向传播逻辑。
        - x.permute: 当前时间步的输入 (seq_len, batch_size, embed_dim)。
        - memory: 跨时间步的记忆。
        """
        x = x.permute(1, 0, 2)
        seq_len, batch_size, _ = x.size()
        x = self.embedding(x)
        pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(1).repeat(1, batch_size, 1)
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x, memory)
        x = x.permute(1, 0, 2)

        return x

class GTrXLBlock(nn.Module):
    """
    单层带门控的 Transformer 块。
    GatedTransformerBlock(nn.Module):
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        # 门控机制
        self.gate1 = nn.Parameter(torch.ones(embed_dim))
        self.gate2 = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x, memory=None):
        """
        前向传播逻辑。
        - x: 当前时间步的输入 (seq_len, batch_size, embed_dim)。
        - memory: 可选的跨时间步记忆 (seq_len_mem, batch_size, embed_dim)。
        """
        # 拼接历史记忆
        if memory is not None:
            x = torch.cat([memory, x], dim=0)

        # 自注意力模块
        x_norm = self.layernorm1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.gate1 * attn_output  # 残差连接 + 门控

        # 前馈模块
        y_norm = self.layernorm2(x)
        ff_output = self.feedforward(y_norm)
        x = x + self.gate2 * ff_output  # 残差连接 + 门控

        return x


class GTrXL(nn.Module):
    """
    多层门控 Transformer，用于 Actor 和 Critic 的骨干网络。
    参考论文"Stabilizing Transformers for Reinforcement Learning"
    没有门控机制的 GTrXL，多层堆叠，重命名为 ITrXL。
    """
    def __init__(self, obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(obs_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(500, embed_dim))
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory=None):
        """
        前向传播逻辑。
        - x.permute: 当前时间步的输入 (seq_len, batch_size, embed_dim)。
        - memory: 跨时间步的记忆。
        """
        x = x.permute(1, 0, 2)
        seq_len, batch_size, _ = x.size()
        x = self.embedding(x)
        pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(1).repeat(1, batch_size, 1)
        x = x + pos_emb
        for layer in self.layers:
            x = layer(x, memory)
        x = x.permute(1, 0, 2)

        return x




class SquashedGaussianTransformerActor(nn.Module):
    def __init__(self, model_name, obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout, act_limit):
        super().__init__()
        if model_name == "trans":
            self.transformer = Transformer(obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        elif model_name == "itr":
            self.transformer = ITrXL(obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        elif model_name == "gtr":
            self.transformer = GTrXL(obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)

        self.mu_layer = nn.Linear(embed_dim, act_dim)
        self.log_std_layer = nn.Linear(embed_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        使用 Transformer 的 Actor 网络。
        - obs: 输入观测 (batch_size, obs_dim)
        """

        # obs:(batch_size, token_num) -> (100, 6)
        # obs:(batch_size, token_num, token_dim) -> (100, 6, 1)
        # embedded:(batch_size, token_num, token_dim) -> (100, 6, 256)

        obs = obs.unsqueeze(-1)
        transformer_out = self.transformer(obs)  # (batch_size, token_num, embed_dim)
        # transformer_out = transformer_out.squeeze(0)  # 去掉token维度，变为 (batch_size, embed_dim)
        # 第一个token
        # first_token_output = transformer_out[:, 0]  # (batch_size, embed_dim)
        # Pool Mean
        pooled_output_mean = transformer_out.mean(dim=1)  # (batch_size, embed_dim)
        # Pool Max 
        # pooled_output = transformer_out.max(dim=1).values  # (batch_size, embed_dim)

        # 输出动作分布参数
        mu = self.mu_layer(pooled_output_mean)
        log_std = self.log_std_layer(pooled_output_mean)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # 构造分布并采样动作
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        # pi_action = pi_action.squeeze(-1)
        pi_action = torch.tanh(pi_action)  # 使用 Tanh 归一化动作
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class TransformerQFunction(nn.Module):
    def __init__(self, model_name, obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout):
        super().__init__()
        # self.embedding = nn.Linear(obs_dim + act_dim, embed_dim)
        if model_name == "trans":
            self.transformer = Transformer(obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        elif model_name == "itr":
            self.transformer = ITrXL(obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        elif model_name == "gtr":
            self.transformer = GTrXL(obs_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        self.q_layer = nn.Linear(embed_dim, 1)

    def forward(self, obs, act):
        """
        使用 Transformer 的 Q 网络。
        - obs: 输入观测 (batch_size, obs_dim)
        - act: 动作输入 (batch_size, act_dim)
        """
        # 拼接观测和动作，并添加序列维度
        x = torch.cat([obs, act], dim=-1).unsqueeze(-1)  # (batch_size, obs_dim + act_dim, -1)
        transformer_out = self.transformer(x)  # (1, batch_size, embed_dim)
        transformer_out = transformer_out.squeeze(0)  # (batch_size, embed_dim)
        pooled_output_mean = transformer_out.mean(dim=1)  # (batch_size, embed_dim)

        q = self.q_layer(pooled_output_mean)  # 输出 Q 值 (batch_size, 1)
        return torch.squeeze(q, -1)


class TransformerActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, embed_dim=32, num_heads=4, feedforward_dim=32, num_layers=2, dropout=0.1):
        super().__init__()

        # obs_dim = observation_space.n
        obs_dim = 1
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # 构建 Actor 和 Critic
        self.pi = SquashedGaussianTransformerActor("trans", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout, act_limit)
        self.q1 = TransformerQFunction("trans", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        self.q2 = TransformerQFunction("trans", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs = obs.unsqueeze(0)
            a, _ = self.pi(obs, deterministic, False)
            b = a.numpy()
            return a.numpy()

class ITrXLActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, embed_dim=32, num_heads=4, feedforward_dim=32, num_layers=2, dropout=0.1):
        super().__init__()

        # obs_dim = observation_space.n
        obs_dim = 1
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # 构建 Actor 和 Critic
        self.pi = SquashedGaussianTransformerActor("itr", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout, act_limit)
        self.q1 = TransformerQFunction("itr", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        self.q2 = TransformerQFunction("itr", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs = obs.unsqueeze(0)
            a, _ = self.pi(obs, deterministic, False)
            b = a.numpy()
            return a.numpy()

class GTrXLActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, embed_dim=32, num_heads=4, feedforward_dim=32, num_layers=2, dropout=0.1):
        super().__init__()

        # obs_dim = observation_space.n
        obs_dim = 1
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # 构建 Actor 和 Critic
        self.pi = SquashedGaussianTransformerActor("gtr", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout, act_limit)
        self.q1 = TransformerQFunction("gtr", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)
        self.q2 = TransformerQFunction("gtr", obs_dim, act_dim, embed_dim, num_heads, feedforward_dim, num_layers, dropout)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            obs = obs.unsqueeze(0)
            a, _ = self.pi(obs, deterministic, False)
            b = a.numpy()
            return a.numpy()




