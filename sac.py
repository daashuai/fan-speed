from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core as core
from cooling import CoolingEnv
# from logx import EpochLogger
from utils import plot_speed_temp,calculate_energy,calculate_speed_smoothness,calculate_speed_deviation,calculate_temp_deviation,calculate_max_change,calculate_temp_stabilization_time
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from datetime import datetime

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}



def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=5, max_trajectory_len=1000, save_freq=1, eval_interval=10):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_trajectory_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join('exp', time_stamp)
    os.makedirs(experiment_dir, exist_ok=True)

    writer = SummaryWriter(experiment_dir)



    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.n
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]


    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)


    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        #               Q2Vals=q2.detach().numpy())
        q_info = 0

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.detach().numpy())
        pi_info = 0

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        o_tensor = torch.as_tensor(o, dtype=torch.float32).to(device)
        return ac.act(o_tensor, deterministic)


    epoch = 0
    best_score = -np.inf
    best_epoch = -1
    best_model_weights = None

    def test_agent(_epoch):
        test_returns = []
        test_energy = []
        test_speed_smooth = []
        test_temp_deviation = []
        test_max_speed_change = []
        test_stablize_time = []
        
        epoch_start_time = time.time()

        for _ in range(num_test_episodes):
            o, d, trajectory_ret, trajectory_len = test_env.reset(), False, 0, 0
            while not(d or (trajectory_len == max_trajectory_len)):
                # Take deterministic actions at test time 
                # a = get_action(o, True).detach().cpu().numpy()
                o, r, d, _ = test_env.step(get_action(o, True).detach().cpu().numpy())
                trajectory_ret += r
                trajectory_len += 1

            energy = calculate_energy(test_env.speeds)
            speed_smooth = calculate_speed_smoothness(test_env.speeds)
            temp_deviation = calculate_temp_deviation(test_env.temps)
            max_speed_change = calculate_max_change(test_env.speeds)
            stablize_time = calculate_temp_stabilization_time(test_env.temps, test_env.temp_target)

            test_returns.append(trajectory_ret)
            test_energy.append(energy)
            test_speed_smooth.append(speed_smooth)
            test_temp_deviation.append(temp_deviation)
            test_max_speed_change.append(max_speed_change)
            test_stablize_time.append(stablize_time)

            # logger.store(TestEpRet=trajectory_ret, TestEpLen=trajectory_len)
        test_epoch_time = time.time() - epoch_start_time
        avg_ret = np.mean(test_returns)
        avg_energy = np.mean(test_energy)
        avg_speed_smooth = np.mean(test_speed_smooth)
        avg_temp_deviation = np.mean(test_temp_deviation)
        max_max_speed_change = np.max(test_max_speed_change)
        avg_stablize_time = np.mean(test_stablize_time)
        writer.add_scalar("Test/AverageReturn", avg_ret, _epoch)
        writer.add_scalar("Test/EnergyConsume", avg_energy, _epoch)
        writer.add_scalar("Test/SpeedSmooth", avg_speed_smooth, _epoch)
        writer.add_scalar("Test/TempDeviation", avg_temp_deviation, _epoch)
        writer.add_scalar("Test/MaxSpeedChange", max_max_speed_change, _epoch)
        writer.add_scalar("Test/TempStablizeTime", avg_stablize_time, _epoch)
        writer.add_scalar("Test/EpochTime", test_epoch_time, _epoch)
        return avg_ret

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, trajectory_ret, trajectory_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o).detach().cpu().numpy()
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        trajectory_ret += r
        trajectory_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if trajectory_len==max_trajectory_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        
        # if (t+1)%steps_per_epoch == 0:
        #     epoch = (t+1)//steps_per_epoch
        #     if epoch == 10 or epoch ==20 or epoch ==30 or epoch == 40 or epoch == 50:
        #         env.plot(env.speeds, env.temps)
        
        # if (t+1)%steps_per_epoch == 0:
        #     env.plot(env.speeds, env.temps)
        
        if (t+1)%steps_per_epoch == 0:
            plot_speed_temp(writer, epoch, env.speeds, env.temps)

        # End of trajectory handling
        if d or (trajectory_len == max_trajectory_len):
            epoch_time = time.time() - start_time
            # logger.store(EpRet=trajectory_ret, EpLen=trajectory_len)
            writer.add_scalar("Train/EpochTime", epoch_time, epoch)
            writer.add_scalar("Train/TotalReturn", trajectory_ret, epoch)
            print("Epoch:" + str(epoch) + " :" + " TotalReturn :" + str(trajectory_ret) + "\n")
            with open(os.path.join(experiment_dir, "log.txt"), "a+") as log_file:
                print("Epoch:" + str(epoch) + " :" + " TotalReturn :" + str(trajectory_ret) + "\n", file=log_file)
                print("Epoch:" + str(epoch) + " :" + " EpochTime :" + str(epoch_time) + "\n", file=log_file)

            energy = calculate_energy(env.speeds)
            speed_smooth = calculate_speed_smoothness(env.speeds)
            temp_deviation = calculate_temp_deviation(env.temps)

            writer.add_scalar("Train/EnergyConsume", energy, epoch)
            writer.add_scalar("Train/SpeedSmooth", speed_smooth, epoch)
            writer.add_scalar("Train/TempDeviation", temp_deviation, epoch)

            o, trajectory_ret, trajectory_len = env.reset(), 0, 0
            start_time = time.time()

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
            # for name, param in ac.named_parameters():
            #     if 'fc1.weight' in name:  # 记录第一层权重
            #         writer.add_histogram(name, param, global_step=epoch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

        # 每25个epoch执行完整评估
        if t % (steps_per_epoch*eval_interval) == 0 or epoch == epochs:
            current_score = test_agent(epoch)
            print(f"Epoch {epoch} | Test Score: {current_score:.2f}")
            with open(os.path.join(experiment_dir, "log.txt"), "a+") as log_file:
                print(f"Epoch {epoch} | Test Score: {current_score:.2f}", file=log_file)
        
        # 保存最佳模型
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_model_weights = deepcopy(ac.state_dict())
            torch.save(best_model_weights, 
                      os.path.join(experiment_dir, f'best_model.pt'))
            print(f"New best model saved at epoch {epoch} with score {current_score:.2f}")
            with open(os.path.join(experiment_dir, "log.txt"), "a+") as log_file:
                print(f"New best model saved at epoch {epoch} with score {current_score:.2f}", file=log_file)


    # 在训练结束后添加最终评估
    if best_model_weights is not None:
        ac.load_state_dict(best_model_weights)
        final_score = test_agent(epochs + 1)
        print(f"Final evaluation score: {final_score:.2f} at Epoch: {best_epoch}")
        with open(os.path.join(experiment_dir, "log.txt"), "a+") as log_file:
            print(f"Final evaluation score: {final_score:.2f} at Epoch: {best_epoch}", file=log_file)
        torch.save(ac.state_dict(), 
                  os.path.join(experiment_dir, 'final_model.pt'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--obs_dim', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='mlp')
    parser.add_argument('--workload_mode', type=str, default='medium')

    args = parser.parse_args()

    # from run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    def make():
        env = CoolingEnv(obs_dim=args.obs_dim, workload_mode=args.workload_mode)
        return env
    
    if args.model_name == "mlp":
        ac = core.MLPActorCritic
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)
    elif args.model_name == "trans":
        ac = core.TransformerActorCritic
        ac_kwargs = dict()
    elif args.model_name == "itr":
        ac = core.ITrXLActorCritic
        ac_kwargs = dict()
    elif args.model_name == "gtr":
        ac = core.GTrXLActorCritic
        ac_kwargs = dict()


    sac(lambda : make(), actor_critic=ac, ac_kwargs=ac_kwargs, 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs)
