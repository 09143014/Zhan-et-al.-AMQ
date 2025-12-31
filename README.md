# Approximate Minimax-Q Learning (AMQ) — Simulation Code

This repository contains the simulation and reproduction code for the paper **"Approximate Minimax Q Learning for Adversarial Markov Games with Unbounded State Spaces"**. It implements **Approximate Minimax-Q (AMQ)** with linear function approximation and provides experiment scripts for adversarial data-flow control tasks (e.g., routing and polling), including evaluation utilities and baseline comparisons to reproduce the main results and figures.

## Code files and outputs

- `RL_attacker_polling_AMQ1.ipynb`: AMQ1 polling-queue attacker/defender simulation. Learns/loads linear AMQ1 weights `w_k`, simulates the queue system, records `rl_history`, and prints per-queue statistics plus the final weight vector.
- `RL_attacker_polling_AMQ2.ipynb`: AMQ2 version of the polling-queue attacker/defender simulation (quadratic feature set). Produces the same printed queue statistics and final weight vector.
- `RL_attacker_polling_nn.ipynb`: Neural-network Q approximation baseline for the polling system. Defines `PollingSystemWithNN`, trains the network, and reports reward/learning diagnostics.
- `random_experiments.ipynb`: Runs multiple random seeds for AMQ1/AMQ2/NNQ; writes learning curves to `outputs/AMQ1_*.out`, `outputs/AMQ2_*.out`, `outputs/NNQ_*.out`, and saves convergence plots to `figures/convergence_*.png`.
- `error_bar_plot.ipynb`: Loads `.out` curves from `outputs/` to generate error-bar comparison plots; saves images like `error_bar_10000.png` and `error_bar_1000_amq.png`, plus colored variants under `colored_error_bar/`.
- `updated_comparison_table.ipynb`: Aggregates runs and computes the comparison table and parameter-convergence summaries; displays tables/plots in the notebook (no file output by default).

## Suggested workflow

1. Run `RL_attacker_polling_AMQ1.ipynb` and `RL_attacker_polling_AMQ2.ipynb` to reproduce single-run AMQ logs and queue statistics.
2. Run `RL_attacker_polling_nn.ipynb` to generate the NNQ baseline behavior for the same polling setup.
3. Run `random_experiments.ipynb` to produce batched learning curves in `outputs/` and convergence figures in `figures/`.
4. Run `updated_comparison_table.ipynb` to compute the comparison table and convergence summaries.
5. Run `error_bar_plot.ipynb` to generate error-bar comparison plots (including colored versions in `colored_error_bar/`).


## Simulation notes (from logs)

## AMQ1

### AMQ1 feature approximators (paper definition)

The approximators for AMQ are as follows: for i = 1, 2, ..., m,

$$
\begin{aligned}
\phi_{i,1}(x,a,b) &= 1 \\
\phi_{i,2}(x,a,b) &= x_i + \delta_i(x,a,b) \\
\phi_{i,3}(x,a,b) &= a \\
\phi_{i,4}(x,a,b) &= b
\end{aligned}
$$


#### 3 queues
```text
    queue_nums = 3
    arrival_rates = [3.0, 3.5, 4.0]
    ini_jobs_list = [0, 20, 18, 10]
    service_rate = 25.0
    switch_time = 2.0
    simulation_time = 10000.0
Simulation finished at time: 10000.015540388787
Total completed jobs: 108137
Queue 0:
  Completed jobs: 30127
  Average wait time: 0.8545863947785566
  Average queue length: 2.177948997203943
Queue 1:
  Completed jobs: 36517
  Average wait time: 0.9180885656087862
  Average queue length: 2.9294541286705273
Queue 2:
  Completed jobs: 41493
  Average wait time: 0.9411366538439391
  Average queue length: 3.480132760978805
Server utilization: 0.8720733164172011

Final w vector:
[-1.20697136  0.26897754 -2.86369133  2.02787411 -1.20697136  0.09243459
 -2.86369133  2.02787411 -1.20697136  0.11301513 -2.86369133  2.02787411]
```

#### 6 queues
```text
queue_nums = 6
    arrival_rates = [3.0, 3.5, 4.0, 2.0, 2.5, 5]
    ini_jobs_list = [0, 20, 18, 10, 15, 5, 25]
    service_rate = 25.0
    switch_time = 2.0
    simulation_time = 10000.0
Simulation finished at time: 10001.463450329908
Total completed jobs: 203362
Queue 0:
  Completed jobs: 30246
  Average wait time: 1.0208702537824879
  Average queue length: 2.754532695210797
Queue 1:
  Completed jobs: 35353
  Average wait time: 1.0113353435763797
  Average queue length: 3.241185382271733
Queue 2:
  Completed jobs: 40801
  Average wait time: 0.9961073121198133
  Average queue length: 3.7387141945798614
Queue 3:
  Completed jobs: 20357
  Average wait time: 1.2631534379686455
  Average queue length: 2.2114972711715977
Queue 4:
  Completed jobs: 25521
  Average wait time: 1.2291172133520762
  Average queue length: 2.7929190118558767
Queue 5:
  Completed jobs: 51084
  Average wait time: 0.9985418147180056
  Average queue length: 4.78281690326802
Server utilization: 0.988681038290318
Final w vector:
[-0.83742967  0.08591887 -1.39645405  1.02146057 -0.83742967  0.03746543
 -1.39645405  1.02146057 -0.83742967 -0.09514415 -1.39645405  1.02146057
 -0.83742967  0.46234868 -1.39645405  1.02146057 -0.83742967  0.3682981
 -1.39645405  1.02146057 -0.83742967 -0.08662211 -1.39645405  1.02146057]
```

## AMQ2

### AMQ2 feature approximators (paper definition)

The approximators for AMQ are as follows: for i = 1, 2, ..., m,

$$
\begin{aligned}
\phi_{i,1}(x, a, b) &= 1\\
\phi_{i,2}(x, a, b) &= x_i + \delta_i(x,a,b)\\
\phi_{i,3}(x, a, b) &= x_i + \delta_i^2(x,a,b)\\
\phi_{i,4}(x, a, b) &= a\\
\phi_{i,5}(x, a, b) &= b\\
\end{aligned}
$$
```

#### 3 queues
```text
    queue_nums = 3
    arrival_rates = [3.0, 3.5, 4.0]
    ini_jobs_list = [0, 20, 18, 10]
    service_rate = 25.0
    switch_time = 2.0
    simulation_time = 10000.0
Simulation finished at time: 10000.028815619833
Total completed jobs: 108155
Queue 0:
  Completed jobs: 30097
  Average wait time: 0.8615977026072765
  Average queue length: 2.204789546739805
Queue 1:
  Completed jobs: 36494
  Average wait time: 0.9152887287706667
  Average queue length: 2.9159537194899166
Queue 2:
  Completed jobs: 41564
  Average wait time: 0.9400367645897536
  Average queue length: 3.477951976831992
Server utilization: 0.8750924207405582

Final w vector:
[-1.98728814  0.143287   -0.0684252  -2.80184743  2.00050136 -1.98728814
  0.89757994 -0.59426948 -2.80184743  2.00050136 -1.98728814  1.11243839
 -0.73895207 -2.80184743  2.00050136]
```

#### 6 queues
```text
queue_nums = 6
    arrival_rates = [3.0, 3.5, 4.0, 2.0, 2.5, 5]
    ini_jobs_list = [0, 20, 18, 10, 15, 5, 25]
    service_rate = 25.0
    switch_time = 2.0
    simulation_time = 10000.0
Simulation finished at time: 10000.171983715814
Total completed jobs: 202739
Queue 0:
  Completed jobs: 29871
  Average wait time: 1.0108764038610485
  Average queue length: 2.695358302328422
Queue 1:
  Completed jobs: 35558
  Average wait time: 1.0070770218272878
  Average queue length: 3.2496751949491904
Queue 2:
  Completed jobs: 40480
  Average wait time: 0.991609838778533
  Average queue length: 3.689655435343337
Queue 3:
  Completed jobs: 20436
  Average wait time: 1.2645338981862295
  Average queue length: 2.214358209567097
Queue 4:
  Completed jobs: 25768
  Average wait time: 1.2383790772638081
  Average queue length: 2.8377027880739574
Queue 5:
  Completed jobs: 50626
  Average wait time: 0.9998131037497232
  Average queue length: 4.743948754349432
Server utilization: 0.9870413241734618

Final w vector:
[-1.08400114 -0.28302239  0.12548904 -1.39547459  1.00754207 -1.08400114
  0.16151146 -0.16890023 -1.39547459  1.00754207 -1.08400114  0.30012919
 -0.2748696  -1.39547459  1.00754207 -1.08400114  0.72891291 -0.52062058
 -1.39547459  1.00754207 -1.08400114  0.75343755 -0.59918131 -1.39547459
  1.00754207 -1.08400114  0.69128533 -0.41764334 -1.39547459  1.00754207]
```

## Shared AMQ hyperparameters

- `gamma = 0.9`
- `eta0 = 0.01`
- `cost_attack = 8.0`
- `cost_defend = 6.0`
- `switch_cost_val = 1.0`

## NNQ baseline architecture

- 输入层：维度 5 * queue_nums（每个队列 5 个特征）
- 隐藏层 1：全连接 128，ReLU
- 隐藏层 2：全连接 64，ReLU
- 输出层：全连接 1，线性输出 Q 值
