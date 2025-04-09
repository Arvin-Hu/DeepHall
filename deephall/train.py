# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging  # 用于记录日志信息，如调试、信息、警告、错误等。
import signal  # 用于处理操作系统信号，如中断信号。SIGINT and SIGTERM
import sys  # 提供对与 Python 解释器紧密相关的变量和函数的访问。
import time  # 提供时间相关的函数，如 sleep, clock 等。

from argparse import ArgumentParser  # 用于解析命令行参数。
from typing import cast  # 用于类型转换，帮助静态类型检查器识别类型。

import chex  # 用于 JAX 的类型和形状检查，确保代码在多设备上正确运行。
import jax  # 用于高性能数值计算和自动微分。
import kfac_jax  # 提供 JAX 版本的 K-FAC（Kronecker-Factored Approximate Curvature）优化器。
import numpy as np  # 基础的科学计算库，用于处理数组、矩阵等。
from chex import PRNGKey  # 用于 JAX 中生成随机数的密钥。
from flax import linen as nn  # Flax 库的神经网络模块，用于构建模型。
from jax import numpy as jnp  # JAX 版本的 NumPy，用于 GPU 和 TPU 上的计算。
from omegaconf import OmegaConf  # 用于创建和管理配置文件，支持合并和覆盖配置。

from deephall import constants, mcmc, optimizers  # 导入本地模块，包括常量、MCMC 方法和优化器。
from deephall.config import Config, OptimizerName  # 配置类和优化器名称枚举。
from deephall.log import CheckpointState, LogManager, init_logging  # 日志管理和初始化。
from deephall.loss import LossMode, make_loss_fn  # 损失函数的模式和构造函数。
from deephall.networks import make_network  # 网络构造函数。
from deephall.types import LogPsiNetwork  # 类型定义，用于类型注解。


logger = logging.getLogger("deephall")  # 获取名为"deephall"的日志记录器


def init_guess(key: PRNGKey, batch: int, nelec: int):
    """Create uniform samples on the sphere.

    Args:
        key: random key.
        batch: number of samples to generate.
        nelec: number of electrons.

    Returns:
        Electron coordinates of shape [batch, nelec, 2]
    """
    key1, key2 = jax.random.split(key)  # 将随机密钥拆分为两个子密钥
    theta = jnp.arccos(jax.random.uniform(key1, (batch, nelec), minval=-1, maxval=1))  # 生成θ角，范围[-1,1]并取反余弦
    phi = jax.random.uniform(key2, (batch, nelec), minval=-jnp.pi, maxval=jnp.pi)  # 生成φ角，范围[-π,π]
    return jnp.stack([theta, phi], axis=-1)  # 将θ和φ堆叠成[batch, nelec, 2]的形状


def initialize_state(cfg: Config, model: nn.Module):
    key_data, key_params = jax.random.split(jax.random.PRNGKey(cfg.seed))  # 根据配置的种子生成两个随机密钥
    data = init_guess(key_data, cfg.batch_size, sum(cfg.system.nspins))  # 生成初始电子坐标
    data = data.reshape((jax.device_count(), -1, *data.shape[-2:]))  # 根据设备数量重塑数据形状
    params = kfac_jax.utils.replicate_all_local_devices(
        model.init(key_params, data[0, 0])  # 初始化模型参数并在所有设备上复制
    )
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(cfg.mcmc.width))  # 将MCMC步长复制到所有设备
    return 0, CheckpointState(params, data, None, mcmc_width)  # 返回初始状态


def setup_mcmc(cfg: Config, network: LogPsiNetwork):
    batch_network = jax.vmap(network, in_axes=(None, 0))  # 对网络进行批处理
    mcmc_step = mcmc.make_mcmc_step(
        batch_network,
        batch_per_device=cfg.batch_size // jax.device_count(),  # 计算每个设备的批量大小
        steps=cfg.mcmc.steps,  # 设置MCMC步数
    )
    pmap_mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)  # 对MCMC步骤进行并行化处理
    pmoves = np.zeros(cfg.mcmc.adapt_frequency)  # 初始化MCMC移动接受率数组
    return pmap_mcmc_step, pmoves  # 返回并行化的MCMC步骤和移动接受率数组


def train_loop(cfg: Config, log_manager: LogManager):
    model = make_network(cfg.system, cfg.network)  # 根据配置创建神经网络模型
    network = cast(LogPsiNetwork, model.apply)  # 将模型应用函数转换为LogPsiNetwork类型
    pmap_mcmc_step, pmoves = setup_mcmc(cfg, network)  # 设置MCMC步骤
    opt_init, training_step = optimizers.make_optimizer_step(cfg, network)  # 创建优化器初始化函数和训练步骤

    key = jax.random.PRNGKey(cfg.seed)  # 根据配置的种子生成随机密钥
    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)  # 在所有设备上生成不同的随机密钥
    initial_step, (params, data, opt_state, mcmc_width) = (
        log_manager.try_restore_checkpoint() or initialize_state(cfg, model)  # 尝试恢复检查点或初始化状态
    )

    if (
        cfg.optim.optimizer == OptimizerName.none
        and cfg.log.restore_path is not None
        and cfg.log.restore_path != cfg.log.save_path
    ):  # 如果优化器为none且恢复路径与保存路径不同
        initial_step = 0  # 重置步数为0

    if opt_state is None:  # 如果优化器状态为空
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)  # 拆分随机密钥
        opt_state = opt_init(params, subkey, data)  # 初始化优化器状态

    logger.info("Start VMC with %s JAX devices", jax.device_count())  # 记录启动VMC的JAX设备数量

    if initial_step == 0:  # 如果初始步数为0
        for _ in range(cfg.mcmc.burn_in):  # 进行MCMC预热
            sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
            data, pmove = pmap_mcmc_step(params, data, subkey, mcmc_width)
        logger.info("Burn in MCMC complete")  # 记录MCMC预热完成
        if cfg.log.initial_energy:  # 如果需要记录初始能量
            # Logging inital energy is helpful for debugging. If we have initial energy
            # but have error in training, it's probably optimizer's fault
            initial_stats, _ = constants.pmap(
                make_loss_fn(network, cfg.system, LossMode.ENERGY_DIFF)  # 计算初始能量
            )(params, data)
            logger.info("Initial energy: %s", initial_stats["energy"][0].real)  # 记录初始能量

    state = CheckpointState(params, data, opt_state, mcmc_width)  # 创建检查点状态

    for step in range(initial_step, cfg.optim.iterations):  # 进入训练循环
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        new_data, pmove = pmap_mcmc_step(
            state.params, state.data, subkey, state.mcmc_width  # 执行MCMC步骤
        )
        new_mcmc_width, pmoves = mcmc.update_mcmc_width(
            step - initial_step,
            state.mcmc_width,
            cfg.mcmc.adapt_frequency,
            pmove,
            pmoves,  # 更新MCMC步长
        )
        state = state._replace(data=new_data, mcmc_width=new_mcmc_width)  # 更新状态
        sharded_key, subkey = kfac_jax.utils.p_split(sharded_key)
        state, stats = training_step(state, subkey)  # 执行训练步骤
        yield step, state, stats, pmove  # 返回当前步数、状态、统计信息和MCMC移动接受率

def train(cfg: Config):
    init_logging()  # 初始化日志系统
    log_manager = LogManager(cfg)  # 创建日志管理器实例
    time_start = None  # 初始化计时器，排除JIT编译时间
    steps = 0  # 初始化步数计数器
    last_save_time = time.time()  # 记录上次保存时间
    killer = GracefulKiller()  # 创建优雅退出处理器
    with log_manager.create_writer() as writer:  # 创建日志写入器
        writer.hide("kinetic", "potential", "Lz_square")  # 隐藏部分日志字段
        for step, state, stats, pmove in train_loop(cfg, log_manager):  # 进入训练循环
            writer.log(  # 记录训练指标
                step=str(step),  # 当前训练步数
                pmove=f"{pmove[0]:.2f}",  # MCMC移动接受率
                energy=f"{stats['energy'].real[0]:.4f}",  # 能量实部
                energy_imag=f"{stats['energy'].imag[0]:+.4f}",  # 能量虚部
                potential=f"{stats['potential'][0]:.4f}",  # 势能
                kinetic=f"{stats['kinetic'].real[0]:.4f}",  # 动能
                variance=f"{stats['variance'][0]:.4f}",  # 方差
                Lz=f"{stats['angular_momentum_z'][0]:+.4f}",  # z方向角动量
                Lz_square=f"{stats['angular_momentum_z_square'][0]:.4f}",  # z方向角动量平方
                L_square=f"{stats['angular_momentum_square'][0]:.4f}",  # 总角动量平方
            )

            current_time = time.time()  # 获取当前时间
            if time_start is None:  # 如果是第一次迭代
                time_start = current_time  # 设置开始时间
            else:
                steps += 1  # 增加步数计数

            if (  # 检查是否需要保存检查点
                jnp.isnan(stats["energy"].real).any()  # 如果能量出现NaN
                or step == cfg.optim.iterations - 1  # 或者达到最大迭代次数
                or killer.kill_now  # 或者收到终止信号
                or (  # 或者满足保存时间间隔和步数间隔
                    current_time - last_save_time > cfg.log.save_time_interval
                    and (step + 1) % cfg.log.save_step_interval == 0
                )
            ):
                last_save_time = current_time  # 更新最后保存时间
                writer.force_flush()  # 强制刷新日志
                log_manager.save_checkpoint(step, state)  # 保存检查点
            if killer.kill_now or jnp.isnan(stats["energy"].real).any():  # 如果需要终止
                raise SystemExit("=" * 30 + " ABORT " + "=" * 30)  # 抛出终止异常
    if steps > 0 and time_start is not None:  # 如果完成了至少一步训练
        logger.info("Time per step: %.3fs", (current_time - time_start) / steps)  # 计算并记录每步平均时间


class GracefulKiller:
    """Capture SIGINT and SIGTERM so that we can save checkpoints before exit."""

    kill_now = False  # 标记是否需要立即终止

    def __init__(self):
        self.original_int = signal.signal(signal.SIGINT, self.exit_gracefully)  # 捕获SIGINT信号
        self.original_term = signal.signal(signal.SIGTERM, self.exit_gracefully)  # 捕获SIGTERM信号

    def exit_gracefully(self, signum, frame):
        """Mark as exit and restore signal handlers."""
        del signum, frame  # 删除未使用的参数
        if self.kill_now:  # 如果已经处理过信号
            return  # 直接返回
        print("\r", end="")  # 清除^C显示
        signal.signal(signal.SIGINT, self.original_int)  # 恢复原始SIGINT处理
        signal.signal(signal.SIGTERM, self.original_term)  # 恢复原始SIGTERM处理
        self.kill_now = True  # 标记需要终止


def cli(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(
        prog="deephall",
        description="Simulating the fractional quantum Hall effect (FQHE) with "
        "neural network variational Monte Carlo.",  # 命令行工具的描述
    )
    parser.add_argument(
        "dotlist", help="path.to.key=value pairs for configuration", nargs="*"  # 支持配置键值对
    )
    parser.add_argument("--yml", help="config YML file to merge")  # 支持YML配置文件
    parser.add_argument("--debug", help="disable JAX pmap", action="store_true")  # 调试模式，禁用JAX pmap
    args = parser.parse_args(argv or sys.argv[1:] or ["--help"])  # 解析命令行参数

    config = OmegaConf.structured(Config)  # 创建结构化配置的Config 实例
    if args.yml:
        config = OmegaConf.merge(config, OmegaConf.load(args.yml))  # 合并YML配置文件
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.dotlist))  # 合并命令行配置
    if args.debug:
        with chex.fake_pmap_and_jit():  # 在调试模式下禁用pmap和jit
            train(Config.from_dict(config))  # 启动训练
    else:
        train(Config.from_dict(config))  # 正常启动训练


if __name__ == "__main__": # 这是 Python 的标准入口点检查，确保代码只有在直接运行脚本时才会执行，而不是在作为模块导入时执行。
    cli()  # 主程序入口
