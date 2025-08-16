#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

# 右手14个自由度对应的原始28维数组索引
RIGHT_HAND_ACTION_INDICES = [7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25, 26, 27]


def filter_right_hand_actions(batch: dict[str, Any]) -> dict[str, Any]:
    """
    过滤批次数据，只保留右手的14个自由度动作

    Args:
        batch: 包含动作数据的批次字典

    Returns:
        过滤后的批次字典，动作维度从28维变为14维
    """
    if "action" in batch and isinstance(batch["action"], torch.Tensor):
        # 提取右手的14个自由度
        batch["action"] = batch["action"][..., RIGHT_HAND_ACTION_INDICES]
        logging.debug(
            f"Filtered action from {batch['action'].shape[-1] + 14} to {batch['action'].shape[-1]} dimensions"
        )

    return batch


def filter_dataset_stats_for_right_hand(
    dataset_stats: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    过滤数据集统计信息，只保留右手的14个自由度

    Args:
        dataset_stats: 原始数据集统计信息

    Returns:
        过滤后的统计信息
    """
    import numpy as np

    filtered_stats = {}
    for key, value in dataset_stats.items():
        if key == "action" and isinstance(value, dict):
            filtered_action_stats = {}
            for stat_type, stat_value in value.items():
                if isinstance(stat_value, torch.Tensor) and stat_value.shape[-1] == 28:
                    # 提取右手的14个自由度统计信息
                    filtered_action_stats[stat_type] = stat_value[..., RIGHT_HAND_ACTION_INDICES]
                    logging.info(
                        f"Filtered {stat_type} stats from {stat_value.shape} to {filtered_action_stats[stat_type].shape}"
                    )
                elif isinstance(stat_value, np.ndarray) and len(stat_value) == 28:
                    # 提取右手的14个自由度统计信息
                    filtered_action_stats[stat_type] = stat_value[RIGHT_HAND_ACTION_INDICES]
                    logging.info(
                        f"Filtered {stat_type} stats from {stat_value.shape} to {filtered_action_stats[stat_type].shape}"
                    )
                elif isinstance(stat_value, list) and len(stat_value) == 28:
                    # 提取右手的14个自由度统计信息
                    filtered_action_stats[stat_type] = [stat_value[i] for i in RIGHT_HAND_ACTION_INDICES]
                    logging.info(
                        f"Filtered {stat_type} stats from length {len(stat_value)} to {len(filtered_action_stats[stat_type])}"
                    )
                else:
                    filtered_action_stats[stat_type] = stat_value
            filtered_stats[key] = filtered_action_stats
        else:
            filtered_stats[key] = value

    return filtered_stats


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    cfg.policy.use_lora = cfg.use_lora

    # 修改数据集元数据以支持14维动作输出
    if hasattr(dataset.meta, "features") and "action" in dataset.meta.features:
        # 创建一个新的元数据对象，手动设置所需的属性
        from copy import deepcopy

        modified_meta = type(dataset.meta)(
            repo_id=dataset.meta.repo_id, root=dataset.meta.root, revision=dataset.meta.revision
        )

        # 复制所有属性
        modified_meta.info = deepcopy(dataset.meta.info)
        modified_meta.tasks = deepcopy(dataset.meta.tasks)
        modified_meta.task_to_task_index = deepcopy(dataset.meta.task_to_task_index)
        modified_meta.episodes = deepcopy(dataset.meta.episodes)
        modified_meta.episodes_stats = deepcopy(dataset.meta.episodes_stats)

        # 修改info中的features
        modified_meta.info["features"] = deepcopy(dataset.meta.info["features"])
        modified_meta.info["features"]["action"] = dict(modified_meta.info["features"]["action"])
        modified_meta.info["features"]["action"]["shape"] = (14,)
        logging.info(f"Updated action feature shape to {modified_meta.info['features']['action']['shape']}")

        # 过滤数据集统计信息以匹配14维动作
        if hasattr(dataset, "stats") and dataset.stats:
            modified_meta.stats = filter_dataset_stats_for_right_hand(dataset.stats)
            logging.info("Filtered dataset statistics for right hand actions")
        else:
            modified_meta.stats = deepcopy(dataset.meta.stats)
            if "action" in modified_meta.stats:
                modified_meta.stats = filter_dataset_stats_for_right_hand(modified_meta.stats)
                logging.info("Filtered existing stats for right hand actions")
    else:
        modified_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=modified_meta,
    )

    logging.info("Creating optimizer and scheduler")
    if cfg.policy.use_lora:
        # Freeze non-LoRA parameters
        for name, param in policy.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # Collect LoRA parameters only
        lora_params = [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad]

        # Set optimizer and scheduler manually
        optimizer = torch.optim.Adam(lora_params, lr=cfg.optimizer.lr or 1e-4)
        lr_scheduler = None
    else:
        optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    logging.info(f"Training right hand only with {len(RIGHT_HAND_ACTION_INDICES)} degrees of freedom")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # 过滤批次数据，只保留右手的14个自由度动作
        batch = filter_right_hand_actions(batch)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
