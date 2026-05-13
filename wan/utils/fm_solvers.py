# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
# Convert dpm solver for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import inspect
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (KarrasDiffusionSchedulers,
                                                   SchedulerMixin,
                                                   SchedulerOutput)
from diffusers.utils import deprecate, is_scipy_available
from diffusers.utils.torch_utils import randn_tensor

if is_scipy_available():
    pass


def _safe_np_to_tensor(arr, dtype=None):
    # 将 numpy / tensor 输入统一转换为 torch.Tensor。
    # 物理含义：这里的 arr 往往是噪声强度 sigma 或时间步等离散采样表，
    # 这些量在扩散/流匹配采样中决定每一步去噪幅度与状态更新尺度。
    # dtype: 目标数据类型；若不传则默认转为 float32（推理中常用精度）。
    if isinstance(arr, torch.Tensor):
        return arr if dtype is None else arr.to(dtype)
    # 保证内存连续，避免 frombuffer/view 类操作出现跨步读取问题。
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.float32:
        # 直接按底层字节构造，减少额外拷贝。
        t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32)
    elif arr.dtype == np.float64:
        t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float64)
    elif arr.dtype in (np.int32, np.uint32):
        t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.int32 if arr.dtype == np.int32 else torch.uint32)
    elif arr.dtype in (np.int64, np.uint64):
        t = torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.int64 if arr.dtype == np.int64 else torch.uint64)
    else:
        t = torch.tensor(arr)
    # reshape 回原始形状，并统一 dtype。
    t = t.reshape(arr.shape).to(torch.float32 if dtype is None else dtype)
    return t


def get_sampling_sigmas(sampling_steps, shift):
    # 线性生成 [1, 0) 的 sigma 序列（长度=采样步数）。
    # sigma 可理解为“噪声占比”，sigma 越大表示样本越噪，越小表示越接近数据流形。
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    # shift 重参数化：非线性拉伸 sigma 曲线，改变“前期/后期”去噪步长分配。
    # shift>1 通常会让前段保留更多噪声、后段更细化。
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    # 统一调度器时间步入口：
    # - timesteps: 直接给离散时间索引；
    # - sigmas: 直接给噪声日程；
    # - num_inference_steps: 仅给步数，由调度器自己生成时间表。
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `FlowDPMSolverMultistepScheduler` is a fast dedicated high-order solver for diffusion ODEs.
    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.
    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model. This determines the resolution of the diffusion process.
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `1`, `2`, or `3`. It is recommended to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling. This affects the number of model outputs stored
            and used in multistep updates.
        prediction_type (`str`, defaults to "flow_prediction"):
            Prediction type of the scheduler function; must be `flow_prediction` for this scheduler, which predicts
            the flow of the diffusion process.
        shift (`float`, *optional*, defaults to 1.0):
            A factor used to adjust the sigmas in the noise schedule. It modifies the step sizes during the sampling
            process.
        use_dynamic_shifting (`bool`, defaults to `False`):
            Whether to apply dynamic shifting to the timesteps based on image resolution. If `True`, the shifting is
            applied on the fly.
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This method adjusts the predicted sample to prevent
            saturation and improve photorealism.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and
            `algorithm_type="dpmsolver++"`.
        algorithm_type (`str`, defaults to `dpmsolver++`):
            Algorithm type for the solver; can be `dpmsolver`, `dpmsolver++`, `sde-dpmsolver` or `sde-dpmsolver++`. The
            `dpmsolver` type implements the algorithms in the [DPMSolver](https://huggingface.co/papers/2206.00927)
            paper, and the `dpmsolver++` type implements the algorithms in the
            [DPMSolver++](https://huggingface.co/papers/2211.01095) paper. It is recommended to use `dpmsolver++` or
            `sde-dpmsolver++` with `solver_order=2` for guided sampling like in Stable Diffusion.
        solver_type (`str`, defaults to `midpoint`):
            Solver type for the second-order solver; can be `midpoint` or `heun`. The solver type slightly affects the
            sample quality, especially for a small number of steps. It is recommended to use `midpoint` solvers.
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        euler_at_final (`bool`, defaults to `False`):
            Whether to use Euler's method in the final step. It is a trade-off between numerical stability and detail
            richness. This can stabilize the sampling of the SDE variant of DPMSolver for small number of inference
            steps, but sometimes may result in blurring.
        final_sigmas_type (`str`, *optional*, defaults to "zero"):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
        lambda_min_clipped (`float`, defaults to `-inf`):
            Clipping threshold for the minimum value of `lambda(t)` for numerical stability. This is critical for the
            cosine (`squaredcos_cap_v2`) noise schedule.
        variance_type (`str`, *optional*):
            Set to "learned" or "learned_range" for diffusion models that predict variance. If set, the model's output
            contains the predicted Gaussian variance.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting=False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        invert_sigmas: bool = False,
    ):
        # num_train_timesteps: 训练时离散时间长度 T。
        # solver_order: 多步求解阶数（1/2/3 阶），高阶通常更少步数下更准。
        # prediction_type: Flow Matching 下应为 flow_prediction（预测“流/速度”）。
        # shift/use_dynamic_shifting: 控制 sigma 时间表形状（固定或动态）。
        # algorithm_type: 选择 ODE/SDE 变体求解公式。
        # solver_type: 二阶时 midpoint 或 heun，不同数值稳定性/误差常数。
        # lower_order_final/euler_at_final: 小步数时末段稳定性技巧。
        # final_sigmas_type: 最后一个 sigma 是 0（理想去噪终点）还是 sigma_min。
        # variance_type: 某些模型会同时预测方差。
        if algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            deprecation_message = f"algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead"
            deprecate("algorithm_types dpmsolver and sde-dpmsolver", "1.0.0",
                      deprecation_message)

        # settings for DPM-Solver
        if algorithm_type not in [
                "dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"
        ]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(
                    f"{algorithm_type} is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}")

        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"
                                 ] and final_sigmas_type == "zero":
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead."
            )

        # setable values
        self.num_inference_steps = None
        # alpha 在本实现中从 1 递减到 1/T，再反转后得到从小到大的“训练离散刻度”。
        # 这里并非 DDPM 的 alpha_cumprod，而是 flow-matching 版本中简化映射变量。
        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        # sigma = 1 - alpha，可理解为噪声分量权重。
        sigmas = 1.0 - alphas
        sigmas = _safe_np_to_tensor(sigmas, dtype=torch.float32)

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 +
                                       (shift - 1) * sigmas)  # pyright: ignore

        self.sigmas = sigmas
        # timesteps 与 sigma 线性映射到 [0, T]。
        self.timesteps = sigmas * num_train_timesteps

        # model_outputs: 多步法历史缓存，保存最近 k 次网络输出用于高阶差分。
        self.model_outputs = [None] * solver_order
        # lower_order_nums: 前几步历史不足时，自动退化为低阶更新计数器。
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

        # self.sigmas = self.sigmas.to(
        #     "cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.
        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[Union[float, None]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        # use_dynamic_shifting=True 时，mu 控制分辨率相关的时间重标定强度。
        # 可理解为根据图像/视频尺度调整“时间流速”。
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            # 默认从 sigma_max -> sigma_min 做均匀离散（去噪主路径）。
            sigmas = np.linspace(self.sigma_max, self.sigma_min,
                                 num_inference_steps +
                                 1).copy()[:-1]  # pyright: ignore

        if self.config.use_dynamic_shifting:
            # 动态位移：根据 mu 进行非线性时间变换。
            sigmas = self.time_shift(mu, 1.0, sigmas)  # pyright: ignore
        else:
            if shift is None:
                shift = self.config.shift
            # 固定位移：通过有理函数重映射 sigma 日程。
            sigmas = shift * sigmas / (1 +
                                       (shift - 1) * sigmas)  # pyright: ignore

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) /
                          self.alphas_cumprod[0])**0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        # 末尾拼接 sigma_last，便于 step() 中访问 step_index+1。
        sigmas = np.concatenate([sigmas, [sigma_last]
                                ]).astype(np.float32)  # pyright: ignore

        self.sigmas = _safe_np_to_tensor(sigmas)
        self.timesteps = _safe_np_to_tensor(timesteps).to(
            device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        self._step_index = None
        self._begin_index = None
        # self.sigmas = self.sigmas.to(
        #     "cpu")  # to avoid too much CPU/GPU communication

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        # batch_size: 批大小；channels: 通道数；remaining_dims: 空间或时空维度(H,W[,T])。
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float(
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        # s: 每个样本的分位阈值，抑制极端像素/潜变量值，防止饱和。
        s = torch.quantile(
            abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(
            1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(
            sample, -s, s
        ) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma):
        # sigma -> 离散时间索引 t（线性映射）。
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        # 在 flow 形式下可写为 x = alpha * x0 + sigma * noise，这里 alpha=1-sigma。
        return 1 - sigma, sigma

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.set_timesteps
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        # 时间重参数化函数：
        # mu 控制整体平移；sigma(此处形参)控制曲线形状；
        # t 为原始时间/噪声刻度，输出为变换后刻度。
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1)**sigma)

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.convert_model_output
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.
        <Tip>
        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.
        </Tip>
        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError(
                    "missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # DPM-Solver++ 以“数据预测积分形式”更新状态。
        # 对 flow_prediction：网络输出可视为速度场 v_theta(x_t, t)。
        if self.config.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.config.prediction_type == "flow_prediction":
                # sigma_t: 当前时刻噪声权重；x0_pred: 反推出的“无噪样本”估计。
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction`, or `flow_prediction` for the FlowDPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred

        # DPM-Solver 以“噪声预测积分形式”更新状态。
        elif self.config.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.config.prediction_type == "flow_prediction":
                # epsilon: 对应噪声项估计（在当前参数化下由 sample 与 flow 输出组合得到）。
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the FlowDPMSolverMultistepScheduler."
                )

            if self.config.thresholding:
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred

            return epsilon

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.dpm_solver_first_order_update
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).
        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop(
            "prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(
                    " missing `sample` as a required keyward argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # s 表示当前步，t 表示下一步（更干净方向）；sigma_s -> sigma_t。
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[
            self.step_index]  # pyright: ignore
        # alpha/sigma 是状态分解系数；lambda=log(alpha)-log(sigma) 为对数信噪比型变量。
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        # h: 对数域步长，决定本次数值积分步幅。
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            # 一阶 ODE 更新（类似 DDIM）：由当前 sample 和模型输出推进到下一时刻。
            x_t = (sigma_t /
                   sigma_s) * sample - (alpha_t *
                                        (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "dpmsolver":
            x_t = (alpha_t /
                   alpha_s) * sample - (sigma_t *
                                        (torch.exp(h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            # SDE 版本：在确定性漂移项外加入随机扩散项（noise）。
            x_t = ((sigma_t / sigma_s * torch.exp(-h)) * sample +
                   (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output +
                   sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = ((alpha_t / alpha_s) * sample - 2.0 *
                   (sigma_t * (torch.exp(h) - 1.0)) * model_output +
                   sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise)
        return x_t  # pyright: ignore

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.multistep_dpm_solver_second_order_update
    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the second-order multistep DPMSolver.
        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        timestep_list = args[0] if len(args) > 0 else kwargs.pop(
            "timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop(
            "prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(
                    " missing `sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # s0: 当前步，s1: 上一步，t: 下一步；二阶法需要两段历史信息。
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],  # pyright: ignore
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],  # pyright: ignore
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        # m0,m1: 最近两次模型输出（流/噪声估计）。
        m0, m1 = model_output_list[-1], model_output_list[-2]

        # r0: 相邻步长比；D0,D1 为 0/1 阶差分项，用于二阶外推。
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = ((sigma_t / sigma_s0) * sample -
                       (alpha_t * (torch.exp(-h) - 1.0)) * D0 - 0.5 *
                       (alpha_t * (torch.exp(-h) - 1.0)) * D1)
            elif self.config.solver_type == "heun":
                x_t = ((sigma_t / sigma_s0) * sample -
                       (alpha_t * (torch.exp(-h) - 1.0)) * D0 +
                       (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1)
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = ((alpha_t / alpha_s0) * sample -
                       (sigma_t * (torch.exp(h) - 1.0)) * D0 - 0.5 *
                       (sigma_t * (torch.exp(h) - 1.0)) * D1)
            elif self.config.solver_type == "heun":
                x_t = ((alpha_t / alpha_s0) * sample -
                       (sigma_t * (torch.exp(h) - 1.0)) * D0 -
                       (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1)
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = ((sigma_t / sigma_s0 * torch.exp(-h)) * sample +
                       (alpha_t * (1 - torch.exp(-2.0 * h))) * D0 + 0.5 *
                       (alpha_t * (1 - torch.exp(-2.0 * h))) * D1 +
                       sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)
            elif self.config.solver_type == "heun":
                x_t = ((sigma_t / sigma_s0 * torch.exp(-h)) * sample +
                       (alpha_t * (1 - torch.exp(-2.0 * h))) * D0 +
                       (alpha_t * ((1.0 - torch.exp(-2.0 * h)) /
                                   (-2.0 * h) + 1.0)) * D1 +
                       sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise)
        elif self.config.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = ((alpha_t / alpha_s0) * sample - 2.0 *
                       (sigma_t * (torch.exp(h) - 1.0)) * D0 -
                       (sigma_t * (torch.exp(h) - 1.0)) * D1 +
                       sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise)
            elif self.config.solver_type == "heun":
                x_t = ((alpha_t / alpha_s0) * sample - 2.0 *
                       (sigma_t * (torch.exp(h) - 1.0)) * D0 - 2.0 *
                       (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1 +
                       sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise)
        return x_t  # pyright: ignore

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.multistep_dpm_solver_third_order_update
    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.Tensor],
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the third-order multistep DPMSolver.
        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by diffusion process.
        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """

        timestep_list = args[0] if len(args) > 0 else kwargs.pop(
            "timestep_list", None)
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop(
            "prev_timestep", None)
        if sample is None:
            if len(args) > 2:
                sample = args[2]
            else:
                raise ValueError(
                    " missing`sample` as a required keyward argument")
        if timestep_list is not None:
            deprecate(
                "timestep_list",
                "1.0.0",
                "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # 三阶法使用 3 段历史输出：s0(当前)、s1、s2。
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],  # pyright: ignore
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],  # pyright: ignore
            self.sigmas[self.step_index - 2],  # pyright: ignore
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        # m0,m1,m2: 最近三次网络输出。
        m0, m1, m2 = model_output_list[-1], model_output_list[
            -2], model_output_list[-3]

        # D0,D1,D2: 0/1/2 阶差分项；三阶公式利用曲率信息减少离散误差。
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = ((sigma_t / sigma_s0) * sample -
                   (alpha_t * (torch.exp(-h) - 1.0)) * D0 +
                   (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1 -
                   (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2)
        elif self.config.algorithm_type == "dpmsolver":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = ((alpha_t / alpha_s0) * sample - (sigma_t *
                                                    (torch.exp(h) - 1.0)) * D0 -
                   (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1 -
                   (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2)
        return x_t  # pyright: ignore

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 在离散时间表中查找给定 timestep 的索引位置。
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        # 若是普通 txt2img 流程，从当前 timestep 自动定位索引；
        # 若是 img2img/inpaint，可能由 begin_index 指定中途起点。
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # Modified from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.step
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.
        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`LEdits++`].
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            # 首次 step 时根据当前 timestep 初始化内部步计数器。
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        # lower_order_final: 最后一两步切换低阶法，减少高阶外推在末端的抖动。
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final or
            (self.config.lower_order_final and len(self.timesteps) < 15) or
            self.config.final_sigmas_type == "zero")
        # lower_order_second: 倒数第二步必要时也降阶。
        lower_order_second = ((self.step_index == len(self.timesteps) - 2) and
                              self.config.lower_order_final and
                              len(self.timesteps) < 15)

        # 将网络原始输出统一映射到求解器期望变量（x0_pred 或 epsilon）。
        model_output = self.convert_model_output(model_output, sample=sample)
        # 维护历史窗口：左移并写入最新输出，供二阶/三阶多步法使用。
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"
                                         ] and variance_noise is None:
            # SDE 采样需要随机项；若未外部传入则在此采样标准噪声。
            noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=torch.float32)
        elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            # 使用外部给定噪声（如可复现实验或特定编辑任务）。
            noise = variance_noise.to(
                device=model_output.device,
                dtype=torch.float32)  # pyright: ignore
        else:
            # ODE 变体不需要随机扩散噪声。
            noise = None

        # 根据配置阶数和当前可用历史长度，选择一阶/二阶/三阶更新公式。
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample)

        if self.lower_order_nums < self.config.solver_order:
            # 前若干步历史不足，逐步提升到目标阶数。
            self.lower_order_nums += 1

        # Cast sample back to expected dtype
        # 输出回到模型原 dtype（如 fp16/bf16）以保持管线一致性。
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        # 内部时间前进一格，准备下次 step。
        self._step_index += 1  # pyright: ignore

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.scale_model_input
    def scale_model_input(self, sample: torch.Tensor, *args,
                          **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`torch.Tensor`):
                The input sample.
        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        # 该调度器不需要额外缩放输入，恒等返回。
        return sample

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.scale_model_input
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        # original_samples: 干净样本 x0；noise: 高斯噪声 z；timesteps: 指定加噪时刻 t。
        # 目标是构造 x_t = alpha_t * x0 + sigma_t * z（前向扩散/训练常用）。
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(
                timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(
                original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(
                original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        if self.begin_index is None:
            # 训练/普通流程：逐个 timestep 查索引。
            step_indices = [
                self.index_for_timestep(t, schedule_timesteps)
                for t in timesteps
            ]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            # 已进入采样中途：使用当前 step_index。
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            # 采样前准备 latent：使用 begin_index 对应噪声级别。
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 取出每个样本对应的 sigma，并扩展维度以匹配张量广播。
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # 前向扰动方程：把干净样本和噪声按权重线性混合得到 noisy_samples。
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        # 返回训练时间长度 T（调度器可用作长度对象）。
        return self.config.num_train_timesteps
