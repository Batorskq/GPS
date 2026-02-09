import concurrent.futures
import inspect
import os
import json
import re
import time
from collections import defaultdict
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Union
from swift.llm import PtEngine
from transformers.trainer_utils import EvalLoopOutput
from tqdm import tqdm
from swift.plugin.orm import extract_xml_answer, cal_rouge, calculate_sari
import pathlib
from torch.utils.data import DataLoader, Dataset


import numpy as np
import torch
import copy
import torch.nn as nn
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from torch.nn import ModuleList
from transformers import PreTrainedModel, TrainerCallback
from trl import GRPOTrainer as HFGRPOTrainer

from swift.llm import InferRequest, MultiModelKeys, RequestConfig, RowPreprocessor, get_model_arch, to_device
from swift.llm.infer.infer_engine import GRPOVllmEngine, set_device_context
from swift.plugin import orms
from swift.utils import (JsonlWriter, gc_collect, get_device, get_device_count, get_dist_setting, get_logger,
                         get_node_setting, is_lmdeploy_available, is_vllm_available, is_wandb_available)
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin

try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    raise ImportError('Please install trl from source using: `pip install git+https://github.com/huggingface/trl.git`')

del HFGRPOTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb




class _JsonlDataset(Dataset):
    """Minimal helper so we can re-use existing prompt template."""
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")
                     if l.strip()]

    def __getitem__(self, idx): return self.rows[idx]
    def __len__(self):          return len(self.rows)


@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator,
    gather_deepspeed3_params=True,
    gather_parameters: List = None,
):
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            import deepspeed
            parameters = [
                parameter for name, parameter in model.named_parameters()
                if not gather_parameters or name in gather_parameters
            ]
            with deepspeed.zero.GatheredParameters(parameters):
                from trl.models.utils import remove_hooks
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                from trl.models.utils import add_hooks
                add_hooks(model)
    else:
        yield unwrapped_model


class GRPOCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.queue = self.trainer.train_queue
        train_dataloader = getattr(state, 'train_dataloader', None) or kwargs.get('train_dataloader')
        self.trainer._prefetch(train_dataloader)


@dataclass
class DataCache:
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)
    distributed_idx: List[List] = field(default_factory=list)


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.args = args
        self.queue = None
        self.train_queue = Queue()
        self.eval_queue = Queue()
        self.processing_class = kwargs.get('template').tokenizer
        self.offload_modules = {}
        self.offload_states = {}
        _, _, _, local_world_size = get_dist_setting()
        if self.args.tensor_parallel_size > 1:
            assert (get_device_count() == local_world_size == self.args.num_infer_workers
                    and local_world_size > 1), ('tensor_parallel_size>1 only supports num_infer_workers==your '
                                                'available device count.')
        if self.args.async_generate:
            assert (local_world_size + self.args.num_infer_workers <=
                    get_device_count()), ('async_generate requires training and rollout use '
                                          'different GPUS.')

        if self.args.sleep_level > 0:
            if local_world_size + self.args.num_infer_workers <= get_device_count():
                logger.warning('You are using different GPUs for training and rollout, '
                               'so you do not need to use sleep_level > 0')

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        self.reward_funcs = reward_funcs
        self.reward_templates = [None] * len(self.reward_funcs)
        if reward_model is not None:
            self.reward_templates.append(kwargs.pop('reward_template', None))
            self.reward_funcs.append(reward_model)
        if not self.reward_funcs:
            raise ValueError('You must specify reward_funcs or reward_model')
        self.template = kwargs["template"]
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        self.num_generations = args.num_generations
        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = lambda features: features
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}

        use_vllm = args.use_vllm
        use_lmdeploy = args.use_lmdeploy

        super().__init__(model, ref_model, *_args, **kwargs)

        self.frozen_model = PtEngine("Qwen/Qwen2.5-7B-Instruct", model_type="qwen2_5", device_map='auto')
        self.frozen_model.model.eval()  # Note: PtEngine stores the underlying model in self.frozen_model.model
        for param in self.frozen_model.model.parameters():
            param.requires_grad = False
        
        ### FOR IDENTICAL PROMPT FOR INFERENCE ###
        input_file = f"datasets/original/{os.environ['DATASET']}_train.jsonl"
        input_file = f"datasets/original/gen_train.jsonl"
        with open(input_file, "r") as f:
            self.train_data = [json.loads(line) for line in f if line.strip()]
        self.reasoning_system = self.train_data[0]["messages"][0]["content"]
        #self.reasoning_prompt = self.train_data[0]["messages"][1]["content"]
        self.base_prompt = "You are a helpful assistant."
        ### FOR IDENTICAL PROMPT FOR INFERENCE ###

        if reward_funcs:
                for i, reward_func in enumerate(reward_funcs):
                    if reward_func in orms:
                        reward_func_class = orms[reward_func]
                        reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                        reward_func_kwargs = {
                            key: getattr(args, key)
                            for key in reward_func_args
                            if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                        }
                        if 'tokenizer' in reward_func_args:
                            reward_func_kwargs['tokenizer'] = self.processing_class
                        # --- Modification for GridAccuracy ---
                        # If the reward function is "accuracy" (GridAccuracy), then add the required arguments.
                        if reward_func == 'accuracy':
                            reward_func_kwargs['frozen_model'] = self.frozen_model
                            reward_func_kwargs['template'] = self.template
                            reward_func_kwargs["train_data"] = self.train_data
                            #import pdb; pdb.set_trace()
                            #reward_func_kwargs['request_config'] = self.request_config
                        # --------------------------------------
                        reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                    elif not callable(reward_func):
                        raise ValueError(f'reward_function {reward_func} is not implemented in swift.llm.plugin')
            # ... [rest of __init__] ...

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f'The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly '
                f'divisible by the number of generations per prompt ({self.num_generations}). Given the current train '
                f'batch size, the valid values for the number of generations are: {possible_values}.')
        if self.args.eval_strategy != 'no':
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f'The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly '
                    f'divisible by the number of generations per prompt ({self.num_generations}). Given the current '
                    f'eval batch size, the valid values for the number of generations are: {possible_values}.')

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
        self.infer_device = None

        if use_vllm or use_lmdeploy:
            if self.infer_rank >= 0:
                fast_infer_device = self.args.vllm_device or self.args.lmdeploy_device
                if fast_infer_device[0] == 'auto':
                    if get_device_count() == 1:
                        fast_infer_device = [get_device()]  # particular case when training with only 1 GPU: share it
                    else:
                        fast_infer_device = []
                        for idx in range(get_device_count() - self.args.num_infer_workers, get_device_count()):
                            fast_infer_device.append(get_device(idx))

                for _device in fast_infer_device:
                    # Check that the requested device is available
                    if _device.split(':')[0] in {'cuda', 'npu'} and int(_device.split(':')[1]) >= get_device_count():
                        raise ValueError(f'The requested device for vllm ({_device}) is not available. '
                                         f'You are likely using vLLM '
                                         'without restricting the number of GPUs for training. '
                                         'Set the `--num_processes` argument to a '
                                         'value lower than the number of GPUs available on your machine—typically, '
                                         'reducing it by one is sufficient. '
                                         f'In your case: `--num_processes {get_device_count() - 1}`.')
                    # Check that the requested device is not also used for training
                    if _device in {get_device(idx) for idx in range(self.accelerator.num_processes)}:
                        logger.warning(f'The requested device {_device} is also used for training. '
                                       f'This may lead to unexpected behavior. '
                                       f'It is recommended to use a dedicated device for vLLM.')

                if use_vllm:
                    if not is_vllm_available():
                        raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                          'Please install vLLM with `pip install vllm -U` to use it.')
                    self.prepare_vllm(model, fast_infer_device)
                    self.infer_device = fast_infer_device[self.local_infer_rank]
                elif use_lmdeploy:
                    if not is_lmdeploy_available():
                        raise ImportError('LMDeploy is not available and `use_lmdeploy` is set to True.'
                                          'Please install LMDeploy with `pip install lmdeploy -U` to use it.')
                    from swift.llm import LmdeployEngine
                    from swift.tuners import Swift
                    with Swift.grpo_context(model, self.template.processor):
                        fast_infer_device = int(fast_infer_device[self.local_infer_rank].split(':')[1])
                        self.engine = LmdeployEngine(
                            model.model_dir,
                            model.model_info.torch_dtype,
                            model_type=model.model_meta.model_type,
                            devices=[fast_infer_device],
                            session_len=args.lmdeploy_session_len,
                            cache_max_entry_count=args.lmdeploy_cache_max_entry_count,
                            reload_weights=True)
                        self.infer_device = fast_infer_device
                    self.engine.default_template = self.template
            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
        )

        self.model_accepts_loss_kwargs = False
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        self.log_completions = args.log_completions
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps
        if self.args.async_generate:
            self.add_callback(GRPOCallback(self))

    def split_batches(self):
        """Sync weights in batches
        Only split LLM layers for now:
        1. N batches for layers
        2. other, embeds, lm_heads in one batch
        3. multi-modal components in one batch
        """
        if self.args.move_model_batches is None:
            # All in one
            return [None], [None]

        model = self.accelerator.unwrap_model(self.model)
        model_arch = get_model_arch(model.model_meta.model_arch)
        non_llm_parameters = []
        llm_embeds = []
        parameters = []
        pattern = r'\.(\d+)\.'

        layer_count = None
        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                    llm = model_arch.language_model
                    if name.startswith(llm):
                        layer_count = len(module)
                else:
                    layer_count = len(module)
        assert layer_count is not None, 'Cannot find ModuleList to split modules.'

        n_layers = ceil(layer_count / self.args.move_model_batches)
        for _ in range(self.args.move_model_batches):
            parameters.append([])

        def replace_lora(name):
            if 'lora_A' in name or 'lora_B' in name:
                return ''
            else:
                return name.replace('base_layer.', '')

        def remove_lora(names):
            names = set([replace_lora(n) for n in names])
            return [n for n in names if n]

        def split_llm(name):
            match = re.search(pattern, name)
            if match:
                number = match.group(1)
                group = int(number) // n_layers
                parameters[group].append(name)
            else:
                llm_embeds.append(name)

        for name, parameter in model.named_parameters():
            if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                llm = model_arch.language_model
                if name.startswith(llm):
                    split_llm(name)
                else:
                    non_llm_parameters.append(name)
            else:
                split_llm(name)

        if llm_embeds:
            parameters.append(llm_embeds)
        if non_llm_parameters:
            parameters.append(non_llm_parameters)
        return parameters, [remove_lora(p_list) for p_list in parameters]

    def prepare_vllm(self, model, fast_infer_device):
        from swift.tuners import Swift
        from swift.llm import VllmEngine
        _, _, _, local_world_size = get_dist_setting()
        if local_world_size == self.args.num_infer_workers == get_device_count() and local_world_size > 1:
            cls = GRPOVllmEngine
        else:
            cls = VllmEngine
        with Swift.grpo_context(model, self.template.processor):
            self.engine = cls(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                device=fast_infer_device[self.local_infer_rank],
                tensor_parallel_size=self.args.tensor_parallel_size,
                gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                max_num_seqs=self.args.vllm_max_num_seqs,
                enforce_eager=self.args.vllm_enforce_eager,
                limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
                num_infer_workers=self.args.num_infer_workers,
                enable_sleep_mode=self.args.sleep_level > 0,
                use_async_engine=False,
                distributed_executor_backend='external_launcher',
                max_model_len=self.args.vllm_max_model_len)
            self.engine.default_template = self.template

    @property
    def infer_rank(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        for _vllm_rank in range(self.args.num_infer_workers):
            if local_rank == _vllm_rank:
                return get_node_setting()[0] * self.args.num_infer_workers + _vllm_rank

        return -1

    @property
    def local_infer_rank(self):
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        for _vllm_rank in range(self.args.num_infer_workers):
            if local_rank == _vllm_rank:
                return _vllm_rank

        return -1

    @staticmethod
    def round_robin(num_reqs, nodes):
        distribution = [[] for _ in range(nodes)]
        for idx in range(num_reqs):
            node_id = idx % nodes
            distribution[node_id].append(idx)
        return distribution

    @staticmethod
    @contextmanager
    def _template_context(template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        mode = template.mode
        if mode in {'vllm', 'pt', 'lmdeploy'}:
            template.set_mode('train')
        template.max_length = None
        try:
            yield
        finally:
            template.set_mode(mode)
            template.max_length = max_length

    @torch.no_grad()
    def offload_model(self):
        if len(self.offload_modules) > 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                self.offload_modules[name] = module.weight.device
                module.to('cpu')
            elif not hasattr(module, 'device'):
                pass
            elif module.device.type != 'cpu':
                self.offload_modules[name] = module.device
                module.to('cpu')

    @torch.no_grad()
    def load_model(self):
        if len(self.offload_modules) == 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, device in self.offload_modules.items():
            module = unwrapped_model.get_submodule(name)
            if isinstance(module, torch.nn.Embedding):
                module.weight.to(device)
            else:
                module.to(device)
        self.offload_modules.clear()

    @torch.no_grad()
    def offload_optimizer(self):
        if len(self.offload_states) > 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        self.offload_states[key] = value.device
                        state[key] = value.to('cpu', non_blocking=True)

    @torch.no_grad()
    def load_optimizer(self):
        if len(self.offload_states) == 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.offload_states[key], non_blocking=True)
        self.offload_states.clear()

    @profiling_decorator
    def _move_model_to_vllm_lmdeploy(self):
        # TODO This may be low efficiency
        # 1. deepspeed parallel == vllm tensor parallel, may be do not need to gather
        # 2. may be each process in tp group only need gather a part of the parameters
        # 3. the split of parameter_groups may be imbalanced
        from accelerate.utils.other import is_compiled_module

        for i, parameter_group in enumerate(self.parameter_groups):
            parameter_group_no_lora = self.parameter_groups_no_lora[i]
            with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    gather_parameters=parameter_group) as unwrapped_model:

                def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
                    if parameter_group and all([self.name not in pg for pg in parameter_group]):
                        # Not this group, skip
                        return
                    else:
                        ret = self.merge_origin(safe_merge, adapter_names)
                        return ret

                def get_delta_weight(self, adapter) -> torch.Tensor:
                    # may be offload
                    self.lora_A[adapter].weight.data = self.lora_A[adapter].weight.data.to(
                        self.base_layer.weight.device)
                    self.lora_B[adapter].weight.data = self.lora_B[adapter].weight.data.to(
                        self.base_layer.weight.device)
                    tensor = self.get_delta_weight_origin(adapter)
                    return tensor.to(self.base_layer.weight.device)

                @contextmanager
                def patch_merge(model):
                    from peft.tuners.lora import LoraLayer
                    for name, module in model.named_modules():
                        if isinstance(module, LoraLayer):
                            module.name = name
                            if not hasattr(module, 'merge_origin') and hasattr(module, 'base_layer'):
                                module.merge_origin = module.merge
                                module.merge = MethodType(merge, module)
                                module.get_delta_weight_origin = module.get_delta_weight
                                module.get_delta_weight = MethodType(get_delta_weight, module)
                    yield
                    for name, module in model.named_modules():
                        if isinstance(module, LoraLayer):
                            if hasattr(module, 'merge_origin'):
                                module.merge = module.merge_origin
                                del module.merge_origin
                                module.get_delta_weight = module.get_delta_weight_origin
                                del module.get_delta_weight_origin

                if is_compiled_module(unwrapped_model):
                    unwrapped_model = unwrapped_model._orig_mod
                if is_peft_model(unwrapped_model):
                    with patch_merge(unwrapped_model):
                        unwrapped_model.merge_adapter()
                    state_dict = unwrapped_model.state_dict()
                    # Remove base_model and base_layer prefixes
                    state_dict = {
                        k.removeprefix('base_model.model.').replace('.base_layer', ''): v
                        for k, v in state_dict.items()
                    }
                    # Remove values with adapter prefix (example: "_lora")
                    state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                    # When module to save, remove its prefix and discard the original module
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                else:
                    state_dict = unwrapped_model.state_dict()
                if parameter_group_no_lora:
                    parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                    state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}
                assert all([state.shape != torch.Size([0]) for state in state_dict.values()])
                if self.infer_rank >= 0:
                    if self.args.async_generate:
                        self._wait_queue()
                    if self.args.use_vllm:
                        llm_model = self.engine.inner_model
                    else:
                        llm_model = self.engine.engine.engine
                    llm_model.load_weights(state_dict.items())
                # Unmerge the adapter to restore the model to its original state.
                # This must be done after loading weights to ensure they correspond to the merged state.
                if is_peft_model(unwrapped_model):
                    unwrapped_model.unmerge_adapter()

    def _wait_queue(self):
        while self.queue.empty():
            time.sleep(0.01)

    @staticmethod
    def reorder_outputs(outputs, distributed_idx):
        index_to_output = {}
        current_position = 0
        for output_idx in distributed_idx:
            for idx in output_idx:
                index_to_output[idx] = outputs[current_position]
                current_position += 1

        return [index_to_output[idx] for idx in sorted(index_to_output.keys())]

    def async_infer(self, inputs, inputs_slice, distributed_idx):

        def infer_task():
            with set_device_context(self.infer_device):
                result = self.engine.infer(
                    infer_requests=inputs_slice, request_config=self.request_config, use_tqdm=False)
                return result

        future: Future = self.executor.submit(infer_task)

        def done(_self):
            self.queue.put(DataCache(inputs, _self.result(), distributed_idx))

        future.add_done_callback(done)

    def _prefetch(self, dataloader):
        inputs = next(iter(dataloader))
        all_inputs = gather_object(inputs)
        distributed_idx = self.round_robin(len(all_inputs), get_node_setting()[1] * self.args.num_infer_workers)
        if self.infer_rank >= 0:
            _input_slice = np.array(all_inputs)[distributed_idx[self.infer_rank]]
            outputs = self.engine.infer(_input_slice, self.request_config, use_tqdm=False)
            self.queue.put(DataCache(inputs, outputs, distributed_idx))
        else:
            self.queue.put(DataCache(inputs, [], distributed_idx))
        if self.accelerator.num_processes > 1:
            self.accelerator.wait_for_everyone()

    def _fast_infer(self, inputs):
        self.state.eval_steps = self.args.eval_steps
        if self.args.sleep_level > 0 and self.infer_rank >= 0:
            if self.args.offload_model:
                self.offload_model()
            if self.args.offload_optimizer:
                self.offload_optimizer()
            if self.args.gc_collect_after_offload:
                gc_collect()
            self.engine.engine.wake_up()
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm_lmdeploy()
            self._last_loaded_step = self.state.global_step
        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_inputs = gather_object(inputs)
        # Distribute inputs to different workers
        # for example, 2 workers, 6 inputs, 0/2/4 dispatch to the first worker
        # 1/3/5 dispatch to the second worker
        # trying to shuffle and average the length
        distributed_idx = self.round_robin(len(all_inputs), get_node_setting()[1] * self.args.num_infer_workers)
        if self.infer_rank >= 0:
            _input_slice = np.array(all_inputs)[distributed_idx[self.infer_rank]]
            if self.args.async_generate:
                self.async_infer(inputs, _input_slice, distributed_idx)
                data_cache = self.queue.get()
                inputs = data_cache.inputs
                outputs = data_cache.outputs
                distributed_idx = data_cache.distributed_idx
            else:
                with set_device_context(self.infer_device):
                    outputs = self.engine.infer(_input_slice, self.request_config, use_tqdm=False)
        else:
            if self.args.async_generate:
                self.queue.put(DataCache(inputs, [], distributed_idx))
                data_cache = self.queue.get()
                inputs = data_cache.inputs
                distributed_idx = data_cache.distributed_idx
            outputs = []
        outputs = gather_object(outputs)
        outputs = self.reorder_outputs(outputs, distributed_idx)
        if self.args.sleep_level > 0 and self.infer_rank >= 0:
            self.engine.engine.sleep(level=self.args.sleep_level)
            if self.args.gc_collect_after_offload:
                gc_collect()
            if self.args.offload_model:
                self.load_model()
            if self.args.offload_optimizer:
                self.load_optimizer()
        return inputs, outputs

    @property
    def old_policy(self):
        return self.num_iterations > 1 or self.args.async_generate

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm or self.args.use_lmdeploy:
            inputs, outputs = self._fast_infer(inputs)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            # outputs = broadcast_object_list(outputs, from_process=0)
        else:
            # Regular generation path
            is_multimodal = self.model.model_meta.is_multimodal
            if is_multimodal:
                models = self.template.remove_post_encode_hook()
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator):
                # same reference
                outputs = self.engine.infer(inputs, self.request_config, use_tqdm=False)
                self.model.train()
            if is_multimodal:
                self.template.register_post_encode_hook(models)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        if self.args.use_vllm or self.args.use_lmdeploy:
            outputs = outputs[process_slice]

        for i, output in enumerate(outputs):
            messages = inputs[i]['messages']
            InferRequest.remove_response(messages)
            messages.append({'role': 'assistant', 'content': output.choices[0].message.content})
        from copy import copy
        template = copy(self.template)
        with self._template_context(template):
            batched_inputs = [template.encode(infer_request) for infer_request in inputs]
            outputs = to_device(template.data_collator(batched_inputs), self.model.device)

        # we only need to compute the logits for the completion tokens
        labels = outputs.pop('labels')
        logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        outputs['logits_to_keep'] = logits_to_keep
        outputs['completion_mask'] = labels[:, -logits_to_keep:] != -100

        with torch.no_grad():
            if self.old_policy:
                outputs['old_per_token_logps'] = self._get_per_token_logps(self.model, outputs)
            else:
                outputs['old_per_token_logps'] = None

            #outputs['old_per_token_logps'] = self._get_per_token_logps(self.model, outputs)

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, outputs)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(self.model, outputs)

        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        completions = [example['messages'][-1]['content'] for example in inputs]

        for i, (reward_func, reward_template) in enumerate(zip(self.reward_funcs, self.reward_templates)):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                with self._template_context(reward_template):
                    batched_inputs = [reward_template.encode(infer_request) for infer_request in inputs]
                    reward_inputs = to_device(reward_template.data_collator(batched_inputs), reward_func.device)

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = RowPreprocessor.rows_to_batched(inputs)
                reward_kwargs["observations"] = inputs
                output_reward_func = reward_func(completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages[process_slice]

        # Log the metrics
        mode = 'eval' if self.control.should_evaluate else 'train'
        completion_length = self.accelerator.gather_for_metrics(outputs['completion_mask'].sum(1)).float().mean().item()
        self._metrics[mode]['completion_length'].append(completion_length)
        # clip ratio
        response_clip_ratio = torch.gt(
            self.accelerator.gather_for_metrics(outputs['completion_mask'].sum(1)),
            self.args.max_completion_length).float().mean().item()
        self._metrics[mode]['response_clip_ratio'].append(response_clip_ratio)
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split('/')[-1]
            else:
                if inspect.isfunction(reward_func):
                    reward_func_name = reward_func.__name__  # function
                else:
                    reward_func_name = reward_func.__class__.__name__  # method
            self._metrics[mode][f'rewards/{reward_func_name}'].append(reward_per_func[i].item())

        self._metrics[mode]['reward'].append(rewards.mean().item())
        self._metrics[mode]['reward_std'].append(std_grouped_rewards.mean().item())
        outputs.update({
            'ref_per_token_logps': ref_per_token_logps,
            'advantages': advantages,
        })
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # For logging
            table = {
                'step': [str(self.state.global_step)] * len(rewards),
                'messages': [inputs['messages'][:-1] for inputs in gather_object(inputs)],
                'completion': gather_object(completions),
                'reward': rewards.tolist(),
            }
            self.jsonl_writer.append(table)
            if 'wandb' in self.args.report_to and wandb.run is not None and self.accelerator.is_main_process:
                import pandas as pd
                df = pd.DataFrame(table)
                wandb.log({'completions': wandb.Table(dataframe=df)})

        return outputs

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError('The GRPOTrainer does not support returning outputs')
        # Compute the per-token log probabilities for the model
        completion_mask = inputs['completion_mask']
        per_token_logps = self._get_per_token_logps(model, inputs)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        old_per_token_logps = inputs['old_per_token_logps'] if self.old_policy else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = 'eval' if self.control.should_evaluate else 'train'

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]['clip_ratio'].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        unwrapped_model = self.accelerator.unwrap_model(model)
        parameters = inspect.signature(unwrapped_model.forward).parameters
        if not unwrapped_model.model_meta.is_multimodal and 'logits_to_keep' in parameters:
            # save memory
            return super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
        inputs = {
            k: v
            for k, v in inputs.items() if k not in
            ['logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps']
        }
        logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens


    def _log_wandb(self, name, value, step = None):
        if (
            "wandb" in self.args.report_to
            and wandb.run is not None
            and self.accelerator.is_main_process
        ):
            wandb.log({name: value}, step=step)


    # def evaluation_loop(self, *args, num_candidates: int = 2, **kwargs):
    #     """
    #     – gen  → ROUGE          (original behaviour)
    #     – sim  → SARI           (runs only when DATASET == 'gen')
    #     – subj → accuracy       (runs only when DATASET == 'gen')
    #     """
    #     ####################################################################
    #     # 0.  little helpers
    #     ####################################################################
    #     def _build_dataloader(path):
    #         return DataLoader(
    #             _JsonlDataset(path),
    #             batch_size=self.args.per_device_eval_batch_size,
    #             shuffle=False,
    #             num_workers=self.args.dataloader_num_workers,
    #             collate_fn=lambda x: x,                 # keep dicts untouched
    #         )

    #     def _single_eval(eval_dl, metric_name, file_suffix):
    #         best_preds, total, correct = [], 0, 0
    #         r1 = r2 = rl = sari_total = 0.0

    #         for step, batch in enumerate(tqdm(eval_dl, desc=f"Eval {file_suffix}")):
    #             with torch.no_grad():
    #                 batch_size  = len(batch)
    #                 observations = [
    #                     b['messages'][1]['content'].split("OBSERVATION: \n\n ")[-1]
    #                     for b in batch
    #                 ]

                    

    #                 if os.environ["PSP"]=="False":
                        
    #                     if "summarization task" in batch[0]['messages'][1]['content']:
    #                         prompt_for_generation =  (
    #         "Your task is to refine a base prompt for another model that performs a "
    #         "summarization task. You will be given the base prompt that you should enhance. "
    #         "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
    #     )   
    #                         prompt_for_generation = prompt_for_generation + """BASE PROMPT: \n\n""How would you rephrase that in a few words?"""
                            
    #                     elif "simplification task" in batch[0]['messages'][1]['content']:
    #                         prompt_for_generation =  (
    #         "Your task is to refine a base prompt for another model that performs a "
    #         "simplification task. You will be given the base prompt that you should enhance. "
    #         "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
    #     )
    #                         prompt_for_generation = prompt_for_generation + "BASE PROMPT: \n\n Simplify the text."
                            
    #                     elif "subjectivity classification task" in batch[0]['messages'][1]['content']:
    #                         prompt_for_generation = (
    #         "Your task is to refine a base prompt for another model that performs a subjectivity classification task. "
    #         "You will be given the base prompt that you should enhance. "
    #         "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
    #     )
    #                         prompt_for_generation = + (
    #         "Your task is to refine a base prompt for another model that performs a subjectivity classification task. "
    #         "You will be given the base prompt that you should enhance. "
    #         "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
    #     )
                            
    #                         prompt_for_generation += "Your task is to refine a base prompt for another model that performs a math task. You will be given the base prompt and the observation for which the prompt should be enhanced for. Improve the instructions to enhance the model's performance. Return only the enhanced prompt.BASE PROMPT: \n\n Solve this riddle and return ONLY the integer answer" 
                            
    #                     else:
    #                         raise Exception("!!!")
                        
    #                     base_reqs = [{"messages": [
    #                                     {"role": "system", "content": self.reasoning_system},
    #                                     {"role": "user",   "content": prompt_for_generation}
    #                                 ]}]
    #                     raw = self.engine.infer(base_reqs, self.request_config)
    #                     generated_prompt = raw[0].choices[0].message.content.strip()
    #                     generated_prompt = extract_xml_answer(generated_prompt)
    #                     #import pdb; pdb.set_trace()
                    
    #                 else:

    #                     # 1) generate candidates with *trainable* model
    #                     base_reqs = [{"messages": [
    #                                     {"role": "system", "content": self.reasoning_system},
    #                                     {"role": "user",   "content": b['messages'][1]['content']}
    #                                 ]}
    #                                 for b in batch]
                        

    #                     raw = self.engine.infer(base_reqs * num_candidates, self.request_config)
    #                     cand_sets = [raw[i*batch_size:(i+1)*batch_size] for i in range(num_candidates)]

    #                 scores        = [[0.0]*batch_size for _ in range(num_candidates)]
    #                 prompts       = [[None]*batch_size for _ in range(num_candidates)] 
    #                 frozen_outs   = [[None]*batch_size for _ in range(num_candidates)]  


    #                 for j in range(num_candidates):
    #                     for i in range(batch_size):
    #                         if os.environ["PSP"] == "True":
    #                             cand_answer = extract_xml_answer(cand_sets[j][i].choices[0].message.content.strip())
    #                         elif os.environ["PSP"] == "False":
    #                             cand_answer = generated_prompt 
    #                         else:
    #                             raise Exception("!!!")

    #                         prompt = (f"{cand_answer}\n{observations[i]}")
    #                         frozen_req = {"messages": [
    #                                         {"role": "system", "content": self.base_prompt},
    #                                         {"role": "user",   "content": prompt}
    #                                     ]}
    #                         frozen_out = self.frozen_model.infer([frozen_req], self.request_config)[0]
    #                         pred_text  = frozen_out.choices[0].message.content.strip() 

    #                         prompts[j][i]     = prompt          
    #                         frozen_outs[j][i] = pred_text      

    #                         ref = batch[i]["solution"]
    #                         if metric_name == "accuracy":
    #                             scores[j][i] = float(pred_text == ref)
    #                         elif metric_name == "rouge":
    #                             r1_i, r2_i, rl_i = cal_rouge(
    #                                 [pred_text.replace("\n", "")],
    #                                 [ref.replace("\n", "")]) 

    #                             #import pdb; pdb.set_trace()
    #                             scores[j][i] = float((r1_i + r2_i + rl_i) / 3.0)
    #                         else:  # sari
    #                             try:
    #                                 #if float(calculate_sari(ref.replace("\n", ""), pred_text.replace("\n", ""), observations[i].replace("\n", ""))) < 35:
    #                                 #    import pdb; pdb.set_trace()
                                    
    #                                 scores[j][i] = float(calculate_sari(ref.replace("\n", ""), pred_text.replace("\n", ""), observations[i].replace("\n", "")))
                                        
    #                             except Exception:
    #                                 scores[j][i] = 0.0

    #                 # 3) pick the best candidate for every example
    #                 for i in range(batch_size):
    #                     best_j = max(range(num_candidates), key=lambda j: scores[j][i])

    #                     ref   = batch[i]["solution"]

    #                     if os.environ["PSP"] == "True":
    #                         final = extract_xml_answer(cand_sets[best_j][i].choices[0].message.content.strip())
    #                     elif os.environ["PSP"] == "False":
    #                         final = generated_prompt
    #                     else:
    #                         raise Exception("!!!")

    #                     best_preds.append({
    #                         "observation":        observations[i],
    #                         "frozen_prompt":      prompts[best_j][i],      
    #                         "frozen_prediction":  frozen_outs[best_j][i],  
    #                         "prediction":         final,
    #                         "reference":          ref,
    #                         "score":              scores[best_j][i],
    #                     })

    #                     if metric_name == "accuracy":
    #                         correct += int(frozen_outs[best_j][i] == ref)
    #                         total   += 1
    #                     elif metric_name == "rouge":
    #                         try:
    #                             r1_i, r2_i, rl_i = cal_rouge(
    #                                 [frozen_outs[best_j][i].replace("\n", "")],
    #                                 [ref.replace("\n", "")])
    #                             r1 += r1_i; r2 += r2_i; rl += rl_i
    #                             correct += (r1_i + r2_i + rl_i) / 3.0
    #                             total   += 1
    #                         except:
    #                             r1_i = 0
    #                             r2_i = 0
    #                             rl_i = 0
    #                     else:  
    #                         try:
    #                             sari_score = calculate_sari(
    #                                 ref.replace("\n", ""),          # ➊ source
    #                                 frozen_outs[best_j][i].replace("\n", ""),   # ➋ prediction
    #                                 observations[i].replace("\n", "")                       # ➌ reference
    #                             )
                                
    #                         except Exception:
    #                             sari_score = 0.0
    #                         sari_total += sari_score
    #                         correct    += sari_score
    #                         total      += 1

    #         # -------- 4. dump preds
    #         suffix = file_suffix
    #         out_path = pathlib.Path(self.args.output_dir) / \
    #                 f"inference_log_{suffix}_{self.state.global_step}.json"
    #         out_path.write_text(json.dumps(best_preds, indent=2, ensure_ascii=False),
    #                             encoding="utf-8")

    #         # -------- 5. final metric(s)
    #         if metric_name == "accuracy":
    #             return correct / total, None
    #         elif metric_name == "rouge":
    #             return correct / total, (r1/total, r2/total, rl/total)
    #         else:  # sari
    #             return sari_total / total, None
    #     ####################################################################

    #     # keep original generation length behaviour
    #     old_num_generations = self.num_generations
    #     self.num_generations = 1
    #     self.model.eval()

    #     # -- 1. GEN (ROUGE) ------------------------------------------------
    #     if "sum" in os.environ.get("INF", ""):
    #         #eval_dl_gen = self.get_eval_dataloader()

    #         ### SMAL SUM ### 

    #         sum_path = f"datasets/original/sum_test.jsonl"
    #         eval_dl_gen = _build_dataloader(sum_path)

    #         ### SMAL SUM ### 

    #         rouge_score, (r1, r2, rl) = _single_eval(eval_dl_gen, "rouge", "gen")

    #         # log immediately 

    #         print("ROUGE-1 SCORE:", r1)
    #         print("ROUGE-2 SCORE:", r2)
    #         print("ROUGE-L SCORE:", rl)

    #         self._log_wandb("best_eval_rouge1", r1)
    #         self._log_wandb("best_eval_rouge2", r2)
    #         self._log_wandb("best_eval_rougel", rl)
    #     else:
    #         r1 = r2 = rl = 0.0

    #     # -- 2. SIM (SARI) -----------------------------------------------------
    #     valset = "test" if os.environ.get("AUX") == "True" else "val"

    #     if os.environ.get("DATASET") == "gen":
    #         if "sim" in os.environ.get("INF", ""):
    #             sim_path = f"datasets/original/sim_{valset}.jsonl"
    #             eval_dl_sim = _build_dataloader(sim_path)
    #             sari_score, _ = _single_eval(eval_dl_sim, "sari", "sim")

    #             print("SARI:", sari_score)

    #             self._log_wandb("best_eval_sari", sari_score)
    #         else:
    #             sari_score = 0.0

    #         # -- 3. SUBJ (ACC) --------------------------------------------------
    #         if "subj" in os.environ.get("INF", ""):
    #             subj_path = f"datasets/original/subj_{valset}.jsonl"
    #             eval_dl_subj = _build_dataloader(subj_path)
    #             acc_subj, _ = _single_eval(eval_dl_subj, "accuracy", "subj")
    #             self._log_wandb("best_eval_accuracy_subj", acc_subj)
    #         else:
    #             acc_subj = 0.0


    #         if "gsm8k" in os.environ.get("INF", ""):
    #             gsm8k_path = f"datasets/original/gsm8k_{valset}.jsonl"
    #             eval_dl_gsm8k = _build_dataloader(gsm8k_path)
    #             acc_gsm8k, _ = _single_eval(eval_dl_gsm8k, "accuracy", "subj")
    #             self._log_wandb("best_eval_accuracy_gsm8k", acc_gsm8k)

    #             print("ACC GSM8K", acc_gsm8k)


    #         else:
    #             acc_gsm8k = 0.0

    #         # -- 4. MR (ACC) ----------------------------------------------------
    #         if "mr" in os.environ.get("INF", ""):
    #             mr_path = f"datasets/original/mr_{valset}.jsonl"
    #             eval_dl_mr = _build_dataloader(mr_path)
    #             acc_mr, _ = _single_eval(eval_dl_mr, "accuracy", "mr")
    #             print("MR ACC:", acc_mr)
    #             self._log_wandb("best_eval_accuracy_mr", acc_mr)
    #         else:
    #             acc_mr = 0.0

    #         # -- 5. CR (ACC) ----------------------------------------------------
    #         if "cr" in os.environ.get("INF", ""):
    #             cr_path = f"datasets/original/cr_{valset}.jsonl"
    #             eval_dl_cr = _build_dataloader(cr_path)
    #             acc_cr, _ = _single_eval(eval_dl_cr, "accuracy", "cr")
    #             self._log_wandb("best_eval_accuracy_cr", acc_cr)
    #         else:
    #             acc_cr = 0.0

    #         # -- 6. SST-2 (ACC) -------------------------------------------------
    #         if any(k in os.environ.get("INF", "") for k in ("sst2", "sst-2")):
    #             sst2_path = f"datasets/original/sst-2_{valset}.jsonl"
    #             eval_dl_sst2 = _build_dataloader(sst2_path)
    #             acc_sst2, _ = _single_eval(eval_dl_sst2, "accuracy", "sst-2")
    #             self._log_wandb("best_eval_accuracy_sst2", acc_sst2)
    #         else:
    #             acc_sst2 = 0.0

    #         # -- 7. SST-5 (ACC) -------------------------------------------------
    #         if any(k in os.environ.get("INF", "") for k in ("sst5", "sst-5")):
    #             sst5_path = f"datasets/original/sst5_{valset}.jsonl"
    #             eval_dl_sst5 = _build_dataloader(sst5_path)
    #             acc_sst5, _ = _single_eval(eval_dl_sst5, "accuracy", "sst-5")

    #             print("SST5 ACC:", acc_sst5)
    #             self._log_wandb("best_eval_accuracy_sst5", acc_sst5)
    #         else:
    #             acc_sst5 = 0.0

    #         # -- 8. NEWS (ACC) --------------------------------------------------
    #         if "news" in os.environ.get("INF", ""):
    #             news_path = f"datasets/original/news_{valset}.jsonl"
    #             eval_dl_news = _build_dataloader(news_path)
    #             acc_news, _ = _single_eval(eval_dl_news, "accuracy", "news")
    #             self._log_wandb("best_eval_accuracy_news", acc_news)
    #         else:
    #             acc_news = 0.0

    #         # -- 9. TREC (ACC) --------------------------------------------------
    #         if "trec" in os.environ.get("INF", ""):
    #             trec_path = f"datasets/original/trec_{valset}.jsonl"
    #             eval_dl_trec = _build_dataloader(trec_path)
    #             acc_trec, _ = _single_eval(eval_dl_trec, "accuracy", "trec")
    #             self._log_wandb("best_eval_accuracy_trec", acc_trec)

    #             print("ACC TREC:", acc_trec)
    #         else:
    #             acc_trec = 0.0
    #     else:
    #         sari_score = acc_subj = 0.0

    #     if os.environ["AUX"] == "True":
    #         exit()

    #     self.num_generations = old_num_generations

    #     # returning just one primary metric keeps HF-Trainer happy; you can
    #     # extend this dict if you have callbacks that need the others.
    #     return EvalLoopOutput(
    #         predictions=[],
    #         label_ids=[],
    #         metrics={
    #             "eval_rouge1":   r1,
    #             "eval_rouge2":   r2,
    #             "eval_rougel":   rl,
    #             "eval_sari":     sari_score,
    #             "eval_acc_subj": acc_subj,
    #             "eval_reward":   rouge_score, 
    #         },
    #         num_samples=len(eval_dl_gen.dataset)
    #     )



    def evaluation_loop(self, *args, num_candidates: int = 15, mbr_mode: str = "normal", **kwargs):
        """
        – gen  → ROUGE          (only when INF contains 'sum')
        – sim  → SARI           (only when DATASET == 'gen' and INF contains 'sim')
        – subj → ACCURACY       (classification tasks when INF contains respective tags)

        MBR selection modes (classification only):
        - "normal"  : unweighted majority vote over evaluator (frozen) labels.
        - "weighted": weighted vote using evaluator-reported confidences/probabilities.
                        (We *parse* confidences from the evaluator output; if none found,
                        we fall back to weight=1.0 → same as "normal".)
        - "max"     : pick the single candidate whose evaluator output has the highest
                        parsed confidence for its own predicted label (fallback to first).

        Notes:
        * No ROUGE is computed for classification. Pairwise-ROUGE is only used for
            "rouge" and (heuristically) for "sari" selection.
        * Runtime for classification stays similar to the previous code (same number
            of frozen-model calls; added logic is just string parsing).
        """
        import os
        import json
        import pathlib
        from collections import Counter, defaultdict
        from typing import List, Tuple, Dict, Optional
        from tqdm import tqdm
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers.trainer_utils import EvalLoopOutput

        assert mbr_mode in {"normal", "weighted", "max"}, f"mbr_mode must be one of 'normal','weighted','max', got {mbr_mode!r}"

        # micro-batch size for candidate prompts during generation (PSP=="True")
        CHUNK = 5

        ####################################################################
        # 0) helpers
        ####################################################################
        class _JsonlDataset(Dataset):
            def __init__(self, path: str):
                self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]

            def __getitem__(self, idx): return self.rows[idx]
            def __len__(self):          return len(self.rows)

        def _build_dataloader(path: str):
            return DataLoader(
                _JsonlDataset(path),
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=self.args.dataloader_num_workers,
                collate_fn=lambda x: x,
            )

        def _mean_rouge(hyp: str, ref: str) -> float:
            r1_i, r2_i, rl_i = cal_rouge([hyp.replace("\n", "")], [ref.replace("\n", "")])
            return float((r1_i + r2_i + rl_i) / 3.0)

        # -------- MBR selection for classification (no ROUGE!) -------------
        def _mbr_select_accuracy_majority(labels: List[str]) -> int:
            """Return index of first occurrence of the majority label."""
            if not labels:
                return 0
            counts = Counter(labels)
            maj_label, _ = max(counts.items(), key=lambda kv: (kv[1], -labels.index(kv[0])))
            return labels.index(maj_label)

        def _parse_confidences(text: str) -> Dict[str, float]:
            """
            Try to parse label→confidence from evaluator text output.
            Supported patterns (examples):
            - 'positive: 0.72, negative: 0.28'
            - 'P(positive)=0.72; P(negative)=0.28'
            - 'confidence=0.72'  (applies to the *whole* response)
            Returns dict mapping (lowercased) label tokens to probabilities in [0,1].
            If nothing parseable is found, returns {}.
            """
            import re

            out: Dict[str, float] = {}

            # pattern: label: number   (comma- or newline-separated)
            for m in re.finditer(r'([A-Za-z0-9_\-\/ ]+)\s*[:=]\s*([01](?:\.\d+)?)', text):
                label = m.group(1).strip().lower()
                try:
                    p = float(m.group(2))
                except ValueError:
                    continue
                if 0.0 <= p <= 1.0:
                    out[label] = p

            # pattern: P(label)=number
            for m in re.finditer(r'P\(\s*([A-Za-z0-9_\-\/ ]+)\s*\)\s*=\s*([01](?:\.\d+)?)', text, flags=re.I):
                label = m.group(1).strip().lower()
                try:
                    p = float(m.group(2))
                except ValueError:
                    continue
                if 0.0 <= p <= 1.0:
                    out[label] = p

            # single 'confidence=number' if no distribution was found
            if not out:
                m = re.search(r'confidence\s*[:=]\s*([01](?:\.\d+)?)', text, flags=re.I)
                if m:
                    try:
                        p = float(m.group(1))
                        if 0.0 <= p <= 1.0:
                            # we don't know the label here; caller will attach to its own chosen label
                            out["__single__"] = p
                    except ValueError:
                        pass

            return out

        def _mbr_select_accuracy_weighted(labels: List[str], texts: List[str]) -> int:
            """
            Weighted vote: sum evaluator confidences for each label (case-insensitive).
            If no confidences are parseable, falls back to unweighted majority.
            """
            if not labels:
                return 0

            # accumulate weights per canonical label (lowercased)
            w_sum: Dict[str, float] = defaultdict(float)
            any_weight = False

            for lbl, txt in zip(labels, texts):
                lbl_key = lbl.strip().lower()
                confs = _parse_confidences(txt)
                if confs:
                    any_weight = True
                    if "__single__" in confs:
                        w_sum[lbl_key] += confs["__single__"]
                    else:
                        # if distribution exists, prefer the weight for *this* predicted label
                        if lbl_key in confs:
                            w_sum[lbl_key] += confs[lbl_key]
                        else:
                            # unseen label in distribution → add tiny epsilon to keep it in play
                            w_sum[lbl_key] += 1e-6
                else:
                    w_sum[lbl_key] += 1.0  # fallback weight

            if not any_weight:
                # identical to majority
                return _mbr_select_accuracy_majority(labels)

            # choose label with max total weight
            best_label, _ = max(w_sum.items(), key=lambda kv: kv[1])

            # return first candidate index that predicted best_label
            for idx, lbl in enumerate(labels):
                if lbl.strip().lower() == best_label:
                    return idx
            return 0

        def _mbr_select_accuracy_maxprob(labels: List[str], texts: List[str]) -> int:
            """
            Max-prob: pick the candidate whose evaluator confidence for its *own* predicted label is highest.
            If confidences are missing, falls back to first occurrence of majority label.
            """
            best_idx, best_p = 0, -1.0
            saw_prob = False
            for i, (lbl, txt) in enumerate(zip(labels, texts)):
                lbl_key = lbl.strip().lower()
                confs = _parse_confidences(txt)
                p = None
                if confs:
                    if "__single__" in confs:
                        p = confs["__single__"]
                    elif lbl_key in confs:
                        p = confs[lbl_key]
                if p is not None:
                    saw_prob = True
                    if p > best_p:
                        best_p = p
                        best_idx = i

            if saw_prob:
                return best_idx
            # fallback: majority
            return _mbr_select_accuracy_majority(labels)

        # -------- selection for generation-like tasks -----------------------
        def _mbr_select_pairwise_rouge(preds_for_item: List[str]) -> int:
            m = len(preds_for_item)
            if m <= 1:
                return 0
            best_j, best_u = 0, -1.0
            for j in range(m):
                u_sum, num = 0.0, 0
                for k in range(m):
                    if k == j:
                        continue
                    try:
                        u_sum += _mean_rouge(preds_for_item[j], preds_for_item[k])
                    except Exception:
                        pass
                    num += 1
                u_mean = u_sum / max(num, 1)
                if u_mean > best_u:
                    best_u, best_j = u_mean, j
            return best_j

        def _select_idx_for_item(metric_name: str, preds: List[str], eval_texts: List[str]) -> int:
            """
            metric_name routing:
            - "accuracy" → classification selector (mbr_mode specific)
            - "rouge", "sari" → pairwise-ROUGE (reference-free selection)
            """
            if metric_name == "accuracy":
                if mbr_mode == "normal":
                    return _mbr_select_accuracy_majority(preds)
                elif mbr_mode == "weighted":
                    return _mbr_select_accuracy_weighted(preds, eval_texts)
                else:  # "max"
                    return _mbr_select_accuracy_maxprob(preds, eval_texts)

            elif metric_name == "rouge":
                return _mbr_select_pairwise_rouge(preds)
            elif metric_name == "sari":
                # heuristic: use pairwise-ROUGE for selection, then report SARI vs gold
                return _mbr_select_pairwise_rouge(preds)
            else:
                raise ValueError(f"Unknown metric_name={metric_name!r}")

        ####################################################################
        # 1) single eval head
        ####################################################################
        def _single_eval(eval_dl, metric_name: str, file_suffix: str):
            """
            Runs one task head with MBR selection & candidate micro-batching.
            Classification: NO ROUGE is computed.
            """
            best_preds = []
            total, correct = 0, 0
            r1 = r2 = rl = sari_total = 0.0
            for _step, batch in enumerate(tqdm(eval_dl, desc=f"Eval {file_suffix}")):
                with torch.no_grad():
                    batch_size = len(batch)
                    observations = [
                        b['messages'][1]['content'].split("OBSERVATION: \n\n ")[-1]
                        for b in batch
                    ]

                    # Prepare per-item containers
                    preds_per_item: List[List[str]] = [[] for _ in range(batch_size)]
                    prompts_per_item: List[List[str]] = [[] for _ in range(batch_size)]
                    cand_texts_per_item: List[List[str]] = [[] for _ in range(batch_size)]
                    eval_texts_per_item: List[List[str]] = [[] for _ in range(batch_size)]  # raw evaluator outputs (for weights parsing)

                    if os.environ.get("PSP") == "False":
                        # Single enhanced prompt path -> one candidate per item
                        if "summarization task" in batch[0]['messages'][1]['content']:
                            prompt_for_generation = (
                                "Your task is to refine a base prompt for another model that performs a "
                                "summarization task. You will be given the base prompt that you should enhance. "
                                "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
                            )
                            prompt_for_generation += 'BASE PROMPT: \n\n"How would you rephrase that in a few words?"'
                        elif "simplification task" in batch[0]['messages'][1]['content']:
                            prompt_for_generation = (
                                "Your task is to refine a base prompt for another model that performs a "
                                "simplification task. You will be given the base prompt that you should enhance. "
                                "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
                            )
                            prompt_for_generation += "BASE PROMPT: \n\n Simplify the text."
                        elif "subjectivity classification task" in batch[0]['messages'][1]['content']:
                            prompt_for_generation = (
                                "Your task is to refine a base prompt for another model that performs a subjectivity classification task. "
                                "You will be given the base prompt that you should enhance. "
                                "Improve the instructions to enhance the model's performance. Return only the enhanced prompt."
                            )

                            prompt_for_generation += "BASE PROMPT: \n\n Please perform Subjectivity Classification task. Given the sentence, assign a label from [’subjective’,  ’objective’]. Return label only without any other text."
                        else:
                            raise Exception("!!!")

                        base_reqs_single = [{"messages": [
                            {"role": "system", "content": self.reasoning_system},
                            {"role": "user",   "content": prompt_for_generation}
                        ]}]
                        raw_single = self.engine.infer(base_reqs_single, self.request_config)
                        generated_prompt = extract_xml_answer(raw_single[0].choices[0].message.content.strip())

                        print(f"GENERATED PROMPT: {generated_prompt}")
                        exit(0)

                        frozen_reqs = []
                        for i in range(batch_size):
                            prompt = f"{generated_prompt}\n{observations[i]}"
                            frozen_reqs.append({"messages": [
                                {"role": "system", "content": self.base_prompt},
                                {"role": "user",   "content": prompt}
                            ]})
                            prompts_per_item[i].append(prompt)
                            cand_texts_per_item[i].append(generated_prompt)

                        frozen_outs = self.frozen_model.infer(frozen_reqs, self.request_config)
                        
                        for i, out in enumerate(frozen_outs):
                            pred_text = out.choices[0].message.content.strip()
                            preds_per_item[i].append(pred_text)
                            eval_texts_per_item[i].append(pred_text)  # evaluator raw text (for confidence parsing)

                    else:
                        base_reqs = [{
                            "messages": [
                                {"role": "system", "content": self.reasoning_system},
                                {"role": "user",   "content": b['messages'][1]['content']}
                            ]
                        } for b in batch]

                        for t in range(0, num_candidates, CHUNK):
                            chunk_size = min(CHUNK, num_candidates - t)

                            raw = self.engine.infer(base_reqs * chunk_size, self.request_config)

                            # Arrange per (j_in_chunk, i_in_batch)
                            cand_answers_chunk = [[None] * batch_size for _ in range(chunk_size)]
                            for j in range(chunk_size):
                                for i in range(batch_size):
                                    raw_idx = j * batch_size + i
                                    cand = extract_xml_answer(raw[raw_idx].choices[0].message.content.strip())
                                    cand_answers_chunk[j][i] = cand

                            # 2) Build frozen requests for this chunk and run evaluator once
                            frozen_reqs = []
                            idx_map = []  # (i, j_in_chunk)
                            for j in range(chunk_size):
                                for i in range(batch_size):
                                    prompt = cand_answers_chunk[j][i] + "\n" + observations[i].split("OBSERVATION: \n\n")[-1]
                                    frozen_reqs.append({"messages": [
                                        {"role": "system", "content": self.base_prompt},
                                        {"role": "user",   "content": prompt}
                                    ]})
                                    idx_map.append((i, j))
                                    prompts_per_item[i].append(prompt)
                                    cand_texts_per_item[i].append(cand_answers_chunk[j][i])

                            frozen_outs = self.frozen_model.infer(frozen_reqs, self.request_config)

                            # 3) Unpack evaluator predictions back per item
                            for (i, _j), out in zip(idx_map, frozen_outs):
                                pred_text = out.choices[0].message.content.strip()
                                preds_per_item[i].append(pred_text)
                                eval_texts_per_item[i].append(pred_text)  

                    # -------- selection & metric computation (NO ROUGE for classification) --------
                    for i in range(batch_size):
                        best_j = _select_idx_for_item(metric_name, preds_per_item[i], eval_texts_per_item[i])

                        ref   = batch[i]["solution"]
                        final_prompt      = prompts_per_item[i][best_j]
                        final_eval_output = preds_per_item[i][best_j]       # evaluator (frozen) prediction
                        chosen_candidate  = cand_texts_per_item[i][best_j]  # selected instruction candidate

                        if metric_name == "accuracy":
                            correct += int(final_eval_output == ref)
                            total   += 1
                            score_for_log = float(final_eval_output == ref)

                        elif metric_name == "rouge":
                            try:
                                
                                pred = final_eval_output 
                                r1_i, r2_i, rl_i = cal_rouge([pred.replace("\n", "")],[ref.replace("\n", "")])
                                r1 += r1_i; r2 += r2_i; rl += rl_i
                                correct += (r1_i + r2_i + rl_i) / 3.0
                                total   += 1
                                score_for_log = float((r1_i + r2_i + rl_i) / 3.0)
                            except Exception:
                                score_for_log = 0.0

                        else: 
                            try:
                                obs = observations[i].split("OBSERVATION: \n\n")[-1]
                                pred = final_eval_output.replace("\n", "")
                                if "OBSERVATION:" in pred:
                                    pred = pred.split("OBSERVATION:")[-1]
                                elif "OBSERVATION" in pred:
                                    pred = pred.split("OBSERVATION")[-1]
                                sari_score = calculate_sari(obs, pred, ref.replace("\n", ""))
                                #import pdb; pdb.set_trace()
                            except Exception as e:
                                print(e)
                                sari_score = 0.0
                            sari_total += sari_score
                            correct    += sari_score
                            total      += 1
                            score_for_log = float(sari_score)

                        best_preds.append({
                            "observation":       observations[i],
                            "frozen_prompt":     final_prompt,
                            "frozen_prediction": final_eval_output,
                            "prediction":        chosen_candidate,   # selected candidate instruction
                            "reference":         ref,
                            "score":             score_for_log,
                            "mbr_mode":          mbr_mode,
                        })

            # dump preds
            out_path = pathlib.Path(self.args.output_dir) / f"inference_log_{file_suffix}_{self.state.global_step}.json"
            out_path.write_text(json.dumps(best_preds, indent=2, ensure_ascii=False), encoding="utf-8")

            # final metric(s)
            if metric_name == "accuracy":
                return (correct / max(total, 1)), None
            elif metric_name == "rouge":
                return (correct / max(total, 1)), (r1 / max(total, 1), r2 / max(total, 1), rl / max(total, 1))
            else:
                return (sari_total / max(total, 1)), None

        ####################################################################
        # 2) main driver (same structure as before, but explicit branches)
        ####################################################################
        old_num_generations = self.num_generations
        self.num_generations = 1
        self.model.eval()

        eval_dl_gen = None
        rouge_score = 0.0

        # -- GEN (ROUGE) -----------------------------------------------------
        if "sum" in os.environ.get("INF", ""):
            sum_path = f"datasets/original/sum_test.jsonl"
            eval_dl_gen = _build_dataloader(sum_path)
            rouge_score, (r1, r2, rl) = _single_eval(eval_dl_gen, "rouge", "gen")
            print("ROUGE-1 SCORE:", r1)
            print("ROUGE-2 SCORE:", r2)
            print("ROUGE-L SCORE:", rl)
            self._log_wandb("best_eval_rouge1", r1)
            self._log_wandb("best_eval_rouge2", r2)
            self._log_wandb("best_eval_rougel", rl)
        else:
            r1 = r2 = rl = 0.0

        # -- SIM (SARI) ------------------------------------------------------
        valset = "test" if os.environ.get("AUX") == "True" else "val"
        if os.environ.get("DATASET") == "gen":
            if "sim" in os.environ.get("INF", ""):
                sim_path = f"datasets/original/sim_{valset}.jsonl"
                eval_dl_sim = _build_dataloader(sim_path)
                sari_score, _ = _single_eval(eval_dl_sim, "sari", "sim")
                print("SARI:", sari_score)
                self._log_wandb("best_eval_sari", sari_score)
            else:
                sari_score = 0.0

            # -- classification heads (ACCURACY) ------------------------------
            if "subj" in os.environ.get("INF", ""):
                subj_path = f"datasets/original/subj_{valset}.jsonl"
                acc_subj, _ = _single_eval(_build_dataloader(subj_path), "accuracy", "subj")
                print("ACC SUBJ", acc_subj)
                self._log_wandb("best_eval_accuracy_subj", acc_subj)
            else:
                acc_subj = 0.0

            if "gsm8k" in os.environ.get("INF", ""):
                #gsm8k_path = f"datasets/original/gsm8k_{valset}.jsonl"
                gsm8k_path = "datasets/original/gsm8k_test.jsonl"
                acc_gsm8k, _ = _single_eval(_build_dataloader(gsm8k_path), "accuracy", "gsm8k")
                self._log_wandb("best_eval_accuracy_gsm8k", acc_gsm8k)
                print("ACC GSM8K", acc_gsm8k)
            else:
                acc_gsm8k = 0.0

            if "mr" in os.environ.get("INF", ""):
                acc_mr, _ = _single_eval(_build_dataloader(f"datasets/original/mr_{valset}.jsonl"), "accuracy", "mr")
                print("MR ACC:", acc_mr)
                self._log_wandb("best_eval_accuracy_mr", acc_mr)
            else:
                acc_mr = 0.0

            if "cr" in os.environ.get("INF", ""):
                acc_cr, _ = _single_eval(_build_dataloader(f"datasets/original/cr_{valset}.jsonl"), "accuracy", "cr")
                print("CR ACC:", acc_cr)
                self._log_wandb("best_eval_accuracy_cr", acc_cr)
            else:
                acc_cr = 0.0

            if any(k in os.environ.get("INF", "") for k in ("sst2", "sst-2")):
                acc_sst2, _ = _single_eval(_build_dataloader(f"datasets/original/sst-2_{valset}.jsonl"), "accuracy", "sst-2")
                print("SST2 ACC:", acc_sst2)
                self._log_wandb("best_eval_accuracy_sst2", acc_sst2)
            else:
                acc_sst2 = 0.0

            if any(k in os.environ.get("INF", "") for k in ("sst5", "sst-5")):
                acc_sst5, _ = _single_eval(_build_dataloader(f"datasets/original/sst5_{valset}.jsonl"), "accuracy", "sst-5")
                print("SST5 ACC:", acc_sst5)
                self._log_wandb("best_eval_accuracy_sst5", acc_sst5)
            else:
                acc_sst5 = 0.0

            if "news" in os.environ.get("INF", ""):
                acc_news, _ = _single_eval(_build_dataloader(f"datasets/original/news_{valset}.jsonl"), "accuracy", "news")
                print("NEWS ACC:", acc_news)
                self._log_wandb("best_eval_accuracy_news", acc_news)
            else:
                acc_news = 0.0

            if "trec" in os.environ.get("INF", ""):
                acc_trec, _ = _single_eval(_build_dataloader(f"datasets/original/trec_{valset}.jsonl"), "accuracy", "trec")
                self._log_wandb("best_eval_accuracy_trec", acc_trec)
                print("ACC TREC:", acc_trec)
            else:
                acc_trec = 0.0
        else:
            sari_score = acc_subj = 0.0
            acc_gsm8k = acc_mr = acc_cr = acc_sst2 = acc_sst5 = acc_news = acc_trec = 0.0

        if os.environ.get("AUX") == "True":
            exit()

        self.num_generations = old_num_generations

        num_samples = len(eval_dl_gen.dataset) if "sum" in os.environ.get("INF", "") and eval_dl_gen is not None else 0
        return EvalLoopOutput(
            predictions=[],
            label_ids=[],
            metrics={
                "eval_rouge1":   r1,
                "eval_rouge2":   r2,
                "eval_rougel":   rl,
                "eval_sari":     sari_score,
                "eval_acc_subj": acc_subj,
                "eval_reward":   rouge_score,
            },
            num_samples=num_samples
        )
