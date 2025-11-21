import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        print(self.finetuning_args)
        self.global_step = 0  # 初始化全局步数

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def fgsm_attack(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], epsilon: float) -> torch.Tensor:
        """
        L1-FGM: one-step perturbation along the L1-normalized gradient direction.
        This replaces the previous FGSM (L-inf sign) so that 'fgsm' in config now means L1-FGM.
        """
        # 1) 拿到输入嵌入，并创建可求梯度的副本（叶子节点）
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        perturbed = embeddings.clone().detach().requires_grad_(True)

        # 2) 前向 + 反向，拿到对 perturbed 的梯度（兼容 gradient checkpointing 的做法）
        outputs = model(
            inputs_embeds=perturbed,
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss

        if perturbed.grad is not None:
            perturbed.grad.zero_()
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)

        grad = perturbed.grad.detach()

        # 避免污染模型参数上的梯度
        for p in model.parameters():
            p.grad = None

        # 3) 可选：mask 掉 padding 位置的梯度
        if "attention_mask" in inputs:
            grad = grad * inputs["attention_mask"].unsqueeze(-1).to(grad.dtype)

        # 4) 逐样本 L1 归一化：grad_unit 的 L1 范数为 1
        B = grad.size(0)
        grad_flat = grad.view(B, -1)
        eps_num = torch.finfo(grad_flat.dtype).eps
        l1_norm = grad_flat.abs().sum(dim=1, keepdim=True).clamp_min(eps_num)
        grad_unit = (grad_flat / l1_norm).view_as(grad)

        # 5) 沿 L1 单位“方向”前进一步，步长为 epsilon
        epsilon = torch.as_tensor(epsilon, dtype=perturbed.dtype, device=perturbed.device)
        perturbed = perturbed + epsilon * grad_unit
        return perturbed

    def pgd_l2_attack(
            self,
            model: torch.nn.Module,
            inputs: Dict[str, torch.Tensor],
            epsilon: float,
            alpha: float,
            num_steps: int,
    ) -> torch.Tensor:
        """
        L2-PGD: 每步用 L2 归一化的梯度前进 alpha，然后对 (emb - emb0) 做逐样本 L2 投影到半径 epsilon 的球内。
        """
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        # 随机初始化到 L2 球内（可选：这里用 0 初始化更稳，想要随机可自行替换）
        perturbed = embeddings.clone().detach().requires_grad_(True)

        B = embeddings.size(0)

        for _ in range(num_steps):
            outputs = model(
                inputs_embeds=perturbed,
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = outputs.loss
            model.zero_grad()
            if perturbed.grad is not None:
                perturbed.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad = perturbed.grad  # [B, T, D]
                # ---- per-sample L2 normalize ----
                grad_flat = grad.view(B, -1)
                grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-8
                grad_unit = (grad_flat / grad_norm).view_as(grad)

                # 梯度上升一步
                perturbed = perturbed + alpha * grad_unit

                # ---- project to L2 ball (radius=epsilon) per sample ----
                delta = (perturbed - embeddings).view(B, -1)
                delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True) + 1e-8
                # 对超出半径的样本缩放到边界，没超出的样本缩放因子为1
                scale = torch.clamp(epsilon / delta_norm, max=1.0)
                delta = (delta * scale).view_as(perturbed)

                perturbed = (embeddings + delta).detach().requires_grad_(True)

        return perturbed

    def pgd_attack(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], epsilon: float, alpha: float,
                   num_steps: int) -> torch.Tensor:
        embeddings = model.get_input_embeddings()(inputs["input_ids"])  # 确保 perturbed_embeddings 是叶子节点
        perturbed_embeddings = embeddings + torch.empty_like(embeddings).uniform_(-self.finetuning_args.epsilon, self.finetuning_args.epsilon)
        perturbed_embeddings = embeddings.clone().detach().requires_grad_(True)

        for step in range(num_steps):
            outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"])

            # 获取 loss 张量
            loss = outputs.loss
            model.zero_grad()
            loss.backward()  # 对损失张量调用 backward()

            with torch.no_grad():  # 禁用梯度计算，提高速度
                data_grad = perturbed_embeddings.grad.data
                # 生成新的 perturbed_embeddings
                perturbed_embeddings = perturbed_embeddings + alpha * data_grad.sign()
                perturbation = torch.clamp(perturbed_embeddings - embeddings, -epsilon, epsilon)
                perturbed_embeddings = (embeddings + perturbation).detach().requires_grad_(True)  # 再次确保是叶子节点

        return perturbed_embeddings

    def fgm_attack(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], epsilon: float) -> torch.Tensor:
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        perturbed = embeddings.clone().detach().requires_grad_(True)

        outputs = model(inputs_embeds=perturbed,
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"])
        loss = outputs.loss

        # 关键改动：用 backward() 而不是 autograd.grad（兼容 gradient checkpointing）
        # 先清空可能残留的梯度
        if perturbed.grad is not None:
            perturbed.grad.zero_()
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=False)  # 不要传 inputs 参数

        # 取到对 perturbed 的梯度
        grad = perturbed.grad.detach()

        # 避免污染模型参数的梯度（把参数上的梯度清掉）
        for p in model.parameters():
            p.grad = None

        # 可选：mask 掉 padding 的梯度
        if "attention_mask" in inputs:
            grad = grad * inputs["attention_mask"].unsqueeze(-1).to(grad.dtype)

        # 逐样本 L2 归一化
        B = grad.size(0)
        grad_flat = grad.view(B, -1)
        eps_num = torch.finfo(grad_flat.dtype).eps
        grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True).clamp_min(eps_num)
        grad_unit = (grad_flat / grad_norm).view_as(grad)

        epsilon = torch.as_tensor(epsilon, dtype=perturbed.dtype, device=perturbed.device)
        perturbed = perturbed + epsilon * grad_unit
        return perturbed

    def random_attack(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], epsilon: float) -> torch.Tensor:
        """
        Applies a fixed magnitude perturbation (epsilon) with random positive or negative direction to the input embeddings.

        Args:
            model: The model being attacked.
            inputs: The inputs to the model.
            epsilon: The magnitude of the perturbation (fixed value).

        Returns:
            The perturbed input embeddings with random positive or negative direction and fixed magnitude.
        """
        embeddings = model.get_input_embeddings()(inputs["input_ids"])
        # Generate a tensor of 1s and -1s randomly to decide positive or negative perturbation
        sign_matrix = torch.randint(0, 2, embeddings.shape, device=embeddings.device) * 2 - 1

        # Apply the fixed magnitude epsilon with random signs
        random_perturbation = epsilon * sign_matrix.float()  # Positive or negative epsilon

        # Add the random positive or negative perturbation to the embeddings
        perturbed_embeddings = embeddings + random_perturbation
        perturbed_embeddings = perturbed_embeddings.to(torch.float16)
        return perturbed_embeddings
            
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Regular forward pass with labels
        outputs = model(**inputs)
        loss = outputs.loss

        # 控制进行扰动的样本比例
        perturbation_sample_ratio = self.finetuning_args.perturbation_sample_ratio

        # 获取输入的 batch size
        batch_size = inputs["input_ids"].size(0)
        
        # 生成一个掩码，用于确定哪些样本进行扰动，哪些不进行
        sample_mask = torch.rand(batch_size) < perturbation_sample_ratio
        
        # 打印原始的 loss
        print(f"Original loss: {loss.item()}")
        # 只对部分样本进行扰动，其他样本保持不变
        if sample_mask.any():
            perturbed_inputs = {k: v[sample_mask] for k, v in inputs.items()}
            

            # Adversarial training on the selected samples
            epsilon = self.finetuning_args.epsilon
            print(epsilon)
            if self.finetuning_args.attack_method == 'fgsm':
                perturbed_embeddings = self.fgsm_attack(model, perturbed_inputs, epsilon)
            elif self.finetuning_args.attack_method == 'pgd':
                alpha = self.finetuning_args.pgd_step_size
                num_steps = self.finetuning_args.pgd_steps
                perturbed_embeddings = self.pgd_attack(model, perturbed_inputs, epsilon, alpha, num_steps)
            elif self.finetuning_args.attack_method == 'fgm':
                perturbed_embeddings = self.fgm_attack(model, perturbed_inputs, epsilon)
            elif self.finetuning_args.attack_method == 'pgd_l2':
                alpha = self.finetuning_args.pgd_step_size
                num_steps = self.finetuning_args.pgd_steps
                perturbed_embeddings = self.pgd_l2_attack(model, perturbed_inputs, epsilon, alpha, num_steps)

            elif self.finetuning_args.attack_method == 'random':
                perturbed_embeddings = self.random_attack(model, inputs, self.finetuning_args.epsilon)  # Use random attack
            else:
                raise ValueError(f"Unknown attack method: {self.finetuning_args.attack_method}")

            # 计算有扰动样本的 loss
            perturbed_outputs = model(inputs_embeds=perturbed_embeddings, attention_mask=perturbed_inputs["attention_mask"],labels=perturbed_inputs["labels"])
            adversarial_loss = perturbed_outputs.loss

            # 打印对抗训练的 loss
            print(f"Adversarial loss: {adversarial_loss.item()}")

            # 将扰动样本的 loss 与非扰动样本的 loss 合并
            total_loss = (loss * (~sample_mask).sum() + adversarial_loss * sample_mask.sum()) / batch_size
        else:
            total_loss = loss

        # Apply gradient accumulation if necessary
        if self.args.gradient_accumulation_steps > 1:
            total_loss = total_loss / self.args.gradient_accumulation_steps

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        return total_loss.detach()
            
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))

            writer.write("\n".join(res))
