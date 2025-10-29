"""Pi05 PyTorch Implementation with Subtask Generation.

This module implements the Pi05 model in PyTorch, which extends Pi0 with subtask generation
capabilities. The model can autoregressively generate task descriptions before predicting actions.

Key differences from pi0_pytorch.py:
1. **Dual-Loss Training**: Combines cross-entropy loss (subtask generation) + MSE loss (flow matching)
2. **Subtask Generation**: Autoregressive token generation with KV caching
3. **AdaRMSNorm**: Uses adaptive normalization (time embedding injection) in action expert
4. **AR Attention**: Language tokens use causal attention masks instead of bidirectional
5. **No State Projection**: State is discretized and embedded in language tokens
6. **Left-to-Right Alignment**: Sequences are right-aligned for efficient autoregressive generation
7. **KV Cache Reuse**: Subtask generation KV cache is reused during action sampling

Architecture:
- PaliGemma (image + language encoder)
- Action Expert (Gemma with adaRMSNorm for time conditioning)
- Autoregressive subtask decoder
- Flow matching action predictor

Training:
- Loss = subtask_generation_loss + mean(flow_loss)
- token_loss_mask controls which tokens contribute to CE loss

Inference:
1. Generate subtask tokens autoregressively
2. Use generated subtask context for flow matching action prediction
"""
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma05_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned.
    
    Args:
        x: [batch, seq, dim] embeddings
        input_mask: [batch, seq] bool mask
        attn_mask: [batch, seq, seq] attention mask
    """
    batch_size = x.shape[0]
    results_x = []
    results_input_mask = []
    results_attn_mask = []
    
    for b in range(batch_size):
        # Find the actual sequence length (excluding padding)
        seqlen = torch.max(input_mask[b].float() * torch.arange(input_mask[b].shape[0], device=x.device)) + 1
        seqlen = seqlen.long()
        
        # Roll to right-align
        x_rolled = torch.roll(x[b], -seqlen.item(), dims=0)
        mask_rolled = torch.roll(input_mask[b], -seqlen.item(), dims=0)
        attn_rolled = torch.roll(attn_mask[b], (-seqlen.item(), -seqlen.item()), dims=(0, 1))
        
        results_x.append(x_rolled)
        results_input_mask.append(mask_rolled)
        results_attn_mask.append(attn_rolled)
    
    return torch.stack(results_x), torch.stack(results_input_mask), torch.stack(results_attn_mask)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI05Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.token_loss_mask,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # [pi05 subtask generation] AR attention for subtask generation
        num_lang_embs = lang_emb.shape[1]
        att_masks += [1] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions and timestep for action expert."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # time MLP (for adaRMS)
        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        adarms_cond = time_emb

        # Embed actions
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        embs.append(action_emb)

        bsize, action_time_dim = action_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # AR mask: first action token attends to prefix, rest attend causally
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None, real_action_dim=32) -> Tensor:
        """Compute both subtask generation loss and flow matching loss.
        
        Returns:
            Combined loss (subtask_generation_loss + mean(flow_loss))
        """
        images, img_masks, lang_tokens, lang_masks, state, token_loss_mask = self._preprocess_observation(
            observation, train=True
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 1. Subtask Generation Loss
        # Compute targets: predict next token
        vocab_size = self.paligemma_with_expert.paligemma.language_model.config.vocab_size
        targets = F.one_hot(lang_tokens[:, 1:], vocab_size).float()

        # Forward through PaliGemma
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )
        prefix_out = prefix_out[:, :-1]

        # Decode embeddings to logits
        logits = self.paligemma_with_expert.paligemma.language_model.lm_head(prefix_out[:, -targets.shape[1]:])
        logp = F.log_softmax(logits, dim=-1)

        # Compute CE loss on token targets
        loss_mask = token_loss_mask[:, 1:]
        token_pplx = torch.sum(targets * logp, dim=-1)
        subtask_generation_loss = -torch.sum(token_pplx * loss_mask, dim=-1) / torch.clamp(
            torch.sum(loss_mask, dim=-1), min=1
        )

        # 2. Flow Matching Loss
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

        # Attention mask for suffix attending to full prefix + suffix
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        def forward_func(suffix_embs, full_att_2d_masks_4d, position_ids, adarms_cond, past_key_values):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, suffix_embs, full_att_2d_masks_4d, position_ids, adarms_cond, past_key_values
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # Flow loss only on real action dimensions
        flow_loss = torch.mean(
            torch.square(v_t[:, :, :real_action_dim] - u_t[:, :, :real_action_dim]), dim=-1
        )

        return subtask_generation_loss + torch.mean(flow_loss, dim=-1)

    @torch.no_grad()
    def sample_low_level_task(
        self, images, img_masks, lang_tokens, lang_masks, max_decoding_steps=20, temperature=0.0, eos_token=1
    ):
        """Sample low-level task autoregressively."""
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        # Left to right align all input token sequences
        prefix_embs, prefix_pad_masks, prefix_att_2d_masks = left_to_right_align(
            prefix_embs, prefix_pad_masks, prefix_att_2d_masks
        )
        
        prefill_size = prefix_embs.shape[1]
        prefill_len = torch.sum(prefix_pad_masks, dim=-1)
        prefix_start = prefill_size - prefill_len

        # Pad attention mask for future tokens
        prefix_att_2d_masks = F.pad(prefix_att_2d_masks, (0, max_decoding_steps, 0, 0))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        last_token_embedding = prefix_out[:, -1:]
        last_logits = self.paligemma_with_expert.paligemma.language_model.lm_head(last_token_embedding)
        last_logits = F.log_softmax(last_logits, dim=-1)

        output_tokens = torch.zeros((batch_size, max_decoding_steps), dtype=torch.long, device=device)

        for step in range(max_decoding_steps):
            # Sample token
            if temperature > 0.0:
                token = torch.multinomial(torch.exp(last_logits[:, 0] / temperature), num_samples=1)
            else:
                token = torch.argmax(last_logits[:, 0], dim=-1, keepdim=True)

            output_tokens[:, step] = token[:, 0]

            # Check for early stopping
            has_eos = torch.any(token == eos_token, dim=-1)
            if torch.all(has_eos):
                break

            # Decode one step
            token_embedding = self.paligemma_with_expert.embed_language_tokens(token)
            token_embedding = token_embedding * math.sqrt(token_embedding.shape[-1])

            positions = (prefill_len + step)[:, None]

            # Create mask for current step: tokens can attend to all previous tokens
            # mask[b, i, j] = (j >= prefix_start[b]) and (j < prefill_size + step + 1)
            total_len = prefill_size + max_decoding_steps
            j_indices = torch.arange(total_len, device=device)[None, None, :]
            mask = (j_indices >= prefix_start[:, None, None]) & (j_indices < (prefill_size + step + 1))
            mask_4d = self._prepare_attention_masks_4d(mask)

            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=mask_4d,
                position_ids=positions,
                past_key_values=past_key_values,
                inputs_embeds=[token_embedding, None],
                use_cache=True,
                adarms_cond=[None, None],
            )

            last_token_embedding = prefix_out[:, -1:]
            last_logits = self.paligemma_with_expert.paligemma.language_model.lm_head(last_token_embedding)
            last_logits = F.log_softmax(last_logits, dim=-1)

        # Create full masks including generated tokens
        token_mask = output_tokens != 0
        full_pad_mask = torch.cat([prefix_pad_masks, token_mask], dim=1)
        
        num_generated = output_tokens.shape[1]
        full_att_mask = torch.cat(
            [prefix_att_masks, torch.ones(batch_size, num_generated, dtype=torch.bool, device=device)], dim=1
        )

        return output_tokens, past_key_values, full_pad_mask, full_att_mask

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> tuple[Tensor, Tensor]:
        """Generate subtasks then sample actions."""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state, _ = self._preprocess_observation(observation, train=False)

        # Generate low-level task
        output_tokens, past_key_values, prefix_pad_masks, prefix_att_masks = self.sample_low_level_task(
            images, img_masks, lang_tokens, lang_masks, max_decoding_steps=20, temperature=0.0, eos_token=1
        )

        # Now perform flow matching to generate actions
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt

        return x_t, output_tokens

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """Apply one denoising step.
        
        Note: The KV cache from subtask generation is reused but NOT updated,
        since suffix tokens (action_horizon > 1) are processed together.
        The transformers library automatically concatenates new K/V with cached K/V.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # Suffix tokens can attend to all prefix tokens
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        # Suffix tokens have causal attention among themselves
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        # Combine: [batch, suffix_len, prefix_len + suffix_len]
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position IDs: continue from where prefix ended
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        # use_cache=False: we don't update the KV cache during denoising
        # past_key_values from subtask generation are reused for prefix context
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
