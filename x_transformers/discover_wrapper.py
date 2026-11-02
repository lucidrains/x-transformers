# DISCOVER: The Emergent Symbolic Structure of Artificial Neural Networks

# McCoy et al. https://arxiv.org/abs/2608.29530

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import einsum, Tensor
from torch.nn import Embedding, Linear, Parameter, SiLU, Module, Sequential

from torch_einops_utils import batched_index_select, lens_to_mask

from einops import rearrange

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.x_transformers import TransformerWrapper, exists, default

ROLE_SCHEMES = ('bidirectional', 'right_to_left', 'left_to_right', 'wickel', 'bag_of_words')

def num_roles_for_scheme(
    scheme: str,
    *,
    max_seq_len: int,
    num_fillers: int
) -> int:
    if scheme in ('left_to_right', 'right_to_left'):
        return max_seq_len
    if scheme == 'bidirectional':
        return max_seq_len * (max_seq_len + 1) // 2
    if scheme == 'wickel':
        return (num_fillers + 1) ** 2
    return 1

def role_ids_for_tokens(
    tokens: Tensor,
    lengths: Tensor,
    scheme: str,
    *,
    max_seq_len: int,
    num_fillers: int
) -> Tensor:
    """positional role schemes (roles shared across sequence lengths)"""
    device = tokens.device
    seq_len = tokens.shape[1]

    positions = rearrange(torch.arange(seq_len, device = device), 'n -> 1 n')
    valid = lens_to_mask(lengths, seq_len)
    lengths = rearrange(lengths, 'b -> b 1')
    from_right = lengths - 1 - positions

    if scheme == 'left_to_right':
        roles = positions
    elif scheme == 'right_to_left':
        roles = from_right
    elif scheme == 'bidirectional':
        pair_sum = positions + from_right
        roles = pair_sum * (pair_sum + 1) // 2 + positions
    elif scheme == 'wickel':
        boundary = num_fillers
        prev_letter = torch.where(positions == 0, torch.full_like(positions, boundary), torch.roll(tokens, 1, dims = 1))
        next_letter = torch.where(positions == lengths - 1, torch.full_like(positions, boundary), torch.roll(tokens, -1, dims = 1))
        roles = prev_letter * (num_fillers + 1) + next_letter
    else:
        roles = torch.zeros_like(positions)

    return roles.masked_fill(~valid, 0)

class TPREncoder(Module):
    """TPR fitted to minimize MSE against some target encoding; optional MLP readout

    note: role-scheme contrasts (which scheme the network respects) are only diagnostic
    with a linear readout - an MLP readout tests expressibility, not structure
    """

    def __init__(
        self,
        *,
        dim: int,
        num_fillers: int,
        num_roles: int,
        filler_dim: int = 24,
        role_dim: int = 24,
        mlp_dim: int | None = None
    ):
        super().__init__()
        self.dim = dim
        self.filler_emb = Embedding(
            num_fillers,
            filler_dim
        )
        self.role_emb = Embedding(
            num_roles,
            role_dim
        )
        self.binding = Parameter(torch.zeros(dim, filler_dim, role_dim))
        self.bias = Parameter(torch.zeros(dim))

        torch.nn.init.normal_(self.filler_emb.weight, std = 0.05)
        torch.nn.init.normal_(self.role_emb.weight, std = 0.05)

        self.mlp = None
        if exists(mlp_dim):
            self.mlp = Sequential(
                Linear(dim, mlp_dim),
                SiLU(),
                Linear(mlp_dim, dim)
            )

    def state(
        self,
        tokens: Tensor,
        role_ids: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        """binding state per sequence - the sum over role-filler terms, before the readout"""
        bindings = einsum(
            'o f g, F f -> F o g',
            self.binding,
            self.filler_emb.weight
        )
        per_pair = bindings[tokens]
        role = self.role_emb(role_ids)
        contributions = einsum(
            'b n o g, b n g -> b n o',
            per_pair,
            role
        )

        if exists(mask):
            contributions = contributions.masked_fill(~rearrange(mask, 'b n -> b n 1'), 0.)

        return contributions.sum(dim = 1) + self.bias

    def readout(
        self,
        state: Tensor
    ) -> Tensor:
        """map the binding state to the target encoding space (identity unless an MLP is used)"""
        return self.mlp(state) if exists(self.mlp) else state

    def fit(
        self,
        tokens: Tensor,
        role_ids: Tensor,
        target_encodings: Tensor,
        *,
        mask: Tensor | None = None,
        steps: int = 2000,
        batch_size: int = 1024,
        lr: float = 3e-3,
        ema_decay: float = 0.9,
        optim: Callable = torch.optim.Adam
    ) -> Tensor:
        """fit by MSE against (frozen) encodings; returns the running minibatch MSE as a diagnostic"""
        num_samples, max_seq_len = tokens.shape
        assert role_ids.shape == tokens.shape, 'role ids must be one per position'
        assert target_encodings.shape == (num_samples, self.dim), f'target encodings must be of shape {(num_samples, self.dim)}'

        mask = default(
            mask,
            torch.ones(
                num_samples,
                max_seq_len,
                device = tokens.device,
                dtype = torch.bool
            )
        )

        optimizer = optim(self.parameters(), lr = lr)
        running_loss = 0.

        for _ in range(steps):
            indices = torch.randint(
                0,
                num_samples,
                (batch_size,),
                device = tokens.device
            )

            optimizer.zero_grad()
            predictions = self(
                tokens[indices],
                role_ids[indices],
                mask[indices]
            )
            loss = F.mse_loss(
                predictions,
                target_encodings[indices].detach()
            )
            loss.backward()
            optimizer.step()

            running_loss = ema_decay * running_loss + (1 - ema_decay) * loss.item()

        return running_loss

    def forward(
        self,
        tokens: Tensor,
        role_ids: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        if exists(mask):
            tokens = tokens.masked_fill(~mask, 0)
            role_ids = role_ids.masked_fill(~mask, 0)

        assert (
            (tokens.min() >= 0) and (tokens.max() < self.filler_emb.num_embeddings)
        ), 'tokens must be dense filler ids in [0, num_fillers); signal padding via `mask`'
        assert (
            (role_ids.min() >= 0) and (role_ids.max() < self.role_emb.num_embeddings)
        ), 'role ids out of range for this role scheme'

        return self.readout(self.state(tokens, role_ids, mask))

class DiscoverDecoder(Module):
    """token-level transformer as a bottleneck: input sequence -> one encoding vector -> output sequence"""

    def __init__(
        self,
        *,
        decoder: TransformerWrapper,
        num_fillers: int,
        bos_id: int,
        pad_id: int,
        max_seq_len: int,
        enc_pos: int | str = 'last'
    ):
        super().__init__()
        self.decoder = decoder
        self.num_fillers = num_fillers
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.enc_pos = enc_pos

    @property
    def dim(self) -> int:
        return self.decoder.attn_layers.dim

    def encode(
        self,
        tokens: Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        """encoding of each input sequence - hidden state of the transformer at the chosen position"""
        batch, seq_len = tokens.shape
        device = tokens.device

        mask = default(
            mask,
            torch.ones(
                batch,
                seq_len,
                device = device,
                dtype = torch.bool
            )
        )

        assert mask.dtype == torch.bool, 'mask must be a boolean tensor'
        mask = mask.bool()

        logits, intermediates = self.decoder(
            tokens,
            mask = mask,
            return_intermediates = True
        )
        hiddens = intermediates.last_layer_hiddens

        pos = mask.sum(dim = -1) - 1 if self.enc_pos == 'last' else torch.full(
            (batch,),
            int(self.enc_pos),
            device = device,
            dtype = torch.long
        )

        encodings = batched_index_select(
            hiddens,
            rearrange(pos, 'b -> b 1'),
            dim = 1
        )
        return rearrange(encodings, 'b 1 d -> b d')

    @torch.no_grad()
    def decode(
        self,
        encodings: Tensor,
        lengths: Tensor | None = None,
        *,
        eos_id: int | None = None,
        max_len: int | None = None
    ) -> Tensor:
        """
        greedily decode from the (possibly edited) encodings, prepended as embeddings to the
        autoregressive wrapper; terminate each row either at its exact `lengths` or at the first
        `eos_id` token (the rest of the row is padded)
        """
        assert exists(lengths) ^ exists(eos_id), 'provide either `lengths` or `eos_id`'

        batch = encodings.shape[0]
        device = encodings.device
        total_len = lengths.max().item() if exists(lengths) else default(max_len, self.max_seq_len)

        assert total_len <= self.max_seq_len

        prompts = torch.full((batch, 1), self.bos_id, device = device)
        prepend_embeds = rearrange(encodings, 'b d -> b 1 d')
        autoregressive = AutoregressiveWrapper(self.decoder)

        outputs = autoregressive.generate(
            prompts,
            total_len,
            eos_token = eos_id,
            temperature = 0.,
            cache_kv = False,
            prepend_embeds = prepend_embeds,
            prepend_mask = torch.ones(batch, 1, device = device, dtype = torch.bool)
        )

        outputs = F.pad(outputs, (0, total_len - outputs.shape[-1]), value = self.pad_id)

        if exists(eos_id):
            is_eos = outputs == eos_id
            lengths = torch.where(
                is_eos.any(dim = -1),
                is_eos.float().argmax(dim = -1) + 1,
                torch.full((batch,), total_len, device = device)
            )

        return outputs.masked_fill(
            rearrange(torch.arange(total_len, device = device), 't -> 1 t') >= rearrange(lengths, 'b -> b 1'),
            self.pad_id
        )

    @torch.no_grad()
    def surgery(
        self,
        tokens: Tensor,
        edited_tokens: Tensor,
        role_ids_old: Tensor,
        role_ids_new: Tensor,
        tpr: TPREncoder
    ) -> Tensor:
        """
        constituent surgery: retake the target encoding and add the TPR prediction for the edited
        sequence, minus the prediction for the original (exact for any readout; with a linear
        readout this is the paper's per-pair edit), then decode; role assignments are supplied by
        the caller (adjacent pair roles also change under `wickel`)
        """
        device = tokens.device

        assert edited_tokens.shape == tokens.shape == role_ids_old.shape == role_ids_new.shape, 'one token and role id per position for both sequences'

        mask = tokens != self.pad_id
        assert (
            (mask == (edited_tokens != self.pad_id)).all()
        ), 'edits must fall inside each sequence (never on padded positions)'
        assert (
            (tokens.masked_select(mask).min() >= 0) and (tokens.masked_select(mask).max() < self.num_fillers)
        ), 'tokens must be dense filler ids within their length'
        assert (
            (edited_tokens.masked_select(mask).min() >= 0) and (edited_tokens.masked_select(mask).max() < self.num_fillers)
        ), 'edited tokens must be dense filler ids'

        encodings = self.encode(
            tokens,
            mask
        )

        state_old = tpr.state(
            tokens.masked_fill(~mask, 0),
            role_ids_old,
            mask
        )
        state_new = tpr.state(
            edited_tokens.masked_fill(~mask, 0),
            role_ids_new,
            mask
        )

        edits = tpr.readout(state_new) - tpr.readout(state_old)

        return self.decode(
            encodings + edits,
            lengths = mask.sum(dim = -1)
        )

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None
    ) -> Tensor:
        """encode the input sequences; if targets are given, teacher-force decode them from E and return logits"""
        batch = tokens.shape[0]
        device = tokens.device

        encodings = self.encode(tokens)

        if not exists(targets):
            return encodings

        decode_ids = torch.cat(
            (torch.full((batch, 1), self.bos_id, device = device), targets[:, :-1]),
            dim = 1
        )
        prepend_embeds = rearrange(encodings, 'b d -> b 1 d')

        logits = self.decoder(
            decode_ids,
            prepend_embeds = prepend_embeds,
            prepend_mask = torch.ones(batch, 1, device = device, dtype = torch.bool)
        )

        return logits[:, 1:]
