# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "x-transformers",
#   "fire",
#   "tqdm",
#   "wandb",
#   "accelerate"
# ]
# ///

import random
from tqdm import tqdm
import fire

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import wandb
from accelerate import Accelerator

from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper, RMSNorm
from x_transformers.next_latent_wrapper import NextLatentWrapper

# constants

PAD_TOKEN   = 0
EDGE_TOKEN  = 1
START_TOKEN = 2
END_TOKEN   = 3
PATH_TOKEN  = 4

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_collate(batch):
    return pad_sequence(batch, batch_first = True, padding_value = PAD_TOKEN)

# dataset

class PathStarDataset(Dataset):
    def __init__(
        self,
        num_samples,
        num_branches = 7,
        nodes_per_branch = 7,
        max_node_id = 100,
        seed = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_branches = num_branches
        self.nodes_per_branch = nodes_per_branch
        self.max_node_id = max_node_id

        self.node_offset = 5
        self.vocab_size = max_node_id + self.node_offset

        self.rng = random.Random(seed)
        self.data = [torch.tensor(self.generate_sample(), dtype = torch.long) for _ in range(num_samples)]
        self.max_seq_len = max(len(seq) for seq in self.data)

    def generate_sample(self):
        num_nodes = 1 + self.num_branches * (self.nodes_per_branch - 1)
        assert num_nodes <= self.max_node_id, f"cannot sample {num_nodes} from {self.max_node_id} ids"

        sampled_ids = self.rng.sample(range(self.max_node_id), num_nodes)
        sampled_ids = [node_id + self.node_offset for node_id in sampled_ids]

        center_node, *rest_nodes = sampled_ids

        arms = []
        for i in range(self.num_branches):
            start_idx = i * (self.nodes_per_branch - 1)
            end_idx = start_idx + (self.nodes_per_branch - 1)
            arms.append(rest_nodes[start_idx:end_idx])

        edges = []
        for arm in arms:
            edges.append((center_node, arm[0]))
            edges.extend([(arm[i], arm[i + 1]) for i in range(len(arm) - 1)])

        self.rng.shuffle(edges)

        directed_edges = [(u, v) if self.rng.random() < 0.5 else (v, u) for u, v in edges]

        target_arm = self.rng.choice(arms)
        target_leaf_node = target_arm[-1]
        target_path = [center_node, *target_arm]

        seq = []
        for u, v in directed_edges:
            seq.extend([EDGE_TOKEN, u, v])

        seq.extend([START_TOKEN, center_node, END_TOKEN, target_leaf_node, PATH_TOKEN, *target_path])
        return seq

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# training

def train(
    num_branches: int = 3,
    nodes_per_branch: int = 3,
    max_node_id: int = 100,
    num_train: int = 20000,
    num_val: int = 1000,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.1,
    dim: int = 384,
    depth: int = 4,
    heads: int = 6,
    seed: int = 42,
    wandb_project: str = 'path_star',
    wandb_run_name: str | None = None,
    use_nextlat: bool = True,
    dynamics_type: str = 'residual',
    next_latent_loss_weight: float = 1.0,
    kl_loss_weight: float = 1.0,
    num_rollouts: int = 2,
    dynamic_rollout_loss_weight: bool = True,
    dynamic_loss_decay: float = 1.0,
    dynamic_loss_threshold: float = 0.5,
    max_grad_norm: float | None = 100.0
):
    wandb.init(project = wandb_project, name = wandb_run_name, config = locals())

    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(seed)
    random.seed(seed)

    accelerator.print(f"Generating datasets for G({num_branches}, {nodes_per_branch})...")

    train_dataset = PathStarDataset(num_train, num_branches, nodes_per_branch, max_node_id, seed = seed)
    val_dataset   = PathStarDataset(num_val, num_branches, nodes_per_branch, max_node_id, seed = seed + 1)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = pad_collate)
    val_loader   = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, collate_fn = pad_collate)

    accelerator.print(f"Vocab size: {train_dataset.vocab_size}, Max seq len: {train_dataset.max_seq_len}")
    accelerator.print(f"Using device: {device}")

    if use_nextlat:
        accelerator.print(f"Next latent is ENABLED (dynamics_type: {dynamics_type})")
    else:
        accelerator.print("Next latent is DISABLED. You can turn it on by passing `--use_nextlat=True`")

    # base model

    base_model = TransformerWrapper(
        num_tokens = train_dataset.vocab_size,
        max_seq_len = train_dataset.max_seq_len,
        attn_layers = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            rotary_pos_emb = True
        )
    )

    # optionally wrap with next-latent

    if use_nextlat:
        model = NextLatentWrapper(
            base_model,
            dim = dim,
            dynamics_type = dynamics_type,
            next_latent_loss_weight = next_latent_loss_weight,
            kl_loss_weight = kl_loss_weight,
            num_rollouts = num_rollouts,
            ignore_index = PAD_TOKEN,
            pad_value = PAD_TOKEN,
            dynamic_rollout_loss_weight = dynamic_rollout_loss_weight,
            dynamic_loss_decay = dynamic_loss_decay,
            dynamic_loss_threshold = dynamic_loss_threshold
        )
    else:
        model = AutoregressiveWrapper(
            base_model,
            pad_value = PAD_TOKEN,
            ignore_index = PAD_TOKEN
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.

        pbar = tqdm(train_loader, desc = f"Epoch {epoch}/{epochs} [Train]", disable = not accelerator.is_local_main_process)

        for batch in pbar:
            if use_nextlat:
                loss, loss_breakdown = model(batch, return_loss_breakdown = True)
                loss_dict = {k: v.item() for k, v in loss_breakdown._asdict().items() if getattr(v, 'item', None) is not None and v > 0.}
            else:
                loss = model(batch)
                loss_dict = {}

            accelerator.backward(loss)

            if accelerator.sync_gradients and exists(max_grad_norm):
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            pbar_dict = {k: f"{v:.2f}" for k, v in loss_dict.items()}
            pbar_dict['loss'] = f"{loss.item():.2f}" if use_nextlat else f"{loss.item():.4f}"
            pbar.set_postfix(**pbar_dict)

            if accelerator.is_main_process:
                log_dict = {'train/loss': loss.item(), 'train/step': step}
                log_dict.update({f'train/{k}_loss': v for k, v in loss_dict.items()})
                wandb.log(log_dict)

            step += 1

        avg_loss = total_loss / len(train_loader)
        accelerator.print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")

        if accelerator.is_main_process:
            wandb.log({'train/epoch_loss': avg_loss, 'epoch': epoch})

        # validation

        model.eval()
        correct_paths = 0
        total_paths = 0

        val_pbar = tqdm(val_loader, desc = f"Epoch {epoch}/{epochs} [Val]", disable = not accelerator.is_local_main_process)

        unwrapped_model = accelerator.unwrap_model(model)

        generator = AutoregressiveWrapper(unwrapped_model.net) if use_nextlat else unwrapped_model

        with torch.no_grad():
            for batch in val_pbar:
                for seq in batch:
                    path_mask = (seq == PATH_TOKEN)

                    if not path_mask.any():
                        continue

                    path_idx = path_mask.nonzero(as_tuple = True)[0][0].item()

                    prompt = seq[:path_idx + 1].unsqueeze(0)
                    target_path = seq[path_idx + 1:]
                    target_path = target_path[target_path != PAD_TOKEN]

                    generated = generator.generate(prompt, seq_len = len(target_path))

                    if torch.equal(generated[0], target_path):
                        correct_paths += 1

                    total_paths += 1

                val_pbar.set_postfix(acc = f"{(correct_paths / max(1, total_paths)) * 100:.1f}%")

        accuracy = correct_paths / max(1, total_paths)
        accelerator.print(f"Epoch {epoch} Val Path Accuracy: {accuracy * 100:.2f}% ({correct_paths}/{total_paths})")

        if accelerator.is_main_process:
            wandb.log({
                'val/accuracy': accuracy,
                'val/correct_paths': correct_paths,
                'val/total_paths': total_paths,
                'epoch': epoch
            })

if __name__ == '__main__':
    fire.Fire(train)
