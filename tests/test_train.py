import torch

from arche.config import Llama32Config
from arche.model import Llama
from arche.train.dpo import dpo_loss, train_dpo_step
from arche.train.pretrain import train_pretrain_steps
from arche.train.sft import train_sft_step


def _tiny() -> Llama:
    cfg = Llama32Config(
        name="tiny",
        vocab_size=48,
        num_layers=1,
        embed_dim=32,
        intermediate_dim=64,
        num_heads=2,
        num_kv_heads=1,
        head_dim=16,
        max_seq_len=32,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
    )
    return Llama(cfg)


def test_pretrain_loss_decreases():
    torch.manual_seed(0)
    model = _tiny()
    # Repeat one batch so the optimizer has a stable target to overfit.
    batch = torch.randint(0, 48, (2, 8))
    history = train_pretrain_steps(model, [batch] * 8, lr=5e-2)
    assert len(history) == 8
    assert history[-1] < history[0]


def test_sft_step_runs():
    model = _tiny()
    ids = torch.randint(0, 48, (2, 10))
    prompt_lens = torch.tensor([3, 4])
    loss = train_sft_step(model, ids, prompt_lens, lr=1e-2)
    assert loss > 0


def test_dpo_loss_and_step():
    torch.manual_seed(0)
    policy = _tiny()
    reference = _tiny()
    reference.load_state_dict(policy.state_dict())
    chosen = torch.randint(0, 48, (2, 6))
    rejected = torch.randint(0, 48, (2, 6))
    # Synthetic logprob tensors for the pure loss helper.
    loss = dpo_loss(
        torch.zeros(2),
        torch.ones(2) * -1,
        torch.zeros(2),
        torch.zeros(2),
        beta=0.1,
    )
    assert loss.ndim == 0
    step_loss = train_dpo_step(policy, reference, chosen, rejected, lr=1e-3)
    assert isinstance(step_loss, float)
