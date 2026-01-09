import torch
import torch.nn.functional as F


def compute_fishr_penalty(
    logits,
    labels,
    classifier,
    num_envs=2
):

    device = logits.device
    batch_size = logits.size(0)


    env_ids = torch.arange(batch_size, device=device) % num_envs


    grads = []

    for env in range(num_envs):
        idx = env_ids == env
        if idx.sum() <= 1:
            continue

        loss_env = F.cross_entropy(logits[idx], labels[idx])

        grad_env = torch.autograd.grad(
            loss_env,
            classifier.parameters(),
            create_graph=True,
            retain_graph=True
        )


        grad_env = torch.cat([g.reshape(-1) for g in grad_env])
        grads.append(grad_env)


    if len(grads) <= 1:
        return torch.tensor(0.0, device=device)

    grads = torch.stack(grads)  # [num_envs, P]


    fishr_penalty = grads.var(dim=0).mean()

    return fishr_penalty
