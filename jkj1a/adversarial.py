"""Pixel-space attacks and evaluation for Day 2."""
import torch
from torch.nn import functional as F


def ifgsm(model, x, y, *, eps=8/255, alpha=2/255, n_steps=10, targeted=False):
    """L-infinity I-FGSM; x is in [0,1], y is the target for targeted attacks."""
    if eps < 0 or alpha < 0 or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError('eps and alpha must be nonnegative; n_steps must be a positive integer')
    was_training = model.training
    model.eval()
    original = x.detach()
    adv = original.clone()
    try:
        for _ in range(n_steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(model(adv), y)
            grad, = torch.autograd.grad(loss, adv)
            direction = -1 if targeted else 1
            adv = adv.detach() + direction * alpha * grad.sign()
            adv = original + (adv - original).clamp(-eps, eps)
            adv = adv.clamp(0, 1).detach()
    finally:
        model.train(was_training)
    return adv


def evaluate_attack(model, loader, *, attack=ifgsm, eps=8/255, alpha=2/255,
                    n_steps=10, max_batches=8):
    """Report clean/attacked accuracy and success among initially correct cases."""
    if max_batches is not None and max_batches < 1:
        raise ValueError('max_batches must be positive or None')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total = clean_correct = adv_correct = successful = 0
    try:
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                clean_ok = model(x).argmax(1) == y
            adv = attack(model, x, y, eps=eps, alpha=alpha, n_steps=n_steps)
            with torch.no_grad():
                adv_ok = model(adv).argmax(1) == y
            total += len(y)
            clean_correct += clean_ok.sum().item()
            adv_correct += adv_ok.sum().item()
            successful += (clean_ok & ~adv_ok).sum().item()
    finally:
        model.train(was_training)
    return {'n': total, 'clean_accuracy': clean_correct/total,
            'adversarial_accuracy': adv_correct/total,
            'initially_correct': clean_correct,
            'success_rate': successful/clean_correct if clean_correct else None}


def show_attack(model, x, adv, labels, *, index=0):
    import matplotlib.pyplot as plt
    was_training = model.training
    model.eval()
    with torch.no_grad():
        scores = model(torch.stack((x[index], adv[index]))).softmax(1).cpu()
    model.train(was_training)
    before, after = x[index].detach().cpu(), adv[index].detach().cpu()
    delta = after - before
    scale = max(delta.abs().max().item(), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for ax, picture in zip(axes, (before, .5 + delta/(2*scale), after)):
        ax.imshow(picture.permute(1, 2, 0).clamp(0, 1)); ax.axis('off')
    for ax, score, title in zip((axes[0], axes[2]), scores, ('Original', 'Attacked')):
        value, label = score.max(0)
        ax.set_title(f'{title}\n{labels[label.item()]} ({value:.3f})')
    axes[1].set_title(f'Perturbation (rescaled)\nmax |delta| = {delta.abs().max():.5f}')
    plt.show(); plt.close(fig)
