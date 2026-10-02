"""Animation helpers for the Day 1 ReLU function experiments."""
from __future__ import annotations

import io
import ipywidgets as widgets
import matplotlib.pyplot as plt


def make_training_replay(history, model, x, target, *, show_sliders=True):
    """Create a widget for replaying a recorded Trainer history."""
    order = x.argsort()
    x, target = x[order], target[order]
    history = [(step, prediction[order], error, weights, shifts)
               for step, prediction, error, weights, shifts in history]
    colors = []
    parameter_widgets = []
    if show_sliders:
        colors = [plt.cm.turbo(i / max(model.weight.numel() - 1, 1))
                  for i in range(model.weight.numel())]
        terms = []
        for i, color in enumerate(colors):
            color_hex = "#%02x%02x%02x" % tuple(int(255 * v) for v in color[:3])
            terms.append(
                f"<span style='color: black'><span style='color: {color_hex}'>w{i+1}</span> "
                f"ReLU(x - <span style='color: {color_hex}'>a{i+1}</span>)</span>"
            )
        formula = widgets.HTML(
            "<div style='font-size: 18px; line-height: 1.5; text-align: center; "
            "overflow-wrap: anywhere; background-color: white; color: black; padding: 6px'>"
            + " + ".join(terms) + "</div>"
        )
        parameter_widgets.append(formula)
    image = widgets.Image(format="png", layout=widgets.Layout(width="850px"))
    frame = widgets.IntSlider(value=0, min=0, max=len(history)-1, step=1,
                              description="step", continuous_update=True,
                              layout=widgets.Layout(width="500px"))
    play = widgets.Play(value=0, min=0, max=len(history)-1, step=1, interval=120)
    stop = widgets.Button(description="Stop", layout=widgets.Layout(width="70px"))
    widgets.jslink((play, "value"), (frame, "value"))
    ws, shifts = [], []
    if show_sliders:
        ws = [widgets.FloatSlider(value=0, min=-4, max=4, step=0.01, description="", disabled=True,
                                  layout=widgets.Layout(width="170px")) for _ in range(model.weight.numel())]
        shifts = [widgets.FloatSlider(value=0, min=-3.5, max=3.5, step=0.01, description="", disabled=True,
                                      layout=widgets.Layout(width="170px")) for _ in range(model.shift.numel())]
        for sliders, index, minimum in ((ws, 3, 4), (shifts, 4, 3.5)):
            bound = max(minimum, max(frame[index].abs().max().item() for frame in history) * 1.05)
            for slider in sliders:
                slider.min, slider.max = -bound, bound
    low = min(target.min().item(), min(frame[1].min().item() for frame in history))
    high = max(target.max().item(), max(frame[1].max().item() for frame in history))
    padding = max((high - low) * 0.05, 0.1)
    rows = []
    for i, color in enumerate(colors if show_sliders else []):
        color_hex = "#%02x%02x%02x" % tuple(int(255 * v) for v in color[:3])
        wl = widgets.HTML(f"<span style='color:{color_hex};font-weight:bold;width:28px'>w{i+1}</span>")
        al = widgets.HTML(f"<span style='color:{color_hex};font-weight:bold;width:28px'>a{i+1}</span>")
        rows.append(widgets.HBox([widgets.HBox([wl, ws[i]]), widgets.HBox([al, shifts[i]])]))

    def render(change=None):
        step, prediction, error, weights, shift_values = history[frame.value]
        if show_sliders:
            for slider, value in zip(ws, weights.tolist()): slider.value = value
            for slider, value in zip(shifts, shift_values.tolist()): slider.value = value
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="white", constrained_layout=True)
        ax.set_facecolor("white")
        ax.plot(x, target, color="#bbbbbb", linestyle="--", linewidth=3, label="target")
        ax.plot(x, prediction, color="#1f77b4", linewidth=2.5, label="model")
        ax.set(xlim=(x.min().item(), x.max().item()), ylim=(low-padding, high+padding), xlabel="x", ylabel="f(x)")
        ax.set_title(f"step={step}, MSE={error:.5f}")
        ax.legend(); ax.grid(alpha=0.3)
        buffer = io.BytesIO(); fig.savefig(buffer, format="png", dpi=120, facecolor="white")
        plt.close(fig); image.value = buffer.getvalue()

    stop.on_click(lambda _: setattr(play, "playing", False))
    frame.observe(render, names="value"); render()
    controls = widgets.HBox([play, stop, frame], layout=widgets.Layout(justify_content="center", width="100%"))
    if show_sliders:
        parameter_widgets.append(widgets.VBox(rows, layout=widgets.Layout(align_items="center")))
    return widgets.VBox(parameter_widgets + [image, controls],
                        layout=widgets.Layout(align_items="center", width="100%"))



def _record_training_frame(trainer):
    import torch
    with torch.no_grad():
        prediction = torch.stack([trainer.model(x) for x, _ in trainer._animation_data])
        trainer.history.append((
            trainer._animation_epoch, prediction.detach().cpu().clone(),
            trainer.loss_fn(prediction, trainer._animation_data[:, 1]).item(),
            trainer.model.weight.detach().cpu().clone() if trainer._animation_show_sliders else None,
            trainer.model.shift.detach().cpu().clone() if trainer._animation_show_sliders else None,
        ))


def animate_step(method):
    """Record the state after an epoch without changing its return value."""
    from functools import wraps

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        if getattr(self, '_animation_active', False):
            self._animation_epoch += 1
            if self._animation_epoch % self._animation_interval == 0:
                _record_training_frame(self)
        return result
    return wrapped


def animate(method):
    """Capture the initial state and return a replay after training."""
    from functools import wraps
    from inspect import signature

    @wraps(method)
    def wrapped(self, *args, with_animation=True, show_sliders=True,
                animation_interval=None, **kwargs):
        arguments = signature(method).bind(self, *args, **kwargs)
        arguments.apply_defaults()
        self._animation_interval = (arguments.arguments['print_interval']
                                    if animation_interval is None else animation_interval)
        if not isinstance(self._animation_interval, int) or self._animation_interval <= 0:
            raise ValueError('animation_interval must be a positive integer')
        self._animation_epoch = 0
        self.history = []
        self._animation_active = with_animation
        self._animation_show_sliders = show_sliders
        try:
            if with_animation:
                # Keep plotting coordinates fixed even when training shuffles in place.
                order = self.data[:, 0].argsort()
                self._animation_data = self.data[order].detach().clone()
                _record_training_frame(self)
            result = method(self, *args, **kwargs)
            if with_animation:
                if self.history[-1][0] != self._animation_epoch:
                    _record_training_frame(self)
                print("アニメーション生成中...", flush=True)
                return make_training_replay(
                    self.history, self.model,
                    self._animation_data[:, 0].cpu(), self._animation_data[:, 1].cpu(),
                    show_sliders=show_sliders,
                )
            return result
        finally:
            self._animation_active = False
    return wrapped
