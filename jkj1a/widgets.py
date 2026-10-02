"""Standard ipywidgets controls; no Colab-only JavaScript or display backend."""
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
def show_relu_single():
    """Show one shifted ReLU unit with a movable breakpoint."""
    import io
    import matplotlib.pyplot as plt

    def make_png(a):
        x = np.linspace(-1.5, 2.5, 600)
        y = np.maximum(x - a, 0)
        fig, ax = plt.subplots(figsize=(8, 3), facecolor="white", constrained_layout=True)
        ax.set_facecolor("white")
        ax.plot(x, y, color="#1f77b4", linewidth=3)
        ax.axvline(a, color="#d62728", linestyle="--", label=f"a = {a:.2f}")
        ax.set(xlim=(-1.5, 2.5), ylim=(-0.2, 2.6), xlabel="x", ylabel="ReLU(x - a)")
        ax.legend()
        ax.grid(alpha=0.3)
        for spine in ax.spines.values():
            spine.set_color("black")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor="white")
        plt.close(fig)
        return buffer.getvalue()

    slider = widgets.FloatSlider(value=0.5, min=-1, max=2, step=0.05,
                                 description="", continuous_update=True)
    slider_row = widgets.HBox([
        widgets.HTML("<span style='color: #d62728; font-weight: bold; width: 18px'>a</span>"),
        slider,
    ], layout=widgets.Layout(justify_content="center", width="100%"))
    image = widgets.Image(format="png", layout=widgets.Layout(width="850px"))
    formula = widgets.HTML(
        "<div style='font-size: 24px; text-align: center; background-color: white; "
        "color: black; padding: 6px'><code style='background-color: white; color: black'>"
        "ReLU(x - <span style='color: #d62728'>a</span>) = "
        "max(0, x - <span style='color: #d62728'>a</span>)</code></div>"
    )

    def update(change=None):
        image.value = make_png(slider.value)

    slider.observe(update, names="value")
    update()
    display(widgets.VBox([formula, slider_row, image]))

def show_relu_fit_exercise(n_hiddens=4, target="triangle"):
    """Interactive exercise: fit a target with a fixed number of hidden ReLU units."""
    import io
    import matplotlib.pyplot as plt

    if not 1 <= n_hiddens <= 32:
        raise ValueError("n_hiddens must be between 1 and 32")
    targets = {"linear", "triangle", "two_triangles", "trapezoid", "two_trapezoids", "sin"}
    if target not in targets:
        raise ValueError(f"target must be one of {sorted(targets)}")

    if target == "sin":
        x = np.linspace(0, np.pi, 600)
        target_y = np.sin(x)
        target_title = "sin(x), 0 ≤ x ≤ π"
    elif target == "linear":
        x = np.linspace(0, 2.5, 600)
        target_y = x
        target_title = "y = x, 0 ≤ x ≤ 2.5"
    else:
        x = np.linspace(-1.5, 2.5, 600)

        def triangle(center, width=0.8):
            return np.maximum(1 - np.abs(x - center) / width, 0)

        def trapezoid(center, half_top=0.25, half_base=0.5):
            y = np.minimum((x - (center - half_base)) / (half_base - half_top), 1)
            y = np.minimum(y, ((center + half_base) - x) / (half_base - half_top))
            return np.clip(y, 0, 1)

        if target == "triangle":
            target_y, target_title = triangle(0), "Triangle (center 0)"
        elif target == "two_triangles":
            target_y, target_title = triangle(0) + triangle(1), "Two triangles (centers 0, 1)"
        elif target == "trapezoid":
            target_y, target_title = trapezoid(0), "Trapezoid (center 0)"
        else:
            target_y, target_title = trapezoid(0) + trapezoid(1), "Two trapezoids (centers 0, 1)"

    colors = [plt.cm.turbo(i / max(n_hiddens - 1, 1)) for i in range(n_hiddens)]
    weights = [widgets.FloatSlider(value=0, min=-4, max=4, step=0.05,
                                   description="", continuous_update=True,
                                   layout=widgets.Layout(width="170px"))
               for _ in range(n_hiddens)]
    shifts = [widgets.FloatSlider(value=0, min=-3.5, max=3.5, step=0.05,
                                  description="", continuous_update=True,
                                  layout=widgets.Layout(width="170px"))
              for _ in range(n_hiddens)]
    weight_rows, shift_rows, terms = [], [], []
    for i, color in enumerate(colors):
        color_hex = "#%02x%02x%02x" % tuple(int(255 * v) for v in color[:3])
        weight_rows.append(widgets.HBox([
            widgets.HTML(f"<span style='color: {color_hex}; font-weight: bold; width: 28px'>w{i + 1}</span>"),
            weights[i]
        ], layout=widgets.Layout(align_items="center")))
        shift_rows.append(widgets.HBox([
            widgets.HTML(f"<span style='color: {color_hex}; font-weight: bold; width: 28px'>a{i + 1}</span>"),
            shifts[i]
        ], layout=widgets.Layout(align_items="center")))
        terms.append(
            f"<span style='color: black'><span style='color: {color_hex}'>w{i + 1}</span> "
            f"ReLU(x - <span style='color: {color_hex}'>a{i + 1}</span>)</span>"
        )
    rows = [widgets.HBox([weight_rows[i], shift_rows[i]],
                          layout=widgets.Layout(align_items="center"))
            for i in range(n_hiddens)]
    controls = widgets.VBox(rows, layout=widgets.Layout(align_items="center", width="100%"))
    formula = widgets.HTML(
        "<div style='font-size: 18px; line-height: 1.5; text-align: center; "
        "overflow-wrap: anywhere; background-color: white; color: black; padding: 6px'>"
        + " + ".join(terms) + "</div>"
    )
    image = widgets.Image(format="png", layout=widgets.Layout(width="850px"))
    error_label = widgets.HTML(
        layout=widgets.Layout(width="100%", text_align="center", margin="6px 0")
    )

    def update(_=None):
        y = np.zeros_like(x)
        for weight, shift in zip(weights, shifts):
            y += weight.value * np.maximum(x - shift.value, 0)
        error = np.max(np.abs(y - target_y))
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="white", constrained_layout=True)
        ax.set_facecolor("white")
        ax.plot(x, target_y, color="#bbbbbb", linewidth=3, linestyle="--", label="target")
        ax.plot(x, y, color="#1f77b4", linewidth=2.5, label="your function")
        ax.set(xlim=(x[0], x[-1]), xlabel="x", ylabel="f(x)", title=target_title)
        ax.legend()
        ax.grid(alpha=0.3)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor="white")
        plt.close(fig)
        image.value = buffer.getvalue()
        error_label.value = (
            f"<div style='text-align: center; font-size: 18px; color: #222'>"
            f"最大絶対誤差: {error:.4f}</div>"
        )

    for slider in weights + shifts:
        slider.observe(update, names="value")
    update()
    display(widgets.VBox([formula, controls, image, error_label],
                         layout=widgets.Layout(align_items="center", width="100%")))


def show_relu_triangle():
    """Show how three shifted ReLU units form a triangular bump."""
    import io
    import matplotlib.pyplot as plt

    def make_png(a, b, c):
        x = np.linspace(-0.2, 1.6, 600)
        parts = [
            np.maximum(x - a, 0),
            -2 * np.maximum(x - b, 0),
            np.maximum(x - c, 0),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True,
                                 facecolor="white")
        for ax in axes:
            ax.set_facecolor("white")
            ax.tick_params(colors="black")
            for spine in ax.spines.values():
                spine.set_color("black")
        colors = ["#d62728", "#2ca02c", "#ff7f0e"]
        for part, label, color in zip(
            parts, ["ReLU(x - a)", "-2 ReLU(x - b)", "ReLU(x - c)"], colors
        ):
            axes[0].plot(x, part, color=color, label=label)
        axes[0].legend()
        axes[0].set(title="Three contributions", xlim=(-0.2, 1.6), ylim=(-1.2, 1.2))
        axes[1].plot(x, sum(parts), color="#1f77b4", linewidth=3)
        axes[1].set(title="Sum", xlim=(-0.2, 1.6), ylim=(-1.2, 1.2))
        for ax in axes:
            ax.set_xlabel("x")
            ax.grid(alpha=0.3)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor="white")
        plt.close(fig)
        return buffer.getvalue()

    controls = [
        widgets.FloatSlider(min=0, max=0.8, step=0.02, value=0.2,
                            description="", continuous_update=True),
        widgets.FloatSlider(min=0.1, max=1.1, step=0.02, value=0.5,
                            description="", continuous_update=True),
        widgets.FloatSlider(min=0.3, max=1.5, step=0.02, value=0.8,
                            description="", continuous_update=True),
    ]
    image = widgets.Image(format="png", layout=widgets.Layout(width="1000px"))

    def update(_=None):
        image.value = make_png(*(control.value for control in controls))

    for control in controls:
        control.observe(update, names="value")
    formula = widgets.HTML(
        "<div style='font-size: 24px; text-align: center; margin: 8px 0; width: 100%; "
        "background-color: white; color: black; padding: 6px'>"
        "<code style='background-color: white; color: black'>"
        "ReLU(x - <span style='color: #d62728'>a</span>) - 2 ReLU(x - "
        "<span style='color: #2ca02c'>b</span>) + ReLU(x - "
        "<span style='color: #ff7f0e'>c</span>)</code></div>"
    )
    update()
    clear_output(wait=True)
    colors = ["#d62728", "#2ca02c", "#ff7f0e"]
    labels = [widgets.HTML(
        f"<span style='color: {color}; font-weight: bold; width: 18px'>{name}</span>"
    ) for name, color in zip(("a", "b", "c"), colors)]
    control_rows = [widgets.HBox(
        [label, control],
        layout=widgets.Layout(justify_content="center", width="100%"),
    ) for label, control in zip(labels, controls)]
    display(widgets.VBox([formula, *control_rows, image]))


def show_relu_approximation(n_segments=4):
    import io
    import matplotlib.pyplot as plt

    def target(x):
        return 0.3 * np.sin(3 * np.pi * x) + 0.6 * x**2

    if n_segments not in (2, 4, 8, 16, 32):
        raise ValueError("n_segments must be one of 2, 4, 8, 16, 32")
    width = widgets.SelectionSlider(options=[2, 4, 8, 16, 32], value=n_segments,
                                    description="Width", continuous_update=True)
    image = widgets.Image(format="png", layout=widgets.Layout(width="900px"))

    def update(change=None):
        segments = width.value
        knots = np.linspace(0, 1, segments + 1)
        values = target(knots)
        slopes = np.diff(values) / np.diff(knots)
        coefficients = np.r_[slopes[0], np.diff(slopes)]
        x = np.linspace(0, 1, 1000)
        hidden = np.maximum(x[:, None] - knots[:-1], 0)
        prediction = values[0] + hidden @ coefficients
        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="white", constrained_layout=True)
        ax.set_facecolor("white")
        ax.plot(x, target(x), label="Target")
        ax.plot(x, prediction, label=f"ReLU MLP: width {segments}")
        ax.scatter(knots, values, s=25)
        ax.set(xlabel="x", ylabel="f(x)",
               title=f"Max error on displayed grid: {np.max(abs(prediction-target(x))):.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor="white")
        plt.close(fig)
        image.value = buffer.getvalue()

    width.observe(update, names="value")
    update()
    display(widgets.VBox([width, image],
                         layout=widgets.Layout(align_items="center", width="100%")))

