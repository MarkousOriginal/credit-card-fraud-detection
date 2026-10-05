"""README figures in the reference data-viz palette: light surface, hairline grid, ink-coloured text."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NAMES = {"dummy_prior": "No-skill baseline", "logistic_regression": "Logistic regression",
         "random_forest": "Random forest", "xgboost": "XGBoost"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "semibold"})


def axes(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelcolor=SECONDARY, pad=6)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    return fig, ax


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def model_comparison(metrics, chosen, path):
    """Validation average precision with bootstrap intervals; the chosen model is the only coloured one."""
    rows = metrics.sort_values("average_precision").reset_index(drop=True)
    fig, ax = axes((7.6, 3.4))
    ax.grid(axis="y", visible=False)
    picked = [(row.model, row.class_weighting) == chosen for row in rows.itertuples()]
    for i, row in rows.iterrows():
        color = BLUE if picked[i] else MUTED
        # Unclipped, so the no-skill dot at zero is drawn whole on the axis.
        ax.plot([row.ap_ci_low, row.ap_ci_high], [i, i], color=color, linewidth=2, solid_capstyle="round",
                clip_on=False)
        ax.plot(row.average_precision, i, "o", color=color, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, clip_on=False)
        if picked[i]:
            ax.annotate(f"{row.average_precision:.3f}  chosen", (row.ap_ci_high, i), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK, fontweight="semibold")
    labels = [NAMES[row.model] + (", class-weighted" if row.class_weighting == "balanced" else "")
              for row in rows.itertuples()]
    ax.set_yticks(range(len(rows)), labels)
    for label, emphasis in zip(ax.get_yticklabels(), picked):
        if emphasis:
            label.set(color=INK, fontweight="semibold")
    ax.set_xlim(0, 1)
    ax.set_title("Average precision on the validation period, with 95% bootstrap intervals",
                 loc="left", color=INK, pad=12)
    save(fig, path)


def savings_curves(curves, review_cost, path, labelled="Test"):
    """Share of fraud losses avoided against alert volume, one line per period, with the two cut-offs marked.

    Only the labelled period's markers carry values, the comparison the figure is about."""
    fig, ax = axes((7.6, 4.6))
    shapes = {"frozen": ("o", "Threshold frozen on validation"), "default": ("D", "Default 0.5 cut-off")}
    offsets = {"frozen": (0, -16), "default": (0, 10)}
    handles = []
    for (period, (curve, points)), color in zip(curves.items(), [BLUE, ORANGE]):
        shown = curve[(curve.alerts_per_hour >= 1) & (curve.alerts_per_hour <= 100)]
        ax.plot(shown.alerts_per_hour, shown.savings_share, color=color, linewidth=2, solid_joinstyle="round")
        handles.append(Line2D([], [], color=color, linewidth=2, label=f"{period} period"))
        end = shown.iloc[-1]
        ax.annotate(period, (end.alerts_per_hour, end.savings_share), xytext=(6, 0), textcoords="offset points",
                    va="center", color=SECONDARY)
        for kind, (x, share) in points.items():
            ax.plot(x, share, shapes[kind][0], color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            if period == labelled:
                ax.annotate(f"{share:.0%}", (x, share), xytext=offsets[kind], textcoords="offset points",
                            ha="center", va="center", color=INK, fontsize=9, fontweight="semibold")
    handles += [Line2D([], [], marker=marker, linestyle="", color=MUTED, markersize=7, label=label)
                for marker, label in shapes.values()]
    ax.set_xscale("log")
    ax.set_xlim(1, 140)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Alerts per hour (log scale)", color=SECONDARY)
    ax.set_title(f"Share of fraud losses avoided, net of a review cost of {review_cost:g} per alert",
                 loc="left", color=INK, pad=26)
    ax.text(0, 1.035, "Each point is a threshold; lower thresholds raise more alerts", transform=ax.transAxes,
            color=SECONDARY)
    ax.legend(handles=handles, loc="upper left", frameon=False, labelcolor=SECONDARY)
    save(fig, path)
