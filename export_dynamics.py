"""Export the curves used by the paper and the README.

Reads the epoch logs of the three reported pairs and writes:

    figures/dynamics/{tag}_{4t,ar}.csv             validation S(4) CE
    figures/dynamics/{tag}_{4t,ar}_mark.csv        reported checkpoint on that curve
    figures/dynamics/{tag}_{4t,ar}_train.csv       training CE
    figures/dynamics/{tag}_{4t,ar}_train_mark.csv  the same checkpoint on the training curve
    figures/dynamics/{tag}_{4t,ar}_val.csv         validation CE, the logged window or selection loss
    figures/dynamics/{tag}_{4t,ar}_val_mark.csv    the same checkpoint on that curve
    docs/h4l2_s4_ce.svg                            validation S(4) for H=4, L=2
    docs/h4l2_train_ce.svg                         training CE for H=4, L=2, one axis per model
    docs/h4l2_val_ce.svg                           validation CE for H=4, L=2, one axis per model

The marked epoch is the trainer's selected checkpoint (best_val_epoch on the
last epoch record). For the baseline that selection includes [EOS]. It is not
the minimum of content-token cross-entropy.

Training CE and validation CE are the same mean: for the 4-tensor model, every
target sentence in the window; for the baseline, the free-running loss on S(4),
including [EOS]. Each model is drawn on its own axis. The validation curve is
the quantity used to pick the checkpoint.

Run from anywhere:

    python export_dynamics.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "dynamics"
SVG_S4 = ROOT / "docs" / "h4l2_s4_ce.svg"
SVG_TRAIN = ROOT / "docs" / "h4l2_train_ce.svg"
SVG_VAL = ROOT / "docs" / "h4l2_val_ce.svg"

RUNS = [
    {
        "tag": "h2l2_s43",
        "4t": ROOT / "rocstories_T4max_x16/output_four_tensor_preLN/rocstories_T4N16_dx64_dt32_H2L2_do0.1_b32_s43.jsonl",
        "ar": ROOT / "rocstories_T4max_x16/output_baseline_ar_attention_preLN/rocstories_T4N16_dx64_dt32_H2L2_do0.1_b32_s43.jsonl",
    },
    {
        "tag": "h4l2_s43",
        "4t": ROOT / "rocstories_T4max_x16/output_four_tensor_preLN/rocstories_T4N16_dx64_dt32_H4L2_do0.1_b32_s43.jsonl",
        "ar": ROOT / "rocstories_T4max_x16/output_baseline_ar_attention_preLN/rocstories_T4N16_dx64_dt32_H4L2_do0.1_b32_s43.jsonl",
    },
    {
        "tag": "h4l3_s42",
        "4t": ROOT / "rocstories_T4max_x16/output_four_tensor_preLN/rocstories_T4N16_dx64_dt32_H4L3_do0.1_b32_s42.jsonl",
        "ar": ROOT / "rocstories_T4max_x16/output_baseline_ar_attention_preLN/rocstories_T4N16_dx64_dt32_H4L3_do0.1_b32_s42.jsonl",
    },
]

X_MIN = 0.5
X_MAX = 50.5
S4_Y_MIN = 5.4
S4_Y_MAX = 8.8
# Separate training axes. The 4-tensor loss averages every target sentence.
# The baseline loss is on S(4) only, and its epoch 1 lies above 16.5.
TRAIN_4T_Y_MIN = 1.0
TRAIN_4T_Y_MAX = 2.7
TRAIN_AR_Y_MIN = 5.0
TRAIN_AR_Y_MAX = 16.5
VAL_4T_Y_MIN = 1.28
VAL_4T_Y_MAX = 1.48
VAL_AR_Y_MIN = 5.0
VAL_AR_Y_MAX = 15.0


def load_epochs(path):
    s4_rows = []
    train_rows = []
    val_rows = []
    selected = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record") != "epoch":
                continue
            epoch = int(record["epoch"])
            s4_rows.append((epoch, float(record["val_s_t_ce"])))
            train_rows.append((epoch, float(record["train_loss"])))
            val_rows.append((epoch, float(record["val_loss"])))
            selected = int(record["best_val_epoch"])
    if not s4_rows or selected is None:
        raise SystemExit(f"no epoch records in {path}")
    if selected not in dict(s4_rows):
        raise SystemExit(f"selected epoch {selected} is missing from {path}")
    return s4_rows, train_rows, val_rows, selected


def write_csv(path, column, rows):
    lines = [f"epoch,{column}"]
    lines.extend(f"{epoch},{value:.10f}" for epoch, value in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def box_x(epoch, x0, width):
    return x0 + (epoch - X_MIN) / (X_MAX - X_MIN) * width


def box_y(value, y0, height, ymin, ymax):
    return (y0 + height) - (value - ymin) / (ymax - ymin) * height


def path_d(rows, x0, y0, width, height, ymin, ymax, dashed):
    parts = []
    for index, (epoch, value) in enumerate(rows):
        x = box_x(epoch, x0, width)
        y = box_y(value, y0, height, ymin, ymax)
        parts.append(f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}")
    dash = ' stroke-dasharray="6 3.5"' if dashed else ""
    width_px = "1.6" if dashed else "1.8"
    return f'<path d="{" ".join(parts)}" fill="none" stroke="#222" stroke-width="{width_px}"{dash}/>'


def y_tick_labels(values, x0, width, y0, height, ymin, ymax, digits=1):
    lines = []
    for value in values:
        y = box_y(value, y0, height, ymin, ymax)
        label = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
        lines.append(
            f'<line x1="{x0:.0f}" y1="{y:.2f}" x2="{x0 + width:.0f}" y2="{y:.2f}" stroke="#e6e6e6"/>\n'
            f'<text x="{x0 - 8:.0f}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Times New Roman, Times, serif" font-size="13" fill="#222">{label}</text>'
        )
    return "\n".join(lines)


def x_tick_labels(x0, width, y_text):
    lines = []
    for epoch in (1, 10, 20, 30, 40, 50):
        x = box_x(epoch, x0, width)
        lines.append(
            f'<text x="{x:.2f}" y="{y_text}" text-anchor="middle" '
            f'font-family="Times New Roman, Times, serif" font-size="13" fill="#222">{epoch}</text>'
        )
    return "\n".join(lines)


def curve_panel(rows, mark_epoch, x0, y0, width, height, ymin, ymax, clip_id, dashed):
    bottom = y0 + height
    value = dict(rows)[mark_epoch]
    cx = box_x(mark_epoch, x0, width)
    cy = box_y(value, y0, height, ymin, ymax)
    return f'''<clipPath id="{clip_id}"><rect x="{x0:.0f}" y="{y0:.0f}" width="{width:.0f}" height="{height:.0f}"/></clipPath>
<line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x0:.0f}" y2="{bottom:.0f}" stroke="#222"/>
<line x1="{x0:.0f}" y1="{bottom:.0f}" x2="{x0 + width:.0f}" y2="{bottom:.0f}" stroke="#222"/>
<g clip-path="url(#{clip_id})">
{path_d(rows, x0, y0, width, height, ymin, ymax, dashed)}
</g>
<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="#ffffff" stroke="#222" stroke-width="1.6"/>'''


def write_s4_svg(four, baseline, four_epoch, baseline_epoch):
    x0, y0, width, height = 52.0, 36.0, 650.0, 284.0
    four_map = dict(four)
    base_map = dict(baseline)
    fx, fy = box_x(four_epoch, x0, width), box_y(four_map[four_epoch], y0, height, S4_Y_MIN, S4_Y_MAX)
    bx, by = box_x(baseline_epoch, x0, width), box_y(base_map[baseline_epoch], y0, height, S4_Y_MIN, S4_Y_MAX)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360" viewBox="0 0 720 360" role="img">
<rect width="100%" height="100%" fill="#ffffff"/>
{y_tick_labels((5.5, 6.5, 7.5, 8.5), x0, width, y0, height, S4_Y_MIN, S4_Y_MAX)}
{x_tick_labels(x0, width, 338)}
<clipPath id="s4"><rect x="52" y="36" width="650" height="284"/></clipPath>
<line x1="52" y1="36" x2="52" y2="320" stroke="#222"/>
<line x1="52" y1="320" x2="702" y2="320" stroke="#222"/>
<g clip-path="url(#s4)">
{path_d(baseline, x0, y0, width, height, S4_Y_MIN, S4_Y_MAX, True)}
{path_d(four, x0, y0, width, height, S4_Y_MIN, S4_Y_MAX, False)}
</g>
<circle cx="{fx:.2f}" cy="{fy:.2f}" r="4" fill="#ffffff" stroke="#222" stroke-width="1.6"/>
<circle cx="{bx:.2f}" cy="{by:.2f}" r="4" fill="#ffffff" stroke="#222" stroke-width="1.6"/>
<text x="377.0" y="354" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">epoch</text>
<text x="16" y="178.0" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222" transform="rotate(-90 16 178.0)">S(4) cross-entropy</text>
<line x1="552" y1="50" x2="580" y2="50" stroke="#222" stroke-width="1.8"/>
<text x="586" y="54" font-family="Times New Roman, Times, serif" font-size="13" fill="#222">4-tensor</text>
<line x1="552" y1="68" x2="580" y2="68" stroke="#222" stroke-width="1.6" stroke-dasharray="6 3.5"/>
<text x="586" y="72" font-family="Times New Roman, Times, serif" font-size="13" fill="#222">baseline</text>
</svg>
'''
    SVG_S4.parent.mkdir(parents=True, exist_ok=True)
    SVG_S4.write_text(svg, encoding="utf-8", newline="\n")


def write_train_svg(four, baseline, four_epoch, baseline_epoch):
    y0, height = 36.0, 240.0
    left = (48.0, 290.0)
    right = (400.0, 290.0)
    panels = []
    panels.append(y_tick_labels((1.2, 1.6, 2.0, 2.4), left[0], left[1], y0, height, TRAIN_4T_Y_MIN, TRAIN_4T_Y_MAX))
    panels.append(x_tick_labels(left[0], left[1], 294))
    panels.append(curve_panel(four, four_epoch, left[0], y0, left[1], height, TRAIN_4T_Y_MIN, TRAIN_4T_Y_MAX, "t4", False))
    panels.append(y_tick_labels((6, 10, 14), right[0], right[1], y0, height, TRAIN_AR_Y_MIN, TRAIN_AR_Y_MAX))
    panels.append(x_tick_labels(right[0], right[1], 294))
    panels.append(curve_panel(baseline, baseline_epoch, right[0], y0, right[1], height, TRAIN_AR_Y_MIN, TRAIN_AR_Y_MAX, "tar", True))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="340" viewBox="0 0 720 340" role="img">
<rect width="100%" height="100%" fill="#ffffff"/>
{chr(10).join(panels)}
<text x="193" y="22" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">4-tensor</text>
<text x="545" y="22" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">baseline</text>
<text x="360" y="328" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">epoch</text>
</svg>
'''
    SVG_TRAIN.write_text(svg, encoding="utf-8", newline="\n")


def write_val_svg(four, baseline, four_epoch, baseline_epoch):
    y0, height = 36.0, 240.0
    left = (56.0, 280.0)
    right = (408.0, 280.0)
    panels = []
    panels.append(y_tick_labels((1.30, 1.35, 1.40, 1.45), left[0], left[1], y0, height, VAL_4T_Y_MIN, VAL_4T_Y_MAX, digits=2))
    panels.append(x_tick_labels(left[0], left[1], 294))
    panels.append(curve_panel(four, four_epoch, left[0], y0, left[1], height, VAL_4T_Y_MIN, VAL_4T_Y_MAX, "v4", False))
    panels.append(y_tick_labels((6, 10, 14), right[0], right[1], y0, height, VAL_AR_Y_MIN, VAL_AR_Y_MAX))
    panels.append(x_tick_labels(right[0], right[1], 294))
    panels.append(curve_panel(baseline, baseline_epoch, right[0], y0, right[1], height, VAL_AR_Y_MIN, VAL_AR_Y_MAX, "var", True))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="340" viewBox="0 0 720 340" role="img">
<rect width="100%" height="100%" fill="#ffffff"/>
{chr(10).join(panels)}
<text x="196" y="22" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">4-tensor</text>
<text x="548" y="22" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">baseline</text>
<text x="360" y="328" text-anchor="middle" font-family="Times New Roman, Times, serif" font-size="14" fill="#222">epoch</text>
</svg>
'''
    SVG_VAL.write_text(svg, encoding="utf-8", newline="\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves = {}
    for run in RUNS:
        for kind in ("4t", "ar"):
            s4_rows, train_rows, val_rows, selected = load_epochs(run[kind])
            s4_map = dict(s4_rows)
            train_map = dict(train_rows)
            val_map = dict(val_rows)
            write_csv(OUT / f"{run['tag']}_{kind}.csv", "s4_ce", s4_rows)
            write_csv(OUT / f"{run['tag']}_{kind}_mark.csv", "s4_ce", [(selected, s4_map[selected])])
            write_csv(OUT / f"{run['tag']}_{kind}_train.csv", "train_ce", train_rows)
            write_csv(OUT / f"{run['tag']}_{kind}_train_mark.csv", "train_ce", [(selected, train_map[selected])])
            write_csv(OUT / f"{run['tag']}_{kind}_val.csv", "val_ce", val_rows)
            write_csv(OUT / f"{run['tag']}_{kind}_val_mark.csv", "val_ce", [(selected, val_map[selected])])
            print(
                f"{run['tag']} {kind}: selected e{selected} "
                f"s4={s4_map[selected]:.3f} train={train_map[selected]:.3f} val={val_map[selected]:.3f} "
                f"val e1={val_map[1]:.3f} valmin={min(val_rows, key=lambda item: item[1])[1]:.3f}"
            )
            curves[(run["tag"], kind)] = (s4_rows, train_rows, val_rows, selected)
    four_s4, four_train, four_val, four_epoch = curves[("h4l2_s43", "4t")]
    base_s4, base_train, base_val, baseline_epoch = curves[("h4l2_s43", "ar")]
    write_s4_svg(four_s4, base_s4, four_epoch, baseline_epoch)
    write_train_svg(four_train, base_train, four_epoch, baseline_epoch)
    write_val_svg(four_val, base_val, four_epoch, baseline_epoch)
    print(f"wrote {SVG_S4.relative_to(ROOT)}")
    print(f"wrote {SVG_TRAIN.relative_to(ROOT)}")
    print(f"wrote {SVG_VAL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
