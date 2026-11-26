"""Card figures for the Ling-3.0-tiny GGUF: banner, release-gate (A), ladder (B), sensitivity (C).

    .venv/bin/python -I scripts/figures.py --theme light|dark [--layout results/layout-<file>.tsv]
                                           [--record results/refinement-<id>.json] [--demo]

Writes figures/NAME-THEME.png (200 dpi) and .svg (text as paths), and figures/alt.md (numeric alt text for every
figure rendered). Every plotted value is read from a result file; nothing is typed in. Before rendering, this
theme's old outputs are deleted, so a skipped figure leaves no stale file. Specs: docs/figure-concept.md,
docs/figure-stats.md; colours scripts/style.py.

- banner: layout TSV of the scored GGUF (scripts/gguf_layout.py). It says "codes refined" only with --record, a
  scripts/refine_record.py JSON whose sha256 equals the layout's and whose checks all passed.
- release-gate (A): results/release-gate.tsv, validated by scripts/release_gate_report.py validate(); the verdict
  comes from that one rule. A TSV that fails validation stops the script.
- ladder (B): runs/gate-t/{bloomer,m2,m2r[,m2rd]}-{wiki512,ood512}.f32 (64 x 512 only), Q8-*.f32 reference; CIs via
  boot() of scripts/gate_t_report.py. m2rd is drawn only if adopted by its registered rule; otherwise it is reported
  as a tested branch in the footer and alt text. The last drawn step is the candidate that went to A.
- sensitivity (C): results/ablations.tsv (g_*_iq2_xxs vs q8_all), m2's attained types from results/layout-m2.tsv.
--demo renders A and B from synthetic numbers with a DEMO DATA watermark into figures/demo/ (git-ignored, layout
check only, never referenced by the cards).
"""
import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import release_gate_report as rg  # noqa: E402
import style  # noqa: E402
from q3_types import GROUPS  # noqa: E402  (anchored tensor-group regexes)

boot = rg.boot   # the gate-T bootstrap
NAMES = ["banner", "release-gate", "ladder", "sensitivity"]
LABEL = {"cand": "Ours", "bloomer": "bloomer010 IQ3_XXS", "bartowski": "bartowski Q2_K_L", "Q8": "Q8_0 reference"}
ALT = {}


def gb(b):
    return f"{b / 1e9:.2f} GB"


def save(fig, out, name, T):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}-{T['name']}.{ext}", dpi=200, facecolor=T["bg"])
    plt.close(fig)
    print(f"wrote {out / name}-{T['name']}.png/.svg")


def watermark(fig):
    fig.text(0.5, 0.5, "DEMO DATA", fontsize=64, color="#ef9a6a", alpha=0.35, rotation=28,
             ha="center", va="center", fontweight="bold", fontfamily=style.TITLE, zorder=100)


def C(k):
    return style.COMMON[k]


# ---------- banner ----------

def read_layout(path):
    rows, meta = [], {}
    for line in Path(path).read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0].startswith("#"):
            meta[f[0][1:]] = f[1]
        else:
            rows.append((f[0], f[1], int(f[2])))
    return rows, meta


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m[1]) if m else None


def layer_kinds(rows):
    """Layer -> 'mla' (has attn_kv_a_mqa), 'dense' (KDA layer with a non-expert ffn_gate), else 'kda'."""
    n = max(layer_of(r[0]) for r in rows if layer_of(r[0]) is not None) + 1
    kind = {i: "kda" for i in range(n)}
    for name, _, _ in rows:
        i = layer_of(name)
        if i is not None and "attn_kv_a_mqa" in name:
            kind[i] = "mla"
    for name, _, _ in rows:
        i = layer_of(name)
        if i is not None and re.search(r"\.ffn_gate\.weight$", name) and kind[i] == "kda":
            kind[i] = "dense"
    return kind


def share_group(name, typ, kind):
    if typ == "F32":
        return "F32 norms, router, conv"
    if "_exps." in name:
        return "routed experts"
    if "_shexp." in name:
        return "shared expert"
    if name == "token_embd.weight":
        return "embedding"
    if name == "output.weight":
        return "output"
    if re.search(r"\.ffn_(gate|up|down)\.weight$", name):
        return "dense FFN"
    return "MLA attention" if kind.get(layer_of(name)) == "mla" else "KDA attention"


def refinement(record, meta):
    if not record:
        return False
    rec = json.loads(Path(record).read_text())
    if rec.get("sha256") != meta.get("sha256"):
        raise SystemExit(f"{record}: sha256 does not match the layout's file ({meta.get('file')})")
    if not (rec.get("refined") and rec.get("equal_layout") and rec.get("scales_preserved")
            and rec.get("unchanged_outside_codes")):
        raise SystemExit(f"{record}: refinement checks did not all pass; banner will not say refined")
    return True


def banner(T, out, layout, record):
    if not Path(layout).exists():
        print(f"skip banner: {layout} missing (scripts/gguf_layout.py <gguf> {layout})")
        return
    rows, meta = read_layout(layout)
    refined = refinement(record, meta)
    file_bytes = int(meta["file_bytes"])
    kind = layer_kinds(rows)
    n = len(kind)
    per_layer = np.zeros(n)
    for name, _, b in rows:
        if layer_of(name) is not None:
            per_layer[layer_of(name)] += b
    groups = {}
    for name, typ, b in rows:
        g = groups.setdefault(share_group(name, typ, kind), {})
        g[typ] = g.get(typ, 0) + b
    total = sum(sum(v.values()) for v in groups.values())
    order_t = lambda g: "/".join(sorted(groups[g], key=lambda t: -groups[g][t]))  # noqa: E731
    exp_share = sum(groups["routed experts"].values()) / total
    n_mla = sum(v == "mla" for v in kind.values())
    n_kda = n - n_mla
    dense = [i for i, v in kind.items() if v == "dense"]

    fig = plt.figure(figsize=(10, 4.4))
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.imshow(style.gradient_image(style.BAND, style.BAND_POS, 400, 160, 135), extent=(0, 1, 0, 1), aspect="auto")
    bg.set_axis_off()
    tf = T["title_font"]
    fig.text(0.035, 0.84, "Ling-3.0-tiny GGUF", fontfamily=tf, fontweight="bold", fontsize=36, color="white", va="top")
    fig.text(0.035, 0.64, gb(file_bytes), fontsize=19, color=C("band_meta"), va="top")
    fig.text(0.035, 0.53, f"experts {order_t('routed experts')}" + (", codes refined" if refined else ""),
             fontsize=19, color=C("band_meta"), va="top")
    fig.text(0.035, 0.40, "github.com/me-daniel/ling-3.0-tiny-quant", fontsize=14, color=C("band_link"), va="top")

    ax = fig.add_axes([0.58, 0.47, 0.39, 0.38], zorder=1)
    ax.set_facecolor("none")
    top = per_layer.max() / 1e6
    for i in range(n):
        h = per_layer[i] / 1e6
        if kind[i] == "mla":
            style.gradient_rect(ax, i - 0.38, 0, 0.76, h, style.ACCENT, style.BAND_POS, angle=0, edge="white", lw=0.6)
        else:
            ax.add_patch(Rectangle((i - 0.38, 0), 0.76, h, facecolor=C(kind[i]), edgecolor=C("border"), lw=0.6,
                                   zorder=2))
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, top * 1.05)
    ax.set_axis_off()
    for i in range(n):
        if kind[i] != "kda" or i == n - 1:
            ax.text(i, -top * 0.04, str(i), ha="center", va="top", fontsize=13, color=C("band_meta"))
    ax.text(-0.7, top * 1.12, f"{n} layers", fontsize=14, color=C("band_meta"), va="bottom")
    lx = fig.add_axes([0.58, 0.27, 0.39, 0.10], zorder=1)
    lx.set_axis_off()
    lx.set_xlim(0, 1)
    lx.set_ylim(0, 1)
    lx.add_patch(Rectangle((0, 0.62), 0.03, 0.3, facecolor=C("kda"), edgecolor=C("border"), lw=0.6))
    lx.text(0.045, 0.77, f"{n_kda} KDA layers", va="center", fontsize=13, color="white")
    style.gradient_rect(lx, 0.44, 0.62, 0.03, 0.3, style.ACCENT, style.BAND_POS, angle=0, edge="white", lw=0.6)
    lx.text(0.485, 0.77, f"{n_mla} gated MLA layers", va="center", fontsize=13, color="white")
    if dense:
        lx.add_patch(Rectangle((0, 0.08), 0.03, 0.3, facecolor=C("dense"), edgecolor=C("border"), lw=0.6))
        lx.text(0.045, 0.23, f"incl. layer {', '.join(map(str, dense))}: KDA with dense FFN", va="center", fontsize=12,
                color=C("band_meta"))

    sx = fig.add_axes([0.035, 0.07, 0.935, 0.15], zorder=1)
    sx.set_axis_off()
    sx.set_xlim(0, 1)
    sx.set_ylim(0, 1)
    pale = {"routed experts": "#e9f6ff", "KDA attention": C("kda"), "MLA attention": style.ACCENT[1],
            "dense FFN": C("dense"), "shared expert": C("dense"), "embedding": C("dense"), "output": C("dense"),
            "F32 norms, router, conv": "#c9d3ef"}
    x, small = 0, []
    for g in sorted(groups, key=lambda g: -sum(groups[g].values())):
        w = sum(groups[g].values()) / total
        if g == "routed experts":
            sx.add_patch(Rectangle((x, 0.35), w, 0.6, facecolor=pale[g], lw=2.4 if refined else 0.8, zorder=3,
                                   edgecolor=style.WARM[0] if refined else C("border")))
            sx.text(x + 0.012, 0.65, f"routed experts  {order_t(g)}  {w:.0%}" +
                    ("  ·  codes refined" if refined else ""), va="center", fontsize=15, color=C("oxford"),
                    fontweight="medium", zorder=4)
        else:
            sx.add_patch(Rectangle((x, 0.35), w, 0.6, facecolor=pale[g], edgecolor=C("oxford"), lw=0.5, zorder=2))
            small.append((f"{g} {order_t(g)} {w:.0%}", w))
        x += w
    big = [t for t, w in small if w >= 0.01]
    sx.text(0, 0.05, "  ·  ".join(big), fontsize=12.5,
            color=C("band_meta"), va="center")
    save(fig, out, "banner", dict(T, bg=style.BAND[0]))
    ALT["banner"] = (f"Ling-3.0-tiny, {gb(file_bytes)}. {n} layers: {n_kda} KDA (layer "
                     f"{', '.join(map(str, dense))} with a dense FFN) and {n_mla} gated MLA. Routed experts "
                     f"{order_t('routed experts')}, {exp_share:.0%} of the weights" +
                     (", codes refined (verified record)." if refined else "."))


# ---------- A: release gate ----------

def demo_gate():
    rng = np.random.default_rng(7)
    size = {"cand": 3116281408, "bloomer": 3129076064, "bartowski": 3235879360, "Q8": 8408187872}
    rows = []
    for s, (_, ctx, ch) in rg.SETS.items():
        base = rng.uniform(0.25, 0.45)
        kl = {"cand": base, "bloomer": base * 1.25, "bartowski": base * rng.uniform(0.95, 1.15), "Q8": 0.008}
        for f in rg.FILES:
            r = dict(set=s, ctx=ctx, chunks=ch, file=f, gguf=f"demo-{f}.gguf", bytes=size[f], sha256="demo",
                     mean_kl=kl[f], p99=1.0, top1=80.0, diff=None, ci_lo=None, ci_hi=None, verdict=None)
            if f in rg.COMPS:
                d = kl["cand"] - kl[f]
                r.update(diff=d, ci_lo=d - 0.012, ci_hi=d + 0.012)
            rows.append(r)
    return rows


def release_gate(T, out, demo):
    src = ROOT / "results/release-gate.tsv"
    if demo:
        rows = demo_gate()
    elif src.exists():
        rows = rg.read_tsv(src)
    else:
        print(f"skip release-gate: {src} missing (scripts/release_gate_report.py)")
        return
    try:
        verdict = rg.validate(rows)
    except ValueError as e:
        raise SystemExit(f"release-gate: {src} fails validation, not drawn: {e}")
    sets = list(rg.SETS)
    get = {(r["set"], r["file"]): r for r in rows}
    mk = {"cand": dict(marker="o", ms=9, mfc=T["ours"], mec=T["ours"]),
          "bloomer": dict(marker="o", ms=9, mfc="none", mec=T["bloomer"], mew=1.8),
          "bartowski": dict(marker="s", ms=8.5, mfc="none", mec=T["bart"], mew=1.8)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.08))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.68, bottom=0.2)
    y = np.arange(len(sets))[::-1]
    off = {"cand": 0.0, "bloomer": 0.16, "bartowski": -0.16}
    for yi, s in zip(y, sets):
        for f in ("bartowski", "bloomer", "cand"):
            a1.plot(get[(s, f)]["mean_kl"], yi + off[f], ls="none", zorder=5 if f == "cand" else 3, **mk[f])
        a1.plot([get[(s, "Q8")]["mean_kl"]] * 2, [yi - 0.28, yi + 0.28], color=T["mute"], lw=2, zorder=2)
        for f in ("bloomer", "bartowski"):
            r = get[(s, f)]
            d = r["diff"]
            a2.errorbar(d, yi + off[f], xerr=[[d - r["ci_lo"]], [r["ci_hi"] - d]], ls="none", capsize=3,
                        color=mk[f]["mec"], elinewidth=1.6, zorder=3, **mk[f])
    a1.set_xlim(0, None)
    a1.set_yticks(y, [f"{s}\n{get[(s, 'cand')]['chunks']} × {get[(s, 'cand')]['ctx']:,}" for s in sets],
                  fontsize=10.5, color=T["ink"])
    a1.set_xlabel("mean KL vs BF16 (one BF16 reference per set)")
    a1.grid(axis="y", visible=False)
    a1.annotate("Q8_0 reference", xy=(get[(sets[0], "Q8")]["mean_kl"], y[0] + 0.28), xytext=(4, 4),
                textcoords="offset points", fontsize=9, color=T["mute"])
    a2.axvline(0, color=T["ink"], lw=1)
    a2.set_yticks(y, [""] * len(y))
    a2.set_xlabel("ours − comparator, paired 95% CI (left of 0 = ours closer)")
    a2.grid(axis="y", visible=False)
    for a in (a1, a2):
        a.set_ylim(-0.6, len(sets) - 0.4)

    def key(f, yy):
        fig.add_artist(plt.Line2D([0.03], [yy - 0.022], transform=fig.transFigure, ls="none",
                                  **{k: (v * 0.85 if k == "ms" else v) for k, v in mk[f].items()}))
    fig.text(0.02, 0.965, "Release gate: four held-out sets", va="top", **style.title_kw(T, 15))
    key("cand", 0.895)
    fig.text(0.048, 0.895, f"Ours  {gb(int(get[(sets[0], 'cand')]['bytes']))}", va="top", fontsize=11, color=T["ours"],
             fontweight="medium")
    for i, f in enumerate(rg.COMPS):
        yy = 0.835 - 0.055 * i
        key(f, yy)
        fig.text(0.048, yy, f"{LABEL[f]}  {gb(int(get[(sets[0], f)]['bytes']))}", va="top", fontsize=11, color=mk[f]["mec"])
        v, miss = verdict[f]
        fig.text(0.47, yy, f"vs {LABEL[f]}: {v}" + (f" ({miss[0]})" if miss else ""), va="top", fontsize=11,
                 color=T["warm"], fontweight="medium")
    fig.text(0.47, 0.895, "rule per comparator: lower KL on all 4 sets, CI below 0, file no larger", va="top",
             fontsize=9.5, color=T["mute"])
    fig.text(0.02, 0.06, "Intervals are pointwise 95% (paired bootstrap over chunks, 10,000 resamples, seed 0); the"
             " all-four rule is applied per comparator.", fontsize=9, color=T["mute"])
    fig.text(0.02, 0.02, "PG19: 4 books × 8 chunks, registered chunk interval; book-stratified interval in"
             " results/release-gate.md. 512 and 4,096 sets are never pooled.", fontsize=9, color=T["mute"])
    if demo:
        watermark(fig)
    save(fig, out, "release-gate", T)
    parts = []
    for s in sets:
        c = get[(s, "cand")]
        parts.append(f"{s} ({c['chunks']} × {c['ctx']}): ours {c['mean_kl']:.3g}, " + ", ".join(
            f"{LABEL[f]} {get[(s, f)]['mean_kl']:.3g} (diff {get[(s, f)]['diff']:+.4f} "
            f"[{get[(s, f)]['ci_lo']:+.4f}, {get[(s, f)]['ci_hi']:+.4f}])" for f in rg.COMPS) +
            f", Q8_0 {get[(s, 'Q8')]['mean_kl']:.3g}")
    ALT["release-gate"] = ("Mean KL vs BF16. " + "; ".join(parts) + ". Verdicts: " +
                           ", ".join(f"vs {LABEL[f]} {verdict[f][0]}" for f in rg.COMPS) + ".")


# ---------- B: ladder ----------

STEPS = {"bloomer": "bloomer010\nIQ3_XXS", "m2": "m2\n(Daniel's mix)", "m2r": "m2r\n(+ refined experts)",
         "m2rd": "m2rd\n(+ refined KDA gates)"}
SEGMENT = {"m2": "public file vs Daniel's mix", "m2r": "refined expert codes, same bytes",
           "m2rd": "refined KDA-gate codes, same bytes"}
LSETS = [("wiki512", "wiki.test", "o", "-"), ("ood512", "eval-ood", "s", "--")]


def ladder_data(demo):
    K = ROOT / "runs/gate-t"
    if demo:
        rng = np.random.default_rng(3)
        d = {}
        for s, base in (("wiki512", 0.50), ("ood512", 0.34)):
            cur = rng.gamma(2, base / 2, 64)
            for n, f in (("bloomer", 1), ("m2", 0.78), ("m2r", 0.66), ("m2rd", 0.66)):
                d[(n, s)] = cur * f * rng.normal(1, 0.03, 64)
            d[("Q8", s)] = np.full(64, 0.009 if s == "wiki512" else 0.005)
        return d, list(STEPS)
    have = [n for n in STEPS if all((K / f"{n}-{s}.f32").exists() for s, *_ in LSETS)]
    need = ["bloomer", "m2", "m2r"]
    if have[:3] != need or not all((K / f"Q8-{s}.f32").exists() for s, *_ in LSETS):
        missing = [f"{n}-{s}.f32" for n in need + ["Q8"] for s, *_ in LSETS if not (K / f"{n}-{s}.f32").exists()]
        print(f"skip ladder: runs/gate-t/ lacks {', '.join(missing)}")
        return None, None
    d = {}
    for n in have + ["Q8"]:
        for s, *_ in LSETS:
            v = np.fromfile(K / f"{n}-{s}.f32", dtype=np.float32).astype(np.float64)
            if len(v) != 64 * 255:
                raise SystemExit(f"{n}-{s}: {len(v)} values, ladder needs exactly 64 chunks x 255")
            d[(n, s)] = v.reshape(64, 255).mean(1)
    return d, have


def ladder(T, out, demo):
    d, names = ladder_data(demo)
    if d is None:
        return
    branch = None
    if "m2rd" in names:   # registered: m2rd adopted over m2r if its paired CI vs m2r is below 0 on both sets
        res = [(lab, *boot(d[("m2rd", s)], d[("m2r", s)])) for s, lab, *_ in LSETS]
        if not all(r[2][1] < 0 for r in res):
            names = names[:-1]
            branch = "m2rd (+ refined KDA gates) tested, not adopted (CI vs m2r not below 0 on both sets): " + "; ".join(
                f"{lab} {dd:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" for lab, dd, ci, *_ in res)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.2)
    x = np.arange(len(names))
    grad = style.ramp(style.ACCENT, style.BAND_POS, 64)
    ymax = max(d[(n, s)].mean() for n in names for s, *_ in LSETS)
    boxes, alt = {}, []
    for s, lab, m, ls in LSETS:
        ys = [d[(n, s)].mean() for n in names]
        for i in range(len(names) - 1):
            if i == 0:
                ax.plot(x[i:i + 2], ys[i:i + 2], ls=ls, color=T["dense"], lw=2.4, zorder=2)
            else:
                t = np.linspace(0, 1, 40)
                px, py = x[i] + t, ys[i] + t * (ys[i + 1] - ys[i])
                pts = np.stack([np.c_[px[:-1], py[:-1]], np.c_[px[1:], py[1:]]], 1)
                ax.add_collection(LineCollection(pts, colors=grad[np.linspace(0, 63, 39).astype(int)], lw=2.4,
                                                 linestyles=ls, zorder=2))
            dd, ci, q, qci = boot(d[(names[i + 1], s)], d[(names[i], s)])
            boxes.setdefault(i, []).append((lab.replace("eval-", "").split(".")[0], dd, ci, q))
        ax.plot(x, ys, ls="none", marker=m, ms=8, mfc=T["bg"] if m == "s" else T["ours"], mec=T["ours"], mew=1.8,
                zorder=3)
        ax.annotate(lab, xy=(x[0], ys[0]), xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=10, color=T["ink"])
        ax.axhline(d[("Q8", s)].mean(), color=T["mute"], lw=1, ls=":")
        alt.append(f"{lab}: " + ", ".join(f"{n} {v:.3g}" for n, v in zip(names, ys)) +
                   f", Q8_0 {d[('Q8', s)].mean():.3g}")
    for i, items in boxes.items():
        lines = [SEGMENT[names[i + 1]]] + [f"{k} {dd:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" for k, dd, ci, q in items]
        if names[i + 1] == "m2r":   # registered: ratio <= 0.90 and CI < 0 on both sets
            met = all(q <= 0.9 and ci[1] < 0 for _, _, ci, q in items)
            lines.append("ratios " + " / ".join(f"{q:.3f}" for *_, q in items) + f": rule {'met' if met else 'not met'}")
        if names[i + 1] == "m2rd":
            lines.append("adopted (CI < 0 on both)")
        alt.append(" | ".join(lines))
        ax.text(x[i] + 0.5, 0.04, "\n".join(lines), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.4, color=T["ink"], linespacing=1.45, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", fc=T["panel"], ec=T["rule"], lw=0.6))
    ax.annotate("went to the release gate", xy=(x[-1], d[(names[-1], "wiki512")].mean()), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=9, color=T["ours"], fontweight="medium")
    ax.text(x[-1] + 0.35, d[("Q8", "wiki512")].mean(), "Q8_0 reference", fontsize=8.5, color=T["mute"],
            va="bottom", ha="right")
    ax.set_xlim(-0.75, len(names) - 0.4)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xticks(x, [STEPS[n] for n in names], fontsize=10, color=T["ink"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("mean KL vs BF16")
    fig.text(0.02, 0.965, "From bloomer010's IQ3_XXS to the candidate, step by step", va="top",
             **style.title_kw(T, 15))
    fig.text(0.02, 0.89, "Selection sets (wiki.test, eval-ood; 64 × 512), not the release test. Boxes: paired"
             " difference per step [95% CI].", va="top", fontsize=10, color=T["mute"])
    fig.text(0.02, 0.84, "Slate: public file vs Daniel's mix (allocation, calibration and recipe all differ);"
             " gradient: refined codes at equal bytes.", va="top", fontsize=10, color=T["mute"])
    fig.text(0.02, 0.065, "m2 was chosen among candidates scored on the first 16 wiki chunks and the refiner tuned on these texts:"
             " gaps are optimistic.", fontsize=9, color=T["mute"])
    if branch:
        fig.text(0.02, 0.025, branch, fontsize=8.5, color=T["mute"])
        alt.append(branch)
    if demo:
        watermark(fig)
    save(fig, out, "ladder", T)
    ALT["ladder"] = "Mean KL vs BF16, 64 × 512 selection sets. " + ". ".join(alt) + "."


# ---------- C: sensitivity ----------

PRETTY = {"kda_qkv": "KDA q/k/v", "kda_gates": "KDA gates", "kda_out": "KDA output", "mla_proj": "MLA projections",
          "mla_gate": "MLA gate", "mla_out": "MLA output", "shexp": "shared expert", "dense0": "dense FFN (layer 0)",
          "exp_gate_up": "experts gate/up", "exp_down": "experts down"}
FAMILY = {"kda": ("o", "kda"), "mla": ("D", "mla"), "exp": ("s", "dense"), "shexp": ("^", "dense"),
          "dense0": ("v", "dense")}
PLACE = {"kda_qkv": (8, 6, "left"), "kda_gates": (8, -2, "left"), "kda_out": (8, 4, "left"),
         "mla_proj": (-9, 6, "right"), "mla_gate": (8, 12, "left"), "mla_out": (8, -4, "left"),
         "shexp": (0, 26, "center"), "dense0": (-9, -12, "right"), "exp_gate_up": (-9, 6, "right"),
         "exp_down": (-9, 2, "right")}


def m2_types(layout):
    if not Path(layout).exists():
        return {}
    rows, _ = read_layout(layout)
    out = {}
    for name, typ, b in rows:
        for g, rx in GROUPS.items():
            if re.search(rx, name):
                out.setdefault(g, {}).setdefault(typ, 0)
                out[g][typ] += b
                break
    return {g: "/".join(sorted(v, key=lambda t: -v[t])) for g, v in out.items()}


def sensitivity(T, out):
    src = ROOT / "results/ablations.tsv"
    if not src.exists():
        print(f"skip sensitivity: {src} missing")
        return
    with open(src) as f:
        rows = {r["name"]: r for r in csv.DictReader(f, delimiter="\t")}
    q8 = rows["q8_all"]
    qb, qk, qs = int(q8["bytes"]), float(q8["mean_kld"]), float(q8["kld_se"])
    pts = []
    for name, r in rows.items():
        m = re.fullmatch(r"g_(.+)_iq2_xxs", name)
        if m and m[1] in PRETTY:
            pts.append((m[1], (qb - int(r["bytes"])) / 1e6, float(r["mean_kld"]) - qk, float(r["kld_se"]) + qs))
    if len(pts) != len(PRETTY):
        raise SystemExit(f"sensitivity: {len(pts)} of {len(PRETTY)} groups in {src}")
    types = m2_types(ROOT / "results/layout-m2.tsv")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.77, bottom=0.18)
    for g, x, y, e in pts:
        fam = "kda" if g.startswith("kda") else "mla" if g.startswith("mla") else "exp" if g.startswith("exp") else g
        mark, role = FAMILY[fam]
        ax.errorbar(x, y, yerr=e, ls="none", color=T["mute"], elinewidth=1, capsize=2.5, zorder=2)
        ax.plot(x, y, marker=mark, ms=10, mfc=T[role], mec=T["border"] if role == "kda" else T["bg"], mew=1.3,
                ls="none", zorder=3)
        dx, dy, ha = PLACE[g]
        ax.annotate(PRETTY[g] + (f"\nm2: {types[g]}" if g in types else ""), xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=9, color=T["ink"], linespacing=1.15)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 10_000)
    ax.set_ylim(min(0, min(y - e for _, _, y, e in pts) * 1.1), None)
    ax.set_xlabel("MB saved by putting this group at IQ2_XXS (log scale)")
    ax.set_ylabel(r"$\Delta$KL vs all-Q8_0")
    fig.text(0.02, 0.965, "Which tensor groups can go to 2 bits", va="top", **style.title_kw(T, 15))
    fig.text(0.02, 0.895, "One group at IQ2_XXS on a Q8_0 base; second label line: the type m2 actually got"
             " (from its header).", va="top", fontsize=10, color=T["mute"])
    fig.text(0.02, 0.85, "Circles KDA (pale sky), diamonds MLA (blue), squares routed experts, triangles shared"
             " expert / dense FFN (slate).", va="top", fontsize=10, color=T["mute"])
    fig.text(0.02, 0.06, f"wiki.test {q8['chunks']} × {q8['ctx']}, selection-era data (results/ablations.tsv)."
             " Bars: sum of the two runs' SEs, a bound under token independence;", fontsize=8.8, color=T["mute"])
    fig.text(0.02, 0.025, "not paired chunk intervals. On a log x axis the slope is not KL per MB.", fontsize=8.8,
             color=T["mute"])
    save(fig, out, "sensitivity", T)
    ALT["sensitivity"] = ("Added KL vs MB saved, one group at IQ2_XXS on Q8_0: " + "; ".join(
        f"{PRETTY[g]} +{y:.3g} for " + (f"{x:,.0f}" if x >= 100 else f"{x:.3g}")
        + f" MB ({y / (x / 1e3):.3g} KL per GB)" for g, x, y, e in
        sorted(pts, key=lambda p: -p[2] / p[1])) + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    ap.add_argument("--demo", action="store_true", help="A and B from synthetic numbers into figures/demo/")
    ap.add_argument("--layout", default=str(ROOT / "results/layout-m2.tsv"), help="layout TSV of the scored GGUF")
    ap.add_argument("--record", help="refinement record (scripts/refine_record.py) for that GGUF")
    a = ap.parse_args()
    T = style.theme(a.theme)
    style.apply(T)
    if a.demo:
        out = ROOT / "figures/demo"
        release_gate(T, out, True)
        ladder(T, out, True)
        return
    out = ROOT / "figures"
    for n in NAMES:
        for ext in ("png", "svg"):
            (out / f"{n}-{a.theme}.{ext}").unlink(missing_ok=True)
    banner(T, out, a.layout, a.record)
    release_gate(T, out, False)
    ladder(T, out, False)
    sensitivity(T, out)
    (out / "alt.md").write_text("# Alt text (generated by scripts/figures.py)\n\n" +
                                "\n\n".join(f"- {k}: {v}" for k, v in ALT.items()) + "\n")


if __name__ == "__main__":
    main()
