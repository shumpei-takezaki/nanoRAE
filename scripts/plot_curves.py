"""Training curves for the experiment write-ups: loss per epoch (Stage 1: L1 and LPIPS; Stage 2: flow loss) and, for Stage 2,
FID (10k samples) during training.

Usage: uv run python scripts/plot_curves.py 01 02 04 05 06 07 08 12 13
Reads the training logs in ckpts/ when they exist and writes docs/figures/<exp>/curves.json; otherwise plots that JSON.
Writes docs/figures/<exp>/curves.png.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated all-pairs)
STEPS_PER_EPOCH = 97  # Stage 2: 50k images, batch 512
GAN_EPOCHS = 400      # Stage 1: GAN fine-tuning starts after 400 epochs without GAN

# exp: (folder, stage, [(label, color, linestyle, logs)]); Stage 1 runs list (no-GAN logs, GAN log), later logs override earlier epochs
EXPS = {
    "01": ("01_stage1_decoder_capacity", 1, [
        ("ViT-B (fp32)", BLUE, "-", (["dinov2_decB_nogan"], "dinov2_decB_gan_ft")),
        ("ViT-L", ORANGE, "-", (["dinov2_decL_nogan"], "dinov2_decL_gan_ft")),
        ("ViT-XL", AQUA, "-", (["dinov2_decXL_nogan.crash1", "dinov2_decXL_nogan"], "dinov2_decXL_gan_ft"))]),
    "04": ("04_encoder_capacity", 1, [
        ("Tiny/4 encoder", BLUE, "-", (["dinov2_decXL_nogan.crash1", "dinov2_decXL_nogan"], "dinov2_decXL_gan_ft")),
        ("S/4 encoder", ORANGE, "-", (["dinov2s_decXL_nogan"], "dinov2s_decXL_gan_ft"))]),
    "08": ("08_decoder_noise", 1, [
        ("with noise aug.", BLUE, "-", (["dinov2_decXL_nogan.crash1", "dinov2_decXL_nogan"], "dinov2_decXL_gan_ft")),
        ("without noise aug.", ORANGE, "-", (["dinov2_decXL_nogan_nonoise"], "dinov2_decXL_gan_ft_nonoise"))]),
    "02": ("02_stage2_ditdh_s", 2, [("DiT$^{DH}$-S, 2048 head", BLUE, "-", ["ditdhS_decXLgan"])]),
    "05": ("05_stage2_model_size", 2, [
        ("DiT$^{DH}$-S, 2048 head", BLUE, "-", ["ditdhS_decXLgan"]),
        ("DiT$^{DH}$-S, 512 head", ORANGE, "-", ["ditdh_s_h512_decXLgan"]),
        ("DiT$^{DH}$-Ti", AQUA, "-", ["ditdh_ti_decXLgan"])]),
    "06": ("06_ddt_head", 2, [
        ("DiT$^{DH}$-S, 512 head", BLUE, "-", ["ditdh_s_h512_decXLgan"]),
        ("LightningDiT-S/1", ORANGE, "-", ["lightningdit_s_decXLgan"]),
        ("LightningDiT-Ti/1", AQUA, "-", ["lightningdit_ti_decXLgan"])]),
    "07": ("07_time_shift", 2, [
        ("with shift", BLUE, "-", ["ditdh_s_h512_decXLgan"]),
        ("without shift", ORANGE, "-", ["ditdh_s_h512_noshift_decXLgan"])]),
    "12": ("12_rjf", 2, [
        ("DiT$^{DH}$-S, 512 head + RJF", BLUE, "-", ["rjf_ditdh_s_h512_decXLgan"]),
        ("LightningDiT-S/1 + RJF", ORANGE, "-", ["rjf_lightningdit_s_decXLgan"]),
        ("LightningDiT-Ti/1 + RJF", AQUA, "-", ["rjf_lightningdit_ti_decXLgan"])]),
    "13": ("13_rfm", 2, [
        ("S/1 + RJF", ORANGE, "-", ["rjf_lightningdit_s_decXLgan"]),
        ("S/1 + RFM", ORANGE, "--", ["rfm_lightningdit_s_decXLgan"]),
        ("Ti/1 + RJF", BLUE, "-", ["rjf_lightningdit_ti_decXLgan"]),
        ("Ti/1 + RFM", BLUE, "--", ["rfm_lightningdit_ti_decXLgan"])]),
}

EPOCH = re.compile(r"\[Epoch (\d+)\] (epoch/.*)")
FID = re.compile(r"\[Eval\] Step (\d+) Metrics:\s*\n\s*fid: ([\d.]+)")


def parse(log):
    text = Path(f"ckpts/{log}.log").read_text(errors="ignore")
    losses = {}
    for m in EPOCH.finditer(text):
        losses[int(m.group(1))] = {k.split("/")[1]: float(v) for k, v in re.findall(r"(epoch/\w+): (-?[\d.]+)", m.group(2))}
    fid = {int(s) / STEPS_PER_EPOCH: float(f) for s, f in FID.findall(text)}
    return losses, fid


def extract(stage, runs):
    out = {}
    for label, _, _, logs in runs:
        if stage == 1:
            nogan, gan = logs
            loss = {}
            for log in nogan:
                loss.update(parse(log)[0])
            loss.update({GAN_EPOCHS + e: v for e, v in parse(gan)[0].items()})
            out[label] = {"epoch": sorted(loss), "l1": [loss[e]["loss_recon"] for e in sorted(loss)],
                          "lpips": [loss[e]["loss_lpips"] for e in sorted(loss)]}
        else:
            loss, fid = {}, {}
            for log in logs:
                l, f = parse(log); loss.update(l); fid.update(f)
            out[label] = {"epoch": sorted(loss), "loss": [loss[e]["loss"] for e in sorted(loss)],
                          "fid_epoch": sorted(fid), "fid10k": [fid[e] for e in sorted(fid)]}
    return out


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("epoch")


def plot(stage, runs, data, path):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "legend.frameon": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a in ax:
        style(a)
    for label, color, ls, _ in runs:
        d = data[label]
        if stage == 1:
            ax[0].plot(d["epoch"], d["l1"], ls, color=color, lw=2, label=label)
            ax[1].plot(d["epoch"], d["lpips"], ls, color=color, lw=2, label=label)
        else:
            ax[0].plot(d["epoch"], d["loss"], ls, color=color, lw=2, label=label)
            ep, f = zip(*[(e, v) for e, v in zip(d["fid_epoch"], d["fid10k"]) if e > 0])  # epoch 0: untrained (FID ~238)
            ax[1].plot(ep, f, ls, color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE, label=label)
    if stage == 1:
        for a, key, name in zip(ax, ("l1", "lpips"), ("L1 loss", "LPIPS loss")):
            a.set_ylabel(name)
            a.set_ylim(0.9 * min(min(data[l][key]) for l, *_ in runs), 1.1 * max(data[l][key][20] for l, *_ in runs))  # from epoch 20 on
            a.axvline(GAN_EPOCHS, color=INK2, lw=1, ls=":")
            a.text(GAN_EPOCHS + 4, 0.97, "GAN on", transform=a.get_xaxis_transform(), va="top", color=INK2, fontsize=8)
    else:
        ax[0].set_ylabel("training loss (epoch mean)")
        lo = min(min(data[l]["loss"]) for l, *_ in runs); hi = max(data[l]["loss"][min(20, len(data[l]["loss"]) - 1)] for l, *_ in runs)
        ax[0].set_ylim(lo * 0.97, hi * 1.05)
        ax[1].set_ylabel("FID (10k samples, during training)")
        ax[1].set_xlim(ax[0].get_xlim())
        fids = [v for l, *_ in runs for e, v in zip(data[l]["fid_epoch"], data[l]["fid10k"]) if e > 0]
        if max(fids) / min(fids) > 8:
            ax[1].set_yscale("log")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    for exp in sys.argv[1:]:
        folder, stage, runs = EXPS[exp]
        js = Path(f"docs/figures/{folder}/curves.json")
        if Path("ckpts").exists():
            js.parent.mkdir(parents=True, exist_ok=True)
            js.write_text(json.dumps(extract(stage, runs)))
        plot(stage, runs, json.loads(js.read_text()), js.with_suffix(".png"))
        print("wrote", js.with_suffix(".png"))


if __name__ == "__main__":
    main()
