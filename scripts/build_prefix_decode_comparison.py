#!/usr/bin/env python3
"""Build a labeled prefix-decoding comparison from saved tokenizer previews.

Figure contract
---------------
Question: How does reconstruction evolve as the decoder receives more latent
tokens, and how does prefix-trained behavior differ from full-only training?
Takeaway: Prefix training produces smooth coarse-to-fine refinement; the
full-only tokenizer has a better full reconstruction but no useful early-prefix
path. Surface: static PNG for the experiment report. Evidence: the first six
fixed preview examples, with no manual example selection. Palette: blue versus
orange panel labels plus neutral borders/text; panel position also distinguishes
the models so color is not required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PREFIXES = (1, 2, 4, 8, 16, 32, 64)
SOURCE_COLUMNS = 16
SOURCE_ROWS = 1 + len(PREFIXES)
SOURCE_CELL = 32
SOURCE_PADDING = 2


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    value: str,
    selected_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    box = draw.textbbox((0, 0), value, font=selected_font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), value, font=selected_font, fill=fill)


def load_source(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    expected = (
        SOURCE_COLUMNS * SOURCE_CELL + (SOURCE_COLUMNS + 1) * SOURCE_PADDING,
        SOURCE_ROWS * SOURCE_CELL + (SOURCE_ROWS + 1) * SOURCE_PADDING,
    )
    if image.size != expected:
        raise ValueError(f"{path} has size {image.size}, expected {expected}")
    return image


def load_psnr(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text())
    return {int(key): float(value["psnr_db"]) for key, value in payload["prefix"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progressive_preview",
        default="tokenizer_runs/v5-vae-kl1e4-s1/reconstruction_final.png",
    )
    parser.add_argument(
        "--progressive_metrics",
        default="tokenizer_runs/v5-vae-kl1e4-s1/metrics_final.json",
    )
    parser.add_argument(
        "--unordered_preview",
        default="tokenizer_runs/v8-unordered-vae-s1/reconstruction_final.png",
    )
    parser.add_argument(
        "--unordered_metrics",
        default="tokenizer_runs/v8-unordered-vae-s1/metrics_final.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/2026-08-26_autoencoder_program/"
            "prefix_decode_comparison.png"
        ),
    )
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()

    if not 1 <= args.examples <= SOURCE_COLUMNS:
        raise ValueError(f"examples must lie in [1, {SOURCE_COLUMNS}]")

    panels = [
        (
            "Progressive prefix-trained tokenizer",
            load_source(Path(args.progressive_preview)),
            load_psnr(Path(args.progressive_metrics)),
            (42, 95, 160),
        ),
        (
            "Unordered full-reconstruction tokenizer",
            load_source(Path(args.unordered_preview)),
            load_psnr(Path(args.unordered_metrics)),
            (194, 105, 45),
        ),
    ]

    columns = (None,) + PREFIXES
    cell_pitch = 78
    image_size = 68
    left = 28
    right = 28
    title_height = 82
    panel_heading = 73
    panel_gap = 28
    footnote_height = 42
    panel_body = args.examples * cell_pitch
    width = left + len(columns) * cell_pitch + right
    height = (
        title_height
        + 2 * (panel_heading + panel_body)
        + panel_gap
        + footnote_height
    )

    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    ink = (32, 35, 39)
    muted = (95, 99, 104)
    border = (210, 211, 208)

    draw.text(
        (left, 17),
        "Prefix decoding by available latent tokens",
        font=font(24, bold=True),
        fill=ink,
    )
    draw.text(
        (left, 49),
        "Same first six held-out CIFAR-10 examples; both tokenizers use 64×16 latents.",
        font=font(13),
        fill=muted,
    )

    panel_top = title_height
    for panel_index, (label, source, psnr, accent) in enumerate(panels):
        draw.rectangle((left, panel_top + 3, left + 6, panel_top + 29), fill=accent)
        draw.text((left + 15, panel_top), label, font=font(18, bold=True), fill=ink)
        for column_index, prefix in enumerate(columns):
            center_x = left + column_index * cell_pitch + cell_pitch // 2
            heading = "Reference" if prefix is None else f"k={prefix}"
            centered_text(draw, center_x, panel_top + 34, heading, font(12, bold=True), ink)
            if prefix is not None:
                centered_text(
                    draw,
                    center_x,
                    panel_top + 51,
                    f"{psnr[prefix]:.1f} dB",
                    font(10),
                    muted,
                )

        body_top = panel_top + panel_heading
        for example_index in range(args.examples):
            for column_index, prefix in enumerate(columns):
                source_row = 0 if prefix is None else 1 + PREFIXES.index(prefix)
                source_x = SOURCE_PADDING + example_index * (
                    SOURCE_CELL + SOURCE_PADDING
                )
                source_y = SOURCE_PADDING + source_row * (
                    SOURCE_CELL + SOURCE_PADDING
                )
                crop = source.crop(
                    (
                        source_x,
                        source_y,
                        source_x + SOURCE_CELL,
                        source_y + SOURCE_CELL,
                    )
                ).resize((image_size, image_size), Image.Resampling.NEAREST)
                x = left + column_index * cell_pitch + (cell_pitch - image_size) // 2
                y = body_top + example_index * cell_pitch
                canvas.paste(crop, (x, y))
                draw.rectangle(
                    (x - 1, y - 1, x + image_size, y + image_size),
                    outline=border,
                    width=1,
                )

        panel_top = body_top + panel_body + panel_gap

    draw.text(
        (left, height - 29),
        "Rows are fixed examples, not selected outcomes. PSNR is measured over all 10,000 test images.",
        font=font(11),
        fill=muted,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
