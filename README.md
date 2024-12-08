# OptiEPUB

OptiEPUB is a command-line tool for optimizing EPUB files by [LunNova](https://lunnova.dev/).

## Installation

Install OptiEPUB using cargo:

```bash
cargo install optiepub
```

## Features

- Image optimization with content-aware cropping
- Duplicate and unused asset detection and removal
- Automatic format conversion for efficiency (e.g., PNG to JPEG if transparency isn't needed)
- HTML reference updating and EPUB structure preservation
- Customizable optimization parameters

## Usage

```bash
optiepub [OPTIONS] <input>
```

Example:
```bash
# Create output.epub from input.epub
# Use quality level 80 for JPEGs
# Resize images over 1200 pixels
optiepub -q 80 -d 1200 input.epub -o output.epub
```

Run `optiepub --help` for full option list.
