# ML from Scratch (C)

This repository contains a small machine learning library written in plain C.
The goal is to understand machine learning fundamentals by implementing them
from scratch, starting with linear algebra and basic models.

## Goals
- Learn ML by implementing core concepts manually
- Build a small, reusable linear algebra library
- Keep the code simple, readable, and correct

## Non-goals
- Production performance
- GPU acceleration
- Competing with BLAS / existing ML frameworks

## Structure
- `linalg/`   — basic linear algebra (matrices, operations)
- `ml/`       — machine learning models built on top of linalg
- `examples/` — small programs using the libraries
- `datasets/` — small datasets (e.g. Iris)

## Build
Requires a C compiler (gcc or clang).

```sh
make
```

## Status
Early stage. APIs may change as concepts are added.

## License
MIT (or choose your preferred license).

