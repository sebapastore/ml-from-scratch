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

- `linalg/` — basic linear algebra (matrices, operations)
- `ml/` — machine learning models built on top of linalg
- `examples/` — small programs using the libraries
- `datasets/` — small datasets (e.g. Iris)

### Run synthetic linear regression example

Build and run the example:

```sh
gcc -I./linalg/include -I./ml/include \
    linalg/src/matrix.c \
    linalg/src/linalg.c \
    ml/src/linear_regression.c \
    examples/synthetic_linear_regression.c \
    -o bin/synthetic_linear_regression
```

```sh
./bin/synthetic_linear_regression
```

## Status

Early stage. APIs may change as concepts are added.

## License

MIT
