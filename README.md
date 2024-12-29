# go-mpfr

A Go binding for the [MPFR library](https://www.mpfr.org/) to enable arbitrary-precision floating-point arithmetic with well-defined rounding.

## Features

- High-precision floating-point calculations using MPFR.
- Automatic memory management in Go, with finalizers for MPFR structs.
- Support for multiple rounding modes (nearest, up, down, zero, etc.).
- Ability to set precision as needed.

## Install

First, ensure that GMP and MPFR are installed on your system:

**Ubuntu/Debian**:
```bash
sudo apt-get update && sudo apt-get install libmpfr-dev libgmp-dev
```
**macOS (Homebrew)**:
```bash
brew install mpfr gmp
```

Then, install the package using `go get`:
```bash
go get github.com/mexicantexan/go-mpfr
```

## Usage

A quick example:
```go
package main

import (
    "fmt"
    "github.com/mexicantexan/go-mpfr"
)

func main() {
    a := mpfr.New()
    b := mpfr.New()
    sum := mpfr.New()

    a.SetDouble(1.5, mpfr.RoundNearestTiesToEven)
    b.SetDouble(2.25, mpfr.RoundNearestTiesToEven)

    sum.Add(a, b, mpfr.RoundNearestTiesToEven)
    fmt.Println("1.5 + 2.25 =", sum.GetDouble(mpfr.RoundNearestTiesToEven))
}
```
Run it:
```bash
go run main.go
```

## Contributing
We welcome contributions! See our CONTRIBUTING guide (TODO) for how to get started.

## License
This project is licensed under the MIT License - see the LICENSE (TODO)  file for details.