# go-mpfr

A Go binding for the [MPFR library](https://www.mpfr.org/) to enable arbitrary-precision floating-point arithmetic with well-defined rounding. This is not ready for production use yet, but we welcome contributions!

## Inspiration/The Why


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
	mpfr "github.com/mexicantexan/go-mpfr"
)

func main() {
	x := mpfr.NewFloat().SetFloat64(1.5)
	y := mpfr.NewFloat().SetFloat64(2.25)
	sum := mpfr.NewFloat().Add(x, y, mpfr.RoundToNearest)
	fmt.Printf("1.5 + 2.25 = %v\n", sum.Float64())
}
```
Run it:
```bash
go run main.go
```

## Contributing
We welcome contributions! See our [CONTRIBUTING](https://github.com/mexicantexan/go-mpfr/blob/master/CONTRIBUTING.md) guide (TODO) for how to get started.

## License
This project is licensed under the MIT License - see the [LICENSE] (https://github.com/mexicantexan/go-mpfr/blob/master/LICENSE) file for details.

Inspiration for this project comes from [GMP](https://github.com/ncw/gmp)!
