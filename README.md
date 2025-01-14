# go-mpfr

A Go binding for the [MPFR library](https://www.mpfr.org/) to enable arbitrary-precision floating-point arithmetic with well-defined rounding. This is not ready for production use yet, but we welcome contributions! Please note that this has no association with the MPFR project.

## Inspiration/The Why
This was sparked due to a need for high-precision floating-point calculations in Go. The standard library's `math/big` package is great for most applications, but it lacks certain features that mpfr supports out of the box. This project is being used by another in order to perform some high precision calculations, which is why all of the functionality of mpfr has not been implemented yet. Only the necessary functions have been implemented so far.

Inspiration for this project comes from [GMP](https://github.com/ncw/gmp)! Huge shoutout to all the contributors there and for [mpfr](https://machinecognitis.github.io/Math.Mpfr.Native/html/6146e71f-a352-2e48-f234-6d79ad0f8c3c.htm) for providing such a great library!!

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

A quick example showing different setters, getters, and function calling:

```go
package main

import (
	"fmt"
	mpfr "github.com/mexicantexan/go-mpfr"
        "math/big"
)

func main() {
	// add two separate floats into a new float
	x := mpfr.NewFloat().SetFloat64(1.5)
	y := mpfr.FromFloat64(2.25) // there are many ways to create a float
	sum := mpfr.Add(x, y, mpfr.RoundToNearest)
	fmt.Printf("1.5 + 2.25 = %v\n", sum.Float64())

	// perform "inplace" operations 
	x.Add(y)
	fmt.Printf("1.5 + 2.25 = %v\n", x.Float64())

	// add multiple floats (summing 1.5, 2.25, 3.75, and 1.0)
	t := mpfr.FromInt(1)
	u := mpfr.FromInt64(2)
	v := mpfr.FromUint64(3)
	x = mpfr.FromFloat64(4.01)
	y = mpfr.FromBigInt(big.NewInt(5))
	z := mpfr.NewFloat().SetInt(6)
	t.Add(u, v, x, y, z)
	fmt.Printf("1 + 2 + 3 + 4.01 + 5 + 6 = %v\n", t.Float64())
        fmt.Printf("1 + 2 + 3 + 4.01 + 5 + 6 = %v\n", t.Int64())
	
	// set precision 
	x.SetPrec(256)

	// set RoundingMode 
	x.SetRoundingMode(mpfr.RoundToward0)
}
```
Run the example above:
```bash
go run main.go
```

## Contributing
We welcome contributions! See our [CONTRIBUTING](https://github.com/mexicantexan/go-mpfr/blob/master/CONTRIBUTING.md) guide (TODO) for how to get started.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/mexicantexan/go-mpfr/blob/master/LICENSE) file for details.

## Support the Project
If you find this project helpful and would like to support its development:

<a href="https://buymeacoffee.com/mexicantexan" target="_blank">
<img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important; width: 217px !important;">
</a>
