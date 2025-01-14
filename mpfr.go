// Copyright 2024 go-mpfr Authors
//
// Use of this source code is governed by a license that can be
// found in the LICENSE file.

// Package mpfr provides Go bindings for the MPFR multiple-precision
// floating-point library.
package mpfr

/*
#cgo LDFLAGS: -lmpfr -lgmp
#include <mpfr.h>
#include <stdlib.h>

// For now, we’ll call MPFR functions directly
// from Go via cgo.
*/
import "C"
import (
	"math"
	"math/big"
	"runtime"
	"strings"
	"unsafe"
)

// Float represents a multiple-precision floating-point number.
// The zero value for Float is *not* valid until initialized
// (similar to how the GMP code works).
// Float represents a multiple-precision floating-point number.
type Float struct {
	mpfr         C.mpfr_t // Use C.mpfr_t directly (array of 1 struct)
	init         bool
	RoundingMode Rnd
}

// finalizer is called by the garbage collector when there are no
// more references to this Float. It clears the mpfr_t and releases
// native memory.
func finalizer(f *Float) {
	if f.init {
		C.mpfr_clear(&f.mpfr[0]) // Pass a pointer to the first element
		f.init = false
	}
}

// doinit initializes f.mpfr if it isn’t already initialized.
func (f *Float) doinit() {
	if f.init {
		return
	}
	f.init = true

	// Initialize the mpfr_t struct
	C.mpfr_init(&f.mpfr[0])

	// set the default rounding mode
	f.RoundingMode = RoundToNearest

	// Set the finalizer to clean up the memory when the object is garbage-collected
	runtime.SetFinalizer(f, finalizer)
}

// Clear deallocates the native mpfr_t. After calling Clear,
// the Float must not be used again.
func (f *Float) Clear() {
	if !f.init {
		return
	}
	C.mpfr_clear(&f.mpfr[0]) // Pass a pointer to the first element
	f.init = false
}

// Rnd is the type for MPFR rounding modes.
//
// TODO: MPFR has more rounding modes, need to test them.
type Rnd int

const (
	RoundToNearest Rnd = Rnd(C.MPFR_RNDN) // Round to nearest, ties to even
	RoundToward0   Rnd = Rnd(C.MPFR_RNDZ)
	RoundUp        Rnd = Rnd(C.MPFR_RNDU)
	RoundDown      Rnd = Rnd(C.MPFR_RNDD)
	RoundAway      Rnd = Rnd(C.MPFR_RNDA)
)

// NewFloat allocates and returns a new Float set to 0.0 with MPFR’s default precision.
func NewFloat() *Float {
	f := &Float{}
	f.doinit()
	f.SetFloat64(0.0)
	return f
}

func NewFloatWithPrec(prec uint) *Float {
	f := &Float{}
	f.doinit()
	f.SetFloat64(0.0)
	f.SetPrec(prec)
	return f
}

// GetFloat64 returns the float64 approximation of f, using the specified rounding mode.
func (f *Float) GetFloat64() float64 {
	f.doinit()
	return float64(C.mpfr_get_d(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)))
}

// SetString parses a string into f.
func (f *Float) SetString(s string, base int) error {
	f.doinit()
	cstr := C.CString(s)
	defer C.free(unsafe.Pointer(cstr))
	ret := C.mpfr_set_str(&f.mpfr[0], cstr, C.int(base), C.mpfr_rnd_t(f.RoundingMode))
	if ret != 0 {
		return ErrInvalidString
	}
	return nil
}

// String returns f as a base-10 string representation.
func (f *Float) String() string {
	f.doinit()

	var exp C.mpfr_exp_t
	base := 10
	cstr := C.mpfr_get_str(nil, &exp, C.int(base), 0, &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	if cstr == nil {
		return "<mpfr_get_str_error>"
	}
	defer C.mpfr_free_str(cstr)

	mantissa := C.GoString(cstr)
	intExp := int(exp)
	if intExp >= 0 {
		if intExp > len(mantissa) {
			//	pad with 0's
			mantissa += strings.Repeat("0", intExp-len(mantissa))
			return mantissa + ".0"
		}
		return mantissa[:intExp] + "." + mantissa[intExp:]
	}
	// pad with 0's
	mantissa = strings.Repeat("0", int(-intExp)) + mantissa
	return "0." + mantissa
}

// Copy sets f to x, copying the entire mpfr_t.
func (f *Float) Copy(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(RoundToNearest))
	return f
}

// Add performs sequential addition on the receiver `f` and stores the result:
//
//   - If called with one argument (`x`), the function computes f + x, where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with multiple arguments (`x1, x2, ..., xn`), the function sequentially adds each argument
//     to `f`, i.e., f = f + x1 + x2 + ... + xn. This modifies `f` in place and returns it.
//
// Example Usage:
//
//	// Add x to f (in-place addition):
//	f := NewFloat().SetFloat64(10.0)
//	x := NewFloat().SetFloat64(3.0)
//	f.Add(x) // f is now 10.0 + 3.0 = 13.0
//
//	// Sequentially add multiple values to f:
//	f.SetFloat64(5.0)
//	x1 := NewFloat().SetFloat64(2.0)
//	x2 := NewFloat().SetFloat64(3.0)
//	x3 := NewFloat().SetFloat64(4.0)
//	f.Add(x1, x2, x3) // f is now 5.0 + 2.0 + 3.0 + 4.0 = 14.0
//
// Notes:
// - At least one argument must be provided; otherwise, the function panics.
// - All arguments must be properly initialized before the call.
// - The computation uses the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Add(args ...*Float) *Float {
	if len(args) == 0 {
		// No arguments provided.
		panic("Add requires at least 1 argument")
	}

	f.doinit()

	// Sequentially add the arguments.
	for _, addend := range args {
		addend.doinit()
		C.mpfr_add(&f.mpfr[0], &f.mpfr[0], &addend.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Add sets f to x + y, with the given rounding mode, and returns f.
func Add(x, y *Float, rnd Rnd) *Float {
	x.SetRoundMode(rnd)
	return x.Add(y)
}

// Sub performs sequential subtraction on the receiver `f` and stores the result:
//
//   - If called with one argument (`x`), the function computes f - x, where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with multiple arguments (`x1, x2, ..., xn`), the function sequentially subtracts each argument
//     from `f`, i.e., f = f - x1 - x2 - ... - xn. This modifies `f` in place and returns it.
//
// Example Usage:
//
//	// Subtract x from f (in-place subtraction):
//	f := NewFloat().SetFloat64(10.0)
//	x := NewFloat().SetFloat64(3.0)
//	f.Sub(x) // f is now 10.0 - 3.0 = 7.0
//
//	// Sequentially subtract multiple values from f:
//	f.SetFloat64(20.0)
//	x1 := NewFloat().SetFloat64(5.0)
//	x2 := NewFloat().SetFloat64(3.0)
//	f.Sub(x1, x2) // f is now 20.0 - 5.0 - 3.0 = 12.0
//
// Notes:
// - At least one argument must be provided; otherwise, the function panics.
// - All arguments must be properly initialized before the call.
// - The computation uses the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Sub(args ...*Float) *Float {
	if len(args) == 0 {
		// No arguments provided.
		panic("Sub requires at least 1 argument")
	}
	f.doinit()

	// Sequentially subtract the arguments.
	for _, subtrahend := range args {
		subtrahend.doinit()
		C.mpfr_sub(&f.mpfr[0], &f.mpfr[0], &subtrahend.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

func Sub(x, y *Float, rnd Rnd) *Float {
	x.SetRoundMode(rnd)
	return x.Sub(y)
}

// Mul performs sequential multiplication on the receiver `f` and stores the result:
//
//   - If called with one argument (`x`), the function computes f * x, where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with multiple arguments (`x1, x2, ..., xn`), the function sequentially multiplies `f`
//     by each argument in order, i.e., f = f * x1 * x2 * ... * xn. This modifies `f` in place and returns it.
//
// Example Usage:
//
//	// Multiply f by x (in-place multiplication):
//	f := NewFloat().SetFloat64(2.0)
//	x := NewFloat().SetFloat64(3.0)
//	f.Mul(x) // f is now 2.0 * 3.0 = 6.0
//
//	// Sequentially multiply f by multiple values:
//	f.SetFloat64(1.0)
//	x1 := NewFloat().SetFloat64(2.0)
//	x2 := NewFloat().SetFloat64(3.0)
//	x3 := NewFloat().SetFloat64(4.0)
//	f.Mul(x1, x2, x3) // f is now 1.0 * 2.0 * 3.0 * 4.0 = 24.0
//
// Notes:
// - At least one argument must be provided; otherwise, the function panics.
// - All arguments must be properly initialized before the call.
// - The computation uses the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Mul(args ...*Float) *Float {
	if len(args) == 0 {
		// No arguments provided.
		panic("Mul requires at least 1 argument")
	}

	// Initialize the receiver.
	f.doinit()

	// Sequentially multiply by the arguments.
	for _, multiplier := range args {
		multiplier.doinit()
		C.mpfr_mul(&f.mpfr[0], &f.mpfr[0], &multiplier.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

func Mul(x, y *Float, rnd Rnd) *Float {
	x.SetRoundMode(rnd)
	return x.Mul(y)
}

// Div performs sequential division on the receiver `f` and stores the result:
//
//   - If called with one argument (`x`), the function computes f / x, where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with multiple arguments (`x1, x2, ..., xn`), the function sequentially divides `f`
//     by each argument in order, i.e., f = f / x1 / x2 / ... / xn. This modifies `f` in place and returns it.
//
// Example Usage:
//
//	// Divide f by x (in-place division):
//	f := NewFloat().SetFloat64(10.0)
//	x := NewFloat().SetFloat64(2.0)
//	f.Div(x) // f is now 10.0 / 2.0 = 5.0
//
//	// Sequentially divide f by multiple values:
//	f.SetFloat64(100.0)
//	x1 := NewFloat().SetFloat64(2.0)
//	x2 := NewFloat().SetFloat64(5.0)
//	f.Div(x1, x2) // f is now 100.0 / 2.0 / 5.0 = 10.0
//
// Notes:
// - At least one argument must be provided; otherwise, the function panics.
// - All arguments must be properly initialized before the call.
// - Division by zero will result in behavior as defined by the MPFR library (e.g., Inf or NaN).
// - The computation uses the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Div(args ...*Float) *Float {
	if len(args) == 0 {
		// No arguments provided.
		panic("Div requires at least 1 argument")
	}

	f.doinit()

	// Sequentially divide by the arguments.
	for _, divisor := range args {
		divisor.doinit()
		C.mpfr_div(&f.mpfr[0], &f.mpfr[0], &divisor.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

func Div(x, y *Float, rnd Rnd) *Float {
	x.SetRoundMode(rnd)
	return x.Div(y)
}

// Quo sets f to the quotient of x / y with the specified rounding mode and returns f.
// If y == 0, it panics with a division-by-zero error.
func (f *Float) Quo(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()

	// Check for division by zero
	if C.mpfr_zero_p(&y.mpfr[0]) != 0 { // mpfr_zero_p returns nonzero if y is zero
		panic("Quo: division by zero")
	}

	C.mpfr_div(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))

	return f
}

// Quo sets f to the quotient of x / y with the specified rounding mode and returns f.
func Quo(x, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Quo(x, y)
}

// Pow computes the power function and stores the result in the receiver `f`:
//
//   - If called with one argument (`y`), the function computes f^y (where `f` is the current value
//     of the receiver) and stores the result in `f`.
//
//   - If called with two arguments (`x`, `y`), the function computes x^y and stores the result in the receiver `f`
//     using the receiver's RoundingMode.
//
//   - If called with three arguments (`x`, `y`, `rnd`), the function computes x^y and stores the result in the receiver `f`
//     using the specified rounding mode `rnd`.
//
// Example Usage:
//
//	// Compute f^y (receiver raised to the power of y):
//	f := NewFloat().SetFloat64(2.0)
//	y := NewFloat().SetFloat64(3.0)
//	f.Pow(y) // f is now 2.0^3.0 = 8.0
//
//	// Compute x^y using the receiver's rounding mode:
//	x := NewFloat().SetFloat64(2.0)
//	y.SetFloat64(3.0)
//	f.Pow(x, y) // f is now 2.0^3.0 = 8.0
//
// Notes:
// - If only one argument is provided, `y` must be initialized before the call.
// - If two or three arguments are provided, both `x` and `y` must be initialized before the call.
// - If `rnd` is not provided, the rounding mode of the receiver `f` is used.
// - The function handles special cases like x = 0 or y = 0 according to the MPFR library rules.
// - Behavior for non-integer exponents and negative bases depends on MPFR.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Pow(args ...interface{}) *Float {
	if len(args) == 1 {
		// Compute f^y (receiver raised to the power of y).
		y, yOk := args[0].(*Float)
		if !yOk {
			panic("Pow expects a *Float as the single argument")
		}
		y.doinit()
		f.doinit()
		C.mpfr_pow(&f.mpfr[0], &f.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else if len(args) > 1 {
		// Compute x^y using the receiver's rounding mode.
		x, xOk := args[0].(*Float)
		y, yOk := args[1].(*Float)
		if !xOk || !yOk {
			panic("Pow expects two *Float arguments (x, y)")
		}
		x.doinit()
		y.doinit()
		f.doinit()
		C.mpfr_pow(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		panic("Pow expects at least one argument")
	}

	return f
}

// Pow sets f to x^y (x raised to the power y), with the given rounding mode, and returns f.
func Pow(x, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Pow(x, y)
}

// Exp computes the exponential function e^x and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes e^f, where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes e^x and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the exponential function of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Exp(x) // f is now e^1.0 ≈ 2.718
//
//	// Compute the exponential function of the receiver's value in place:
//	f.SetFloat64(2.0)
//	f.Exp() // f is now e^2.0 ≈ 7.389
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with no arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The exponential function e^x is defined for all real numbers.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Exp(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute e^f in place.
		C.mpfr_exp(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute e^x and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_exp(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

func Exp(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Exp(x)
}

// Log computes the natural logarithm (ln) of a value and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes ln(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes ln(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the natural logarithm of a specific value:
//	x := NewFloat().SetFloat64(2.718)
//	f := NewFloat()
//	f.Log(x) // f is now ln(2.718) ≈ 1.0
//
//	// Compute the natural logarithm of the receiver's value in place:
//	f.SetFloat64(10.0)
//	f.Log() // f is now ln(10.0)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with no arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - For inputs `x <= 0`, the result will be NaN (if x < 0) or -Inf (if x == 0), depending on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Log(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute ln(f) in place.
		C.mpfr_log(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute ln(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_log(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

func Log(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Log(x)
}

// Cmp compares f and x and returns -1 if f < x, 0 if f == x, +1 if f > x.
func (f *Float) Cmp(x *Float) int {
	f.doinit()
	x.doinit()
	return int(C.mpfr_cmp(&f.mpfr[0], &x.mpfr[0]))
}

// Cmp compares x and y and returns -1 if x < y, 0 if x == y, +1 if x > y.
func Cmp(x, y *Float) int {
	x.doinit()
	y.doinit()
	return int(C.mpfr_cmp(&x.mpfr[0], &y.mpfr[0]))
}

// Abs computes the absolute value of a value, |x|, and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes |f|, where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes |x| and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the absolute value of a specific value:
//	x := NewFloat().SetFloat64(-3.5)
//	f := NewFloat()
//	f.Abs(x) // f is now 3.5
//
//	// Compute the absolute value of the receiver's value in place:
//	f.SetFloat64(-2.5)
//	f.Abs() // f is now 2.5
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with no arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The absolute value function |x| is defined for all real numbers.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Abs(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute |f| in place.
		C.mpfr_abs(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute |x| and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_abs(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

func Abs(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Abs(x)

}

// Acos computes the arccosine of a value, arccos(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes arccos(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes arccos(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the arccosine of a specific value:
//	x := NewFloat().SetFloat64(0.5)
//	f := NewFloat()
//	f.Acos(x) // f is now arccos(0.5)
//
//	// Compute the arccosine of the receiver's value in place:
//	f.SetFloat64(0.3)
//	f.Acos() // f is now arccos(0.3)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called with no arguments, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The arccosine function is defined only for values in the range [-1, 1]; behavior outside this range
//     depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Acos(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute arccos(f) in place.
		C.mpfr_acos(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute arccos(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_acos(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Acos sets f = arccos(x) with rounding mode rnd.
func Acos(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Acos(x)
}

// Acosh computes the inverse hyperbolic cosine of a value, arcosh(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes arcosh(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes arcosh(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the inverse hyperbolic cosine of a specific value:
//	x := NewFloat().SetFloat64(1.5)
//	f := NewFloat()
//	f.Acosh(x) // f is now arcosh(1.5)
//
//	// Compute the inverse hyperbolic cosine of the receiver's value in place:
//	f.SetFloat64(2.0)
//	f.Acosh() // f is now arcosh(2.0)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called with no arguments, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The inverse hyperbolic cosine function, arcosh(x), is defined only for x >= 1; behavior
//     for x < 1 depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Acosh(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute arcosh(f) in place.
		C.mpfr_acosh(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute arcosh(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_acosh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Acosh sets f = arcosh(x) with rounding mode rnd.
func Acosh(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Acosh(x)
}

// Agm sets f = AGM(x, y) (arithmetic-geometric mean), using rnd.
func (f *Float) Agm(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_agm(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Agm returns AGM(x, y) (arithmetic-geometric mean), using rnd.
func Agm(x, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Agm(x, y)
}

// Asin computes the arcsine of a value, arcsin(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes arcsin(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes arcsin(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the arcsine of a specific value:
//	x := NewFloat().SetFloat64(0.5)
//	f := NewFloat()
//	f.Asin(x) // f is now arcsin(0.5)
//
//	// Compute the arcsine of the receiver's value in place:
//	f.SetFloat64(0.3)
//	f.Asin() // f is now arcsin(0.3)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called with no arguments, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The arcsine function is defined only for values in the range [-1, 1]; behavior outside this range
//     depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Asin(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		// Compute arcsin(f) in place.
		C.mpfr_asin(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute arcsin(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_asin(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Asin returns arcsin(x), using rnd.
func Asin(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Asin(x)
}

// Asinh computes the inverse hyperbolic sine of a value, arsinh(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes arsinh(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes arsinh(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the inverse hyperbolic sine of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Asinh(x) // f is now arsinh(1.0)
//
//	// Compute the inverse hyperbolic sine of the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Asinh() // f is now arsinh(0.5)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with no arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The inverse hyperbolic sine function arsinh(x) is well-defined for all real numbers.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Asinh(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute arsinh(f) in place.
		C.mpfr_asinh(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute arsinh(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_asinh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Asinh returns arsinh(x), using rnd.
func Asinh(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Asinh(x)
}

// Atan computes the arctangent of a value, arctan(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes arctan(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes arctan(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the arctangent of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Atan(x) // f is now arctan(1.0) = π/4
//
//	// Compute the arctangent of the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Atan() // f is now arctan(0.5)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with no arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The arctangent function is well-defined for all real numbers.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Atan(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute arctan(f) in place.
		C.mpfr_atan(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute arctan(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_atan(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Atan returns arctan(x), using rnd.
func Atan(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Atan(x)
}

// Atan2 sets f = arctan2(y, x) = angle whose tangent is y/x, using rnd.
func (f *Float) Atan2(y, x *Float) *Float {
	y.doinit()
	x.doinit()
	f.doinit()
	C.mpfr_atan2(&f.mpfr[0], &y.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Atan2 returns arctan2(y, x) = angle whose tangent is y/x, using rnd.
func Atan2(y, x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Atan2(y, x)
}

// Atanh computes the inverse hyperbolic tangent of a value, artanh(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes artanh(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes artanh(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the inverse hyperbolic tangent of a specific value:
//	x := NewFloat().SetFloat64(0.5)
//	f := NewFloat()
//	f.Atanh(x) // f is now artanh(0.5)
//
//	// Compute the inverse hyperbolic tangent of the receiver's value in place:
//	f.SetFloat64(0.3)
//	f.Atanh() // f is now artanh(0.3)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with no arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The function is undefined for |x| ≥ 1 (i.e., x <= -1 or x >= 1); behavior in such cases depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Atanh(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute artanh(f) in place.
		C.mpfr_atanh(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute artanh(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_atanh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Atanh returns artanh(x), using rnd.
func Atanh(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Atanh(x)
}

// Cbrt computes the cube root of a Float.
//
// If no arguments are passed, it computes the cube root of the receiver `f`
// in place, modifying `f` and returning it.
//
// If a single argument `x` is passed, it computes the cube root of `x` and
// stores the result in the receiver `f`, modifying `f` and returning it.
//
// Example Usage:
//
//	f := NewFloat().SetFloat64(8.0)
//	f.Cbrt() // f is now the cube root of 8 (2.0)
//
//	x := NewFloat().SetFloat64(27.0)
//	result := NewFloat().Cbrt(x) // result is now the cube root of 27 (3.0)
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Cbrt(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_cbrt(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_cbrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Cbrt returns cbrt(x), using rnd.
func Cbrt(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Cbrt(x)
}

// Sqrt computes the square root of the Float `x` and stores the result in the receiver `f`.
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// The receiver `f` is modified to hold the result, and the modified receiver is returned.
//
// Example Usage:
//
//	x := NewFloat().SetFloat64(9.0)
//	f := NewFloat()
//	f.Sqrt(x) // f is now the square root of 9 (3.0)
//
// Notes:
// - Both the receiver `f` and the argument `x` must be initialized before calling this function.
// - If `x` is negative, the behavior depends on the underlying C library's handling of invalid input.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Sqrt(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_sqrt(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_sqrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Sqrt returns sqrt(x), using rnd.
func Sqrt(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Sqrt(x)
}

// RootUI sets f to the k-th root of x with the specified rounding mode and returns f.
// If k is zero, it panics with an invalid argument error.
func (f *Float) RootUI(x *Float, k uint) *Float {
	x.doinit()
	f.doinit()

	if k == 0 {
		panic("Root: k must be greater than 0")
	}

	// Perform the root operation using mpfr_rootn_ui
	C.mpfr_rootn_ui(&f.mpfr[0], &x.mpfr[0], C.ulong(k), C.mpfr_rnd_t(f.RoundingMode))
	// check if NaN
	if C.mpfr_nan_p(&f.mpfr[0]) != 0 {
		panic("Root: result is NaN")
	}

	return f
}

// Ceil computes the ceiling of a Float and stores the result in the receiver `f`.
// The ceiling of a number is the smallest integral value greater than or equal to that number.
//
// If called with no arguments, the function computes the ceiling of the receiver `f` in place,
// modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the ceiling of `x` and stores
// the result in the receiver `f`, modifying `f` and returning it.
//
// Example Usage:
//
//	// Compute the ceiling of a new value:
//	x := NewFloat().SetFloat64(2.3)
//	f := NewFloat()
//	f.Ceil(x) // f is now 3.0
//
//	// Compute the ceiling in place:
//	f.SetFloat64(-2.7)
//	f.Ceil() // f is now -2.0
//
// Notes:
// - If called with an argument `x`, both `f` and `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Ceil(args ...*Float) *Float {
	f.doinit()
	var x *Float
	if len(args) > 0 && args[0] != nil {
		x = args[0]
		x.doinit()
	} else {
		x = f
	}
	C.mpfr_ceil(&f.mpfr[0], &x.mpfr[0]) // rop = mpfr_ceil(op)
	return f
}

// Ceil returns ceil(x), using rnd.
func Ceil(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Ceil(x)
}

// CmpAbs compares the absolute values of x and y, returning:
//
//	-1 if |x| <  |y|
//	 0 if |x| == |y|
//	+1 if |x| >  |y|
func CmpAbs(x, y *Float) int {
	x.doinit()
	y.doinit()
	return int(C.mpfr_cmpabs(&x.mpfr[0], &y.mpfr[0]))
}

// Cos computes the cosine of the Float `x` and stores the result in the receiver `f`.
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// If called with no arguments, it computes the cosine of the receiver `f` in place,
// modifying `f` and returning it.
//
// If called with a single argument `x`, it computes the cosine of `x` and stores
// the result in the receiver `f`, modifying `f` and returning it.
//
// Example Usage:
//
//	// Compute cosine of a new value:
//	x := NewFloat().SetFloat64(1.57) // Approx. pi/2
//	f := NewFloat()
//	f.Cos(x) // f is now the cosine of 1.57
//
//	// Compute cosine in place:
//	f.SetFloat64(3.14) // Approx. pi
//	f.Cos() // f is now the cosine of 3.14
//
// Notes:
// - The rounding mode is determined by the `RoundingMode` of the receiver `f`.
// - Both the receiver `f` and the argument `x` (if provided) must be initialized before calling this function.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Cos(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_cos(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_cos(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Cos returns cos(x), using rnd.
func Cos(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Cos(x)
}

// Cosh computes the hyperbolic cosine of a Float and stores the result in the receiver `f`.
// The hyperbolic cosine of a number `x` is defined as (e^x + e^(-x)) / 2.
//
// If called with no arguments, the function computes the hyperbolic cosine of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the hyperbolic cosine of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the hyperbolic cosine of a new value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Cosh(x) // f is now the hyperbolic cosine of 1.0
//
//	// Compute the hyperbolic cosine in place:
//	f.SetFloat64(2.0)
//	f.Cosh() // f is now the hyperbolic cosine of 2.0
//
// Notes:
// - If called with an argument `x`, both `f` and `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Cosh(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		C.mpfr_cosh(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_cosh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Cosh returns cosh(x), using rnd.
func Cosh(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Cosh(x)
}

// Cot computes the cotangent of a Float and stores the result in the receiver `f`.
// The cotangent of a number `x` is defined as 1 / tan(x).
//
// If called with no arguments, the function computes the cotangent of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the cotangent of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the cotangent of a new value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Cot(x) // f is now the cotangent of 1.0
//
//	// Compute the cotangent in place:
//	f.SetFloat64(2.0)
//	f.Cot() // f is now the cotangent of 2.0
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The cotangent function is undefined for integer multiples of π (e.g., x = 0, π, 2π, ...),
//     which may result in an error or undefined behavior depending on the underlying C library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Cot(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		C.mpfr_cot(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_cot(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Cot returns cot(x), using rnd.
func Cot(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Cot(x)
}

// Coth computes the hyperbolic cotangent of a Float and stores the result in the receiver `f`.
// The hyperbolic cotangent of a number `x` is defined as 1 / tanh(x).
//
// If called with no arguments, the function computes the hyperbolic cotangent of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the hyperbolic cotangent of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the hyperbolic cotangent of a new value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Coth(x) // f is now the hyperbolic cotangent of 1.0
//
//	// Compute the hyperbolic cotangent in place:
//	f.SetFloat64(2.0)
//	f.Coth() // f is now the hyperbolic cotangent of 2.0
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The hyperbolic cotangent function is undefined for `x = 0`, which may result in an error
//     or undefined behavior depending on the underlying C library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Coth(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_coth(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_coth(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Coth returns coth(x), using rnd.
func Coth(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Coth(x)
}

// Csc computes the cosecant of a Float and stores the result in the receiver `f`.
// The cosecant of a number `x` is defined as 1 / sin(x).
//
// If called with no arguments, the function computes the cosecant of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the cosecant of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the cosecant of a new value:
//	x := NewFloat().SetFloat64(1.0) // Input in radians
//	f := NewFloat()
//	f.Csc(x) // f is now the cosecant of 1.0
//
//	// Compute the cosecant in place:
//	f.SetFloat64(2.0)
//	f.Csc() // f is now the cosecant of 2.0
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The cosecant function is undefined for integer multiples of π (e.g., x = 0, π, 2π, ...),
//     which may result in an error or undefined behavior depending on the underlying C library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Csc(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_csc(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_csc(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Csc returns csc(x), using rnd.
func Csc(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Csc(x)
}

// Csch computes the hyperbolic cosecant of a Float and stores the result in the receiver `f`.
// The hyperbolic cosecant of a number `x` is defined as 1 / sinh(x).
//
// If called with no arguments, the function computes the hyperbolic cosecant of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the hyperbolic cosecant of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the hyperbolic cosecant of a new value:
//	x := NewFloat().SetFloat64(1.0) // Input in radians
//	f := NewFloat()
//	f.Csch(x) // f is now the hyperbolic cosecant of 1.0
//
//	// Compute the hyperbolic cosecant in place:
//	f.SetFloat64(2.0)
//	f.Csch() // f is now the hyperbolic cosecant of 2.0
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The hyperbolic cosecant function is undefined for `x = 0`, which may result in an error
//     or undefined behavior depending on the underlying C library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Csch(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_csch(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_csch(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// Csch returns csch(x), using rnd.
func Csch(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Csch(x)
}

// Exp10 sets f = 10^x
func (f *Float) Exp10(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp10(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Exp10 returns 10^x, using rnd
func Exp10(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Exp10(x)
}

// Exp2 sets f = 2^x
func (f *Float) Exp2(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp2(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Exp2 returns 2^x, using rnd
func Exp2(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Exp2(x)
}

// Floor computes the floor of a Float and stores the result in the receiver `f`.
// The floor of a number `x` is the largest integral value less than or equal to `x`.
//
// If called with no arguments, the function computes the floor of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the floor of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// Example Usage:
//
//	// Compute the floor of a new value:
//	x := NewFloat().SetFloat64(2.8)
//	f := NewFloat()
//	f.Floor(x) // f is now 2.0
//
//	// Compute the floor in place:
//	f.SetFloat64(-1.3)
//	f.Floor() // f is now -2.0
//
// Notes:
// - If called with an argument `x`, both `f` and `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - This function directly computes the mathematical floor and does not use a rounding mode.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Floor(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_floor(&f.mpfr[0], &f.mpfr[0])
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_floor(&f.mpfr[0], &x.mpfr[0])
	}
	return f
}

// Floor returns floor(x), using rnd
func Floor(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Floor(x)
}

// Fma sets f = (x * y) + z and returns f.
func (f *Float) Fma(x, y, z *Float) *Float {
	x.doinit()
	y.doinit()
	z.doinit()
	f.doinit()
	C.mpfr_fma(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], &z.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Fmma sets f = (a * b) + (c * d), with the given rounding mode, and returns f.
func (f *Float) Fmma(a, b, c, d *Float) *Float {
	a.doinit()
	b.doinit()
	c.doinit()
	d.doinit()
	f.doinit()
	C.mpfr_fmma(&f.mpfr[0], &a.mpfr[0], &b.mpfr[0], &c.mpfr[0], &d.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Fmms sets f = (a * b) - (c * d), with the given rounding mode, and returns f.
func (f *Float) Fmms(a, b, c, d *Float) *Float {
	a.doinit()
	b.doinit()
	c.doinit()
	d.doinit()
	f.doinit()
	C.mpfr_fmms(&f.mpfr[0], &a.mpfr[0], &b.mpfr[0], &c.mpfr[0], &d.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Fmod sets f to the floating-point remainder of x / y, with the given rounding mode, and returns f.
func (f *Float) Fmod(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_fmod(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Fmodquo computes the remainder (f = x mod y) and also returns the integer quotient via mpfr_fmodquo.
func (f *Float) Fmodquo(x, y *Float) (int, *Float) {
	x.doinit()
	y.doinit()
	f.doinit()
	var q C.long
	C.mpfr_fmodquo(&f.mpfr[0], &q, &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return int(q), f
}

// Fms sets f = (x * y) - z, with the given rounding mode, and returns f.
func (f *Float) Fms(x, y, z *Float) *Float {
	x.doinit()
	y.doinit()
	z.doinit()
	f.doinit()
	C.mpfr_fms(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], &z.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Frac computes the fractional part of a Float and stores the result in the receiver `f`.
// The fractional part is defined as:
//   - x - floor(x), if x ≥ 0
//   - x - ceil(x), if x < 0
//
// If called with no arguments, the function computes the fractional part of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the fractional part of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the fractional part of a new value:
//	x := NewFloat().SetFloat64(5.75)
//	f := NewFloat()
//	f.Frac(x) // f is now 0.75
//
//	// Compute the fractional part in place:
//	f.SetFloat64(-3.25)
//	f.Frac() // f is now -0.25
//
// Notes:
// - If called with an argument `x`, both `f` and `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Frac(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_frac(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_frac(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// FreeCache frees internal caches used by MPFR.
func FreeCache() {
	C.mpfr_free_cache()
}

// Gamma computes the Gamma function of a Float and stores the result in the receiver `f`.
// The Gamma function is a generalization of the factorial function, defined as:
//
//	Gamma(x) = ∫_0^∞ t^(x-1) * e^(-t) dt
//
// If called with no arguments, the function computes the Gamma function of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes the Gamma function of `x`
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the Gamma function of a new value:
//	x := NewFloat().SetFloat64(5.0) // Equivalent to 4! (24.0)
//	f := NewFloat()
//	f.Gamma(x) // f is now 24.0
//
//	// Compute the Gamma function in place:
//	f.SetFloat64(2.5)
//	f.Gamma() // f is now the Gamma of 2.5
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The Gamma function is undefined for non-positive integers (e.g., x = 0, -1, -2, ...),
//     which may result in an error or undefined behavior depending on the underlying C library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Gamma(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_gamma(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_gamma(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// GammaInc sets f = GammaInc(a, x) (the incomplete Gamma function) with
// rounding mode rnd, and returns f.
func (f *Float) GammaInc(a, x *Float) *Float {
	a.doinit()
	x.doinit()
	f.doinit()
	C.mpfr_gamma_inc(&f.mpfr[0], &a.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Greater returns true if the value of f is greater than x, false otherwise.
func (f *Float) Greater(x *Float) bool {
	f.doinit()
	x.doinit()
	return C.mpfr_greater_p(&f.mpfr[0], &x.mpfr[0]) != 0
}

// GreaterEqual returns true if the value of f is greater than or equal to x, false otherwise.
func (f *Float) GreaterEqual(x *Float) bool {
	f.doinit()
	x.doinit()
	return C.mpfr_greaterequal_p(&f.mpfr[0], &x.mpfr[0]) != 0
}

// Greater returns true if x > y, false otherwise.
func Greater(x, y *Float) bool {
	x.doinit()
	y.doinit()
	return C.mpfr_greater_p(&x.mpfr[0], &y.mpfr[0]) != 0
}

// GreaterEqual returns true if x >= y, false otherwise.
func GreaterEqual(x, y *Float) bool {
	x.doinit()
	y.doinit()
	return C.mpfr_greaterequal_p(&x.mpfr[0], &y.mpfr[0]) != 0
}

// Hypot sets f = sqrt(x^2 + y^2) with rounding mode rnd, and returns f.
func (f *Float) Hypot(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_hypot(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Hypot returns sqrt(x^2 + y^2), using rnd.
func Hypot(x, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Hypot(x, y)
}

// IsInf returns true if f is infinite, false otherwise.
func (f *Float) IsInf() bool {
	f.doinit()
	return C.mpfr_inf_p(&f.mpfr[0]) != 0
}

// IsInf returns true if x is infinite, false otherwise.
func IsInf(x *Float) bool {
	x.doinit()
	return C.mpfr_inf_p(&x.mpfr[0]) != 0
}

// J0 computes the Bessel function of the first kind of order 0, J₀(x),
// and stores the result in the receiver `f`.
//
// The Bessel function of the first kind of order 0, J₀(x), is a solution
// to the differential equation:
//
//	x²y'' + xy' + x²y = 0.
//
// If called with no arguments, the function computes J₀(f) in place,
// modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes J₀(x)
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute J₀(x) for a new value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.J0(x) // f is now J₀(1.0)
//
//	// Compute J₀ in place:
//	f.SetFloat64(2.5)
//	f.J0() // f is now J₀(2.5)
//
// Notes:
// - If called with an argument `x`, both `f` and `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The Bessel function J₀(x) is well-defined for all real values of `x`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) J0(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 && args[0] != nil {
		C.mpfr_j0(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_j0(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// J0 computes the Bessel function of the first kind of order 0, J₀(x),
func J0(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.J0(x)
}

// J1 computes the Bessel function of the first kind of order 1, J₁(x),
// and stores the result in the receiver `f`.
//
// The Bessel function of the first kind of order 1, J₁(x), is a solution
// to the differential equation:
//
//	x²y'' + xy' + (x² - 1)y = 0.
//
// If called with no arguments, the function computes J₁(f) in place,
// modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes J₁(x)
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute J₁(x) for a new value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.J1(x) // f is now J₁(1.0)
//
//	// Compute J₁ in place:
//	f.SetFloat64(2.5)
//	f.J1() // f is now J₁(2.5)
//
// Notes:
// - If called with an argument `x`, both `f` and `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The Bessel function J₁(x) is well-defined for all real values of `x`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) J1(args ...*Float) *Float {
	f.doinit()
	if len(args) == 0 {
		C.mpfr_j1(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		x := args[0]
		x.doinit()
		C.mpfr_j1(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// J1 computes the Bessel function of the first kind of order 1, J₁(x),
func J1(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.J1(x)
}

// Jn sets f = Jn(n, x) (the Bessel function of the first kind of order n) and returns f.
func (f *Float) Jn(n int, x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_jn(&f.mpfr[0], C.long(n), &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// LessThan returns true if the value of f is less than x, false otherwise.
func (f *Float) LessThan(x *Float) bool {
	f.doinit()
	x.doinit()
	return C.mpfr_less_p(&f.mpfr[0], &x.mpfr[0]) != 0
}

// LessEqual returns true if the value of f is less than or equal to x, false otherwise.
func (f *Float) LessEqual(x *Float) bool {
	f.doinit()
	x.doinit()
	return C.mpfr_lessequal_p(&f.mpfr[0], &x.mpfr[0]) != 0
}

// LessGreater returns true if the value of f is less than or greater to x, false otherwise.
func (f *Float) LessGreater(x *Float) bool {
	f.doinit()
	x.doinit()
	return C.mpfr_lessgreater_p(&f.mpfr[0], &x.mpfr[0]) != 0
}

// LessThan returns true if x < y, false otherwise.
func LessThan(x, y *Float) bool {
	x.doinit()
	y.doinit()
	return C.mpfr_less_p(&x.mpfr[0], &y.mpfr[0]) != 0
}

// LessEqual returns true if x <= y, false otherwise.
func LessEqual(x, y *Float) bool {
	x.doinit()
	y.doinit()
	return C.mpfr_lessequal_p(&x.mpfr[0], &y.mpfr[0]) != 0
}

// LessGreater returns true if x < y or x > y (i.e., x != y), false otherwise.
func LessGreater(x, y *Float) bool {
	x.doinit()
	y.doinit()
	return C.mpfr_lessgreater_p(&x.mpfr[0], &y.mpfr[0]) != 0
}

// Lgamma sets f = log|Gamma(x)| (the natural logarithm of the absolute value of Gamma(x)),
// and returns (sign, f), where sign is the sign of Gamma(x): -1, 0, or +1.
func (f *Float) Lgamma(x *Float) (int, *Float) {
	x.doinit()
	f.doinit()
	var sign C.int
	C.mpfr_lgamma(&f.mpfr[0], &sign, &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return int(sign), f
}

// Li2 computes the dilogarithm function, Li₂(x), and stores the result in the receiver `f`.
// The dilogarithm function, Li₂(x), is defined as:
//
//	Li₂(x) = -∫₀ˣ (ln(1 - t) / t) dt
//
// If called with no arguments, the function computes Li₂(f) in place,
// modifying `f` and returning it.
//
// If called with a single argument `x`, the function computes Li₂(x)
// and stores the result in the receiver `f`, modifying `f` and returning it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute Li₂(x) for a new value:
//	x := NewFloat().SetFloat64(0.5)
//	f := NewFloat()
//	f.Li2(x) // f is now Li₂(0.5)
//
//	// Compute Li₂ in place:
//	f.SetFloat64(-0.2)
//	f.Li2() // f is now Li₂(-0.2)
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The dilogarithm function, Li₂(x), is well-defined for real values of `x`, but special cases
//     (e.g., `x = 1` or `x = 0`) may return specific results, such as 0 for Li₂(0) or π²/6 for Li₂(1).
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Li2(args ...*Float) *Float {
	f.doinit()
	var x *Float
	if len(args) > 0 && args[0] != nil {
		x = args[0]
		x.doinit()
	} else {
		x = f
	}
	C.mpfr_li2(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Li2 computes the dilogarithm function, Li₂(x),
func Li2(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Li2(x)
}

// Lngamma computes the natural logarithm of the absolute value of the Gamma function,
// log|Γ(x)|, and stores the result in the receiver `f`. It returns a tuple (sign, *Float),
// where `sign` indicates the sign of Γ(x), and `f` holds the computed value log|Γ(x)|.
//
// The function is an alias for the MPFR function `mpfr_lgamma`. The naming may differ for
// compatibility purposes, but in many versions of MPFR, `mpfr_lngamma` is treated as a
// synonym for `mpfr_lgamma`.
//
// If called with no arguments, the function computes log|Γ(f)| in place, modifying `f` and
// returning the sign and the modified receiver.
//
// If called with a single argument `x`, the function computes log|Γ(x)| and stores the
// result in the receiver `f`, returning the sign and the modified receiver.
//
// Example Usage:
//
//	// Compute log|Γ(x)| for a new value:
//	x := NewFloat().SetFloat64(5.0) // Γ(5) = 24, so log|Γ(5)| = log(24)
//	f := NewFloat()
//	sign, result := f.Lngamma(x) // sign is 1 (positive), result holds log|Γ(5)|
//
//	// Compute log|Γ(f)| in place:
//	f.SetFloat64(-3.5)
//	sign, result := f.Lngamma() // sign reflects the sign of Γ(-3.5)
//
// Notes:
//   - If called with an argument `x`, both `f` and `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//   - The sign is an integer: `1` for a positive Γ(x) or `-1` for a negative Γ(x).
//   - The function is undefined for non-positive integers (e.g., x = 0, -1, -2, ...), and its
//     behavior in these cases depends on the underlying MPFR library.
//
// Returns:
//
//	A tuple (sign, *Float):
//	    - sign: An integer indicating the sign of Γ(x) (`1` for positive, `-1` for negative).
//	    - *Float: A pointer to the modified receiver `f`, holding the value log|Γ(x)|.
func (f *Float) Lngamma(args ...*Float) (int, *Float) {
	f.doinit()
	var x *Float
	if len(args) > 0 && args[0] != nil {
		x = args[0]
		x.doinit()
	} else {
		x = f
	}
	sign := C.mpfr_lngamma(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return int(sign), f
}

// Lngamma computes the natural logarithm of the absolute value of the Gamma function,
func Lngamma(x *Float, rnd Rnd) (int, *Float) {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Lngamma(x)
}

// Max computes the maximum value among the receiver `f` and any number of additional Float arguments.
// The result is stored in the receiver `f`, modifying it and returning it.
//
// The computation uses the rounding mode specified by the receiver `f`'s RoundingMode to handle
// ties or special cases (e.g., infinities or NaNs).
//
// If no additional arguments are provided, the function effectively does nothing, as `f` remains unchanged.
//
// Example Usage:
//
//	// Compute the maximum of multiple values:
//	x := NewFloat().SetFloat64(3.5)
//	y := NewFloat().SetFloat64(2.7)
//	z := NewFloat().SetFloat64(4.1)
//	f := NewFloat().SetFloat64(1.0)
//	f.Max(x, y, z) // f is now 4.1
//
//	// Compute the maximum in place with no additional arguments:
//	f.SetFloat64(-1.0)
//	f.Max() // f remains -1.0
//
// Notes:
// - All arguments (including the receiver `f`) must be initialized before calling this function.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - Special cases:
//   - If any argument is NaN, the behavior depends on the MPFR library's handling of NaNs.
//   - If multiple arguments are equal, the `RoundingMode` determines how ties are handled.
//
// Returns:
//
//	A pointer to the modified receiver `f`, holding the maximum value among all arguments.
func (f *Float) Max(args ...*Float) *Float {
	f.doinit()

	// Iterate through all provided arguments and update `f` with the maximum value.
	for _, x := range args {
		if x != nil {
			x.doinit()
			C.mpfr_max(&f.mpfr[0], &f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
		}
	}

	return f
}

// MaxFloat computes the maximum value among x and y, using the given rounding mode.
func MaxFloat(x, y *Float, rnd Rnd) *Float {
	x.SetRoundMode(rnd)
	return x.Max(y)
}

// Min computes the minimum value among the receiver `f` and any number of additional Float arguments.
// The result is stored in the receiver `f`, modifying it and returning it.
//
// The computation uses the rounding mode specified by the receiver `f`'s RoundingMode to handle
// ties or special cases (e.g., infinities or NaNs).
//
// If no additional arguments are provided, the function effectively does nothing, as `f` remains unchanged.
//
// Example Usage:
//
//	// Compute the minimum of multiple values:
//	x := NewFloat().SetFloat64(3.5)
//	y := NewFloat().SetFloat64(2.7)
//	z := NewFloat().SetFloat64(4.1)
//	f := NewFloat().SetFloat64(5.0)
//	f.Min(x, y, z) // f is now 2.7
//
//	// Compute the minimum in place with no additional arguments:
//	f.SetFloat64(1.0)
//	f.Min() // f remains 1.0
//
// Notes:
// - All arguments (including the receiver `f`) must be initialized before calling this function.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - Special cases:
//   - If any argument is NaN, the behavior depends on the MPFR library's handling of NaNs.
//   - If multiple arguments are equal, the `RoundingMode` determines how ties are handled.
//
// Returns:
//
//	A pointer to the modified receiver `f`, holding the minimum value among all arguments.
func (f *Float) Min(args ...*Float) *Float {
	f.doinit()

	// Iterate through all provided arguments and update `f` with the minimum value.
	for _, x := range args {
		if x != nil {
			x.doinit()
			C.mpfr_min(&f.mpfr[0], &f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
		}
	}

	return f
}

// MinFloat computes the minimum value among x and y, using the given rounding mode.
func MinFloat(x, y *Float, rnd Rnd) *Float {
	x.SetRoundMode(rnd)
	return x.Min(y)
}

// MinPrec returns the minimum of the precisions of x and y.
func MinPrec(x, y *Float) uint {
	x.doinit()
	y.doinit()

	// Compute the precision manually if the function signature changed
	precX := uint(C.mpfr_min_prec(&x.mpfr[0]))
	precY := uint(C.mpfr_min_prec(&y.mpfr[0]))
	if precX < precY {
		return precX
	}
	return precY
}

// Modf splits a value into its integer and fractional parts, with rounding mode specified.
// The result is:
//
//	x = intPart + fracPart
//
// Both the integer part and fractional part are returned as new `*Float` values. The signs of the
// parts follow MPFR's definition.
//
//   - If called with no arguments, the function splits the current value of the receiver `f` into
//     integer and fractional parts using the rounding mode of `f`.
//
//   - If called with one argument `x`, the function splits `x` into integer and fractional parts
//     using the rounding mode of `f`.
//
//   - If called with two arguments (`x` and `rnd`), the function splits `x` into integer and fractional
//     parts using the specified rounding mode `rnd`.
//
// Example Usage:
//
//	// Split a specific value into integer and fractional parts using the receiver's rounding mode:
//	x := NewFloat().SetFloat64(3.7)
//	intPart, fracPart := f.Modf(x)
//	// intPart is 3.0, fracPart is 0.7
//
//	// Split a specific value using a specified rounding mode:
//	x := NewFloat().SetFloat64(-2.8)
//	intPart, fracPart := f.Modf(x, RndDown)
//	// intPart is -3.0, fracPart is 0.2
//
//	// Split the receiver's value into integer and fractional parts:
//	f.SetFloat64(2.5)
//	intPart, fracPart := f.Modf()
//	// intPart is 2.0, fracPart is 0.5
//
// Notes:
//   - If `rnd` is not provided, the rounding mode of the receiver `f` is used.
//   - The function does not raise exceptions for special cases like infinities or NaNs; behavior
//     depends on the MPFR library.
//
// Returns:
//
//	Two pointers to `*Float` values: `(intPart, fracPart)`:
//	    - `intPart`: The integer part of the value.
//	    - `fracPart`: The fractional part of the value.
func (f *Float) Modf(args ...interface{}) (intPart, fracPart *Float) {
	intPart = NewFloat()
	fracPart = NewFloat()

	if len(args) == 0 {
		// Called with no arguments: use the current value of `f` and its rounding mode.
		f.doinit()
		C.mpfr_modf(&intPart.mpfr[0], &fracPart.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else if len(args) == 1 {
		// Called with one argument: interpret as `Modf(x)` and use `f.RoundingMode`.
		if x, ok := args[0].(*Float); ok {
			x.doinit()
			C.mpfr_modf(&intPart.mpfr[0], &fracPart.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
		} else if rnd, rOk := args[0].(Rnd); rOk {
			// Called with one argument: interpret as `Modf(rnd)`.
			f.doinit()
			C.mpfr_modf(&intPart.mpfr[0], &fracPart.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(rnd))
		} else {
			panic("Modf expects a *Float as the first argument")
		}
	} else if len(args) > 1 {
		// Called with two arguments: interpret as `Modf(x, rnd)`.
		if x, xOk := args[0].(*Float); xOk {
			if rnd, rOk := args[1].(Rnd); rOk {
				x.doinit()
				C.mpfr_modf(&intPart.mpfr[0], &fracPart.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
			} else {
				panic("Modf expects an Rnd as the second argument")
			}
		} else {
			panic("Modf expects a *Float as the first argument")
		}
	}

	return intPart, fracPart
}

// Modf splits a value into its integer and fractional parts, with rounding mode specified.
func Modf(x *Float, rnd Rnd) (intPart, fracPart *Float) {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Modf(x)
}

// MPMemoryCleanup releases any memory that MPFR might be caching for internal purposes.
func MPMemoryCleanup() {
	C.mpfr_mp_memory_cleanup()
}

// Neg negates the value of the receiver `f` or any number of provided Float arguments
// and stores the result in the receiver `f`.
//
// If called with no arguments, the function negates the current value of the receiver `f`
// in place, modifying `f` and returning it.
//
// If called with one or more arguments, the function iteratively negates the provided values,
// storing the first negated value in the receiver `f`.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Negate a single value:
//	x := NewFloat().SetFloat64(3.5)
//	f := NewFloat()
//	f.Neg(x) // f is now -3.5
//
//	// Negate the receiver's value in place:
//	f.SetFloat64(-2.7)
//	f.Neg() // f is now 2.7
//
//	// Negate multiple values (result holds the negation of the last argument):
//	y := NewFloat().SetFloat64(-4.1)
//	z := NewFloat().SetFloat64(-5.0)
//	f.Neg(x, y, z) // f is now 4.1
//
// Notes:
// - If called with arguments, all provided Float values must be initialized before the call.
// - If called without arguments, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - For multiple arguments, the result stored in `f` corresponds to the negation of the last argument.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Neg(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Negate the receiver's value in place.
		C.mpfr_neg(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// negate the first non-nil argument and store the result in f
		for _, x := range args {
			if x != nil {
				x.doinit()
				C.mpfr_neg(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
				break
			}
		}
	}

	return f
}

// NextAbove sets the receiver `f` to the next representable floating-point value
// above its current value or the value of `x` (toward +∞).
//
// If called with no arguments, the function increments the value of the receiver `f`
// to the next representable floating-point value above its current value, modifying `f`
// and returning it.
//
// If called with one argument `x`, the function sets `f` to the value of `x` and
// increments it to the next representable floating-point value above `x`, modifying `f`
// and returning it.
//
// Example Usage:
//
//	// Increment the value of x to the next representable value above:
//	x := NewFloat().SetFloat64(3.5)
//	f := NewFloat()
//	f.NextAbove(x) // f is now the next representable value above 3.5
//
//	// Increment the receiver's value in place:
//	f.SetFloat64(2.7)
//	f.NextAbove() // f is now the next representable value above 2.7
//
// Notes:
// - If called with an argument `x`, `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The function respects the precision and rounding mode of the receiver `f`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) NextAbove(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Increment the receiver's value in place.
		C.mpfr_nextabove(&f.mpfr[0])
	} else {
		// Set the receiver `f` to the value of `x`, then increment `f`.
		x := args[0]
		x.doinit()
		C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
		C.mpfr_nextabove(&f.mpfr[0])
	}

	return f
}

// NextAbove returns the next representable floating-point value above x.
func NextAbove(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.NextAbove(x)
}

// NextBelow sets the receiver `f` to the next representable floating-point value
// below its current value or the value of `x` (toward -∞).
//
// If called with no arguments, the function decrements the value of the receiver `f`
// to the next representable floating-point value below its current value, modifying `f`
// and returning it.
//
// If called with one argument `x`, the function sets `f` to the value of `x` and
// decrements it to the next representable floating-point value below `x`, modifying `f`
// and returning it.
//
// Example Usage:
//
//	// Decrement the value of x to the next representable value below:
//	x := NewFloat().SetFloat64(3.5)
//	f := NewFloat()
//	f.NextBelow(x) // f is now the next representable value below 3.5
//
//	// Decrement the receiver's value in place:
//	f.SetFloat64(2.7)
//	f.NextBelow() // f is now the next representable value below 2.7
//
// Notes:
// - If called with an argument `x`, `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The function respects the precision and rounding mode of the receiver `f`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) NextBelow(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Decrement the receiver's value in place.
		C.mpfr_nextbelow(&f.mpfr[0])
	} else if len(args) == 1 && args[0] != nil {
		// Set the receiver `f` to the value of `x`, then decrement `f`.
		x := args[0]
		x.doinit()
		C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
		C.mpfr_nextbelow(&f.mpfr[0])
	} else {
		// Restrict to 0 or 1 arguments only.
		panic("NextBelow accepts 0 or 1 arguments only")
	}

	return f
}

// NextBelow returns the next representable floating-point value below x.
func NextBelow(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.NextBelow(x)
}

// NextToward sets the receiver `f` to the next representable floating-point value
// based on the provided arguments.
//
//   - If called with one argument `x`, the function moves `f` one step toward the value of `x`.
//     This modifies `f` in place and returns it.
//
//   - If called with two arguments `x` and `y`, the function sets `f` to the value of `x` and
//     moves it one step in the direction of `y`. This modifies `f` and returns it.
//
// Example Usage:
//
//	// Move f toward a single target value:
//	x := NewFloat().SetFloat64(3.5)
//	f := NewFloat().SetFloat64(2.0)
//	f.NextToward(x) // f moves one step toward 3.5
//
//	// Move from x toward y:
//	x := NewFloat().SetFloat64(3.5)
//	y := NewFloat().SetFloat64(4.0)
//	f.NextToward(x, y) // f is set to 3.5, then moves one step toward 4.0
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called with two arguments, both `x` and `y` must be initialized before the call.
// - The computation respects the precision and rounding mode of the receiver `f`.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) NextToward(args ...*Float) *Float {
	f.doinit()

	if len(args) == 1 && args[0] != nil {
		// Move `f` one step toward `x`.
		x := args[0]
		x.doinit()
		C.mpfr_nexttoward(&f.mpfr[0], &x.mpfr[0])
	} else {
		// Set `f` to `x` and move it one step in the direction of `y`.
		x, y := args[0], args[1]
		x.doinit()
		y.doinit()
		C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
		C.mpfr_nexttoward(&f.mpfr[0], &y.mpfr[0])
	}

	return f
}

// NextToward returns the next representable floating-point value toward x.
func NextToward(x *Float, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.NextToward(x, y)
}

// RecSqrt computes the reciprocal square root, 1 / sqrt(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes 1 / sqrt(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes 1 / sqrt(x) and stores the result in `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the reciprocal square root of a value:
//	x := NewFloat().SetFloat64(4.0)
//	f := NewFloat()
//	f.RecSqrt(x) // f is now 1 / sqrt(4.0) = 0.5
//
//	// Compute the reciprocal square root in place:
//	f.SetFloat64(9.0)
//	f.RecSqrt() // f is now 1 / sqrt(9.0) = 0.333...
//
// Notes:
// - If called with an argument `x`, `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The function assumes that the input value is positive; behavior is undefined for negative values.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) RecSqrt(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute 1 / sqrt(f) in place.
		C.mpfr_rec_sqrt(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute 1 / sqrt(x) and store in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_rec_sqrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}
	return f
}

// RecSqrt computes the reciprocal square root, 1 / sqrt(x),
func RecSqrt(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.RecSqrt(x)
}

// IsRegular returns true if f is a normal (regular) number.
// This excludes zeros, subnormals, infinities, and NaN.
func (f *Float) IsRegular() bool {
	f.doinit()
	return C.mpfr_regular_p(&f.mpfr[0]) != 0
}

// IsRegular returns true if x is a normal (regular) number.
// This excludes zeros, subnormals, infinities, and NaN.
func IsRegular(x *Float) bool {
	x.doinit()
	return C.mpfr_regular_p(&x.mpfr[0]) != 0
}

// Reldiff sets f = the relative difference between x and y, i.e. |x - y| / max(|x|, |y|) and returns f.
func (f *Float) Reldiff(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_reldiff(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Reldiff sets f = the relative difference between x and y, i.e. |x - y| / max(|x|, |y|),
// using the specified rounding mode, and returns f.
func Reldiff(x, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Reldiff(x, y)
}

// Remainder sets f = x - n * y, where n is an integer chosen so that f is in (-|y|/2, |y|/2].
func (f *Float) Remainder(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_remainder(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Remainder returns x - n * y, where n is an integer chosen so that the returned value is in (-|y|/2, |y|/2],
func Remainder(x, y *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Remainder(x, y)
}

// Remquo sets f = remainder of x / y, and also returns the integer quotient in an int.
// The remainder is computed such that f is in (-|y|/2, |y|/2] (similar to mpfr_remainder).
func (f *Float) Remquo(x, y *Float) (int, *Float) {
	x.doinit()
	y.doinit()
	f.doinit()
	var q C.long
	C.mpfr_remquo(&f.mpfr[0], &q, &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return int(q), f
}

// Remquo returns the remainder of x / y, and also returns the integer quotient in an int.
func Remquo(x, y *Float, rnd Rnd) (int, *Float) {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Remquo(x, y)
}

// Round sets the receiver `f` to the nearest integer value based on the current MPFR rounding mode,
// which is normally "round to nearest, ties away from zero".
//
//   - If called with no arguments, the function rounds the current value of the receiver `f` in place.
//     This modifies `f` and returns it.
//
//   - If called with one argument `x`, the function rounds `x` to the nearest integer value and stores
//     the result in the receiver `f`. This modifies `f` and returns it.
//
// Example Usage:
//
//	// Round a specific value:
//	x := NewFloat().SetFloat64(3.6)
//	f := NewFloat()
//	f.Round(x) // f is now 4.0
//
//	// Round the receiver's value in place:
//	f.SetFloat64(-2.4)
//	f.Round() // f is now -2.0
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The rounding mode used depends on the receiver `f`'s rounding mode (default: ties away from zero).
//   - This function does not raise exceptions for special cases like infinities or NaNs; the behavior
//     depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Round(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Round the current value of the receiver `f` in place.
		C.mpfr_round(&f.mpfr[0], &f.mpfr[0])
	} else {
		// Round `x` and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_round(&f.mpfr[0], &x.mpfr[0])
	}

	return f
}

// Round returns the nearest integer value based on the current MPFR rounding mode.
func Round(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Round(x)
}

// RoundEven sets the receiver `f` to the nearest integer value based on the "round to nearest, ties to even" rule.
// This is also known as "bankers' rounding."
//
//   - If called with no arguments, the function rounds the current value of the receiver `f` in place.
//     This modifies `f` and returns it.
//
//   - If called with one argument `x`, the function rounds `x` to the nearest integer value with ties to even
//     and stores the result in the receiver `f`. This modifies `f` and returns it.
//
// Example Usage:
//
//	// Round a specific value using ties-to-even rule:
//	x := NewFloat().SetFloat64(2.5)
//	f := NewFloat()
//	f.RoundEven(x) // f is now 2.0 (2.5 rounds to 2.0)
//
//	// Round the receiver's value in place:
//	f.SetFloat64(3.5)
//	f.RoundEven() // f is now 4.0 (3.5 rounds to 4.0)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - This function always uses the "round to nearest, ties to even" rule, regardless of the rounding mode of `f`.
// - The function does not raise exceptions for special cases like infinities or NaNs; the behavior depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) RoundEven(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Round the current value of the receiver `f` in place using ties-to-even.
		C.mpfr_roundeven(&f.mpfr[0], &f.mpfr[0])
	} else {
		// Round `x` using ties-to-even and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_roundeven(&f.mpfr[0], &x.mpfr[0])
	}

	return f
}

// RoundEven returns x rounded to the nearest integer, with ties to even.
func RoundEven(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.RoundEven(x)
}

// Sec computes the secant of a value, sec(x) = 1 / cos(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes sec(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes sec(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the secant of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Sec(x) // f is now 1 / cos(1.0)
//
//	// Compute the secant of the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Sec() // f is now 1 / cos(0.5)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The secant function is undefined for odd multiples of π/2 (e.g., x = ±π/2, ±3π/2, ...),
//     and the behavior in these cases depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Sec(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute sec(f) in place.
		C.mpfr_sec(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute sec(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_sec(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Sec computes the secant of a value, sec(x) = 1 / cos(x).
func Sec(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Sec(x)
}

// Sech computes the hyperbolic secant of a value, sech(x) = 1 / cosh(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes sech(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes sech(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the hyperbolic secant of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Sech(x) // f is now 1 / cosh(1.0)
//
//	// Compute the hyperbolic secant of the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Sech() // f is now 1 / cosh(0.5)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The hyperbolic secant is well-defined for all real numbers.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Sech(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute sech(f) in place.
		C.mpfr_sech(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute sech(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_sech(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Sech computes the hyperbolic secant of a value, sech(x) = 1 / cosh(x).
func Sech(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Sech(x)
}

// Swap exchanges the contents of f and x (their mantissa, sign, exponent, etc.).
func (f *Float) Swap(x *Float) {
	f.doinit()
	x.doinit()
	C.mpfr_swap(&f.mpfr[0], &x.mpfr[0])
}

// Tan computes the tangent of a value, tan(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes tan(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes tan(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the tangent of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Tan(x) // f is now tan(1.0)
//
//	// Compute the tangent of the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Tan() // f is now tan(0.5)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The tangent function is undefined for odd multiples of π/2 (e.g., x = ±π/2, ±3π/2, ...),
//     and the behavior in these cases depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Tan(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute tan(f) in place.
		C.mpfr_tan(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute tan(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_tan(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Tan computes the tangent of a value, tan(x).
func Tan(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Tan(x)
}

// Tanh computes the hyperbolic tangent of a value, tanh(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes tanh(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes tanh(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the hyperbolic tangent of a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Tanh(x) // f is now tanh(1.0)
//
//	// Compute the hyperbolic tangent of the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Tanh() // f is now tanh(0.5)
//
// Notes:
// - If called with one argument, `x` must be initialized before the call.
// - If called without an argument, only the receiver `f` must be initialized.
// - The computation uses the `RoundingMode` of the receiver `f`.
// - The hyperbolic tangent function, tanh(x), is well-defined for all real numbers.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Tanh(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute tanh(f) in place.
		C.mpfr_tanh(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else if len(args) == 1 && args[0] != nil {
		// Compute tanh(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_tanh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Restrict to 0 or 1 arguments only.
		panic("Tanh accepts 0 or 1 arguments only")
	}

	return f
}

// Tanh computes the hyperbolic tangent of a value, tanh(x).
func Tanh(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Tanh(x)
}

// Trunc computes the integer part of a value truncated toward zero and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function truncates the current value of `f` toward zero.
//     This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function truncates `x` toward zero and stores the result in `f`.
//     This modifies `f` and returns it.
//
// Example Usage:
//
//	// Truncate a specific value:
//	x := NewFloat().SetFloat64(3.7)
//	f := NewFloat()
//	f.Trunc(x) // f is now 3.0
//
//	// Truncate the receiver's value in place:
//	f.SetFloat64(-2.8)
//	f.Trunc() // f is now -2.0
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The truncation operation always rounds toward zero, regardless of the rounding mode of `f`.
//   - The function does not raise exceptions for special cases like infinities or NaNs; behavior
//     depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Trunc(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Truncate the current value of `f` in place.
		C.mpfr_trunc(&f.mpfr[0], &f.mpfr[0])
	} else {
		// Truncate `x` and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_trunc(&f.mpfr[0], &x.mpfr[0])
	}

	return f
}

// Trunc returns the integer part of x truncated toward zero.
func Trunc(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Trunc(x)
}

// Y0 computes the Bessel function of the second kind of order 0, Y₀(x),
// and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes Y₀(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes Y₀(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the Bessel function of the second kind for a specific value:
//	x := NewFloat().SetFloat64(1.0)
//	f := NewFloat()
//	f.Y0(x) // f is now Y₀(1.0)
//
//	// Compute the Bessel function of the second kind for the receiver's value in place:
//	f.SetFloat64(0.5)
//	f.Y0() // f is now Y₀(0.5)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The Bessel function Y₀(x) is undefined for non-positive values of `x` (x <= 0),
//     and the behavior in these cases depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Y0(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute Y₀(f) in place.
		C.mpfr_y0(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute Y₀(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_y0(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Y0 computes the Bessel function of the second kind of order 0, Y₀(x).
func Y0(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Y0(x)
}

// Y1 computes the Bessel function of the second kind of order 1, Y₁(x),
// and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes Y₁(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes Y₁(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the Bessel function of the second kind for a specific value:
//	x := NewFloat().SetFloat64(2.0)
//	f := NewFloat()
//	f.Y1(x) // f is now Y₁(2.0)
//
//	// Compute the Bessel function of the second kind for the receiver's value in place:
//	f.SetFloat64(1.0)
//	f.Y1() // f is now Y₁(1.0)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The Bessel function Y₁(x) is undefined for non-positive values of `x` (x <= 0),
//     and the behavior in these cases depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Y1(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute Y₁(f) in place.
		C.mpfr_y1(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute Y₁(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_y1(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Y1 computes the Bessel function of the second kind of order 1, Y₁(x).
func Y1(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Y1(x)
}

// Yn sets f = Yn(n, x) (the Bessel function of the second kind of order n),
// using rounding mode rnd, and returns f.
func (f *Float) Yn(n int, x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_yn(&f.mpfr[0], C.long(n), &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Yn returns Yn(n, x) (the Bessel function of the second kind of order n).
func Yn(n int, x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Yn(n, x)
}

// IsZero returns true if f is exactly zero (positive or negative zero).
func (f *Float) IsZero() bool {
	f.doinit()
	return C.mpfr_zero_p(&f.mpfr[0]) != 0
}

// IsZero returns true if x is exactly zero (positive or negative zero).
func IsZero(x *Float) bool {
	x.doinit()
	return C.mpfr_zero_p(&x.mpfr[0]) != 0
}

// Zeta computes the Riemann zeta function, ζ(x), and stores the result in the receiver `f`.
//
//   - If called with no arguments, the function computes ζ(f), where `f` is the current value
//     of the receiver. This modifies `f` in place and returns it.
//
//   - If called with one argument `x`, the function computes ζ(x) and stores the result in the receiver `f`.
//     This modifies `f` and returns it.
//
// The result is computed using the rounding mode specified by the receiver `f`'s RoundingMode.
//
// Example Usage:
//
//	// Compute the Riemann zeta function for a specific value:
//	x := NewFloat().SetFloat64(2.0)
//	f := NewFloat()
//	f.Zeta(x) // f is now ζ(2.0)
//
//	// Compute the Riemann zeta function for the receiver's value in place:
//	f.SetFloat64(3.0)
//	f.Zeta() // f is now ζ(3.0)
//
// Notes:
//   - If called with one argument, `x` must be initialized before the call.
//   - If called without an argument, only the receiver `f` must be initialized.
//   - The computation uses the `RoundingMode` of the receiver `f`.
//   - The Riemann zeta function ζ(x) is undefined for x = 1 (pole at x = 1),
//     and behavior for x ≤ 0 depends on the MPFR library.
//
// Returns:
//
//	A pointer to the modified receiver `f`.
func (f *Float) Zeta(args ...*Float) *Float {
	f.doinit()

	if len(args) == 0 {
		// Compute ζ(f) in place.
		C.mpfr_zeta(&f.mpfr[0], &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Compute ζ(x) and store the result in `f`.
		x := args[0]
		x.doinit()
		C.mpfr_zeta(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	}

	return f
}

// Zeta computes the Riemann zeta function, ζ(x).
func Zeta(x *Float, rnd Rnd) *Float {
	f := NewFloat()
	f.SetRoundMode(rnd)
	return f.Zeta(x)
}

// SetPrec sets the precision of the Float to the specified number of bits.
// This method changes the precision and clears the content of f, so the value will need to be reinitialized.
func (f *Float) SetPrec(prec uint) *Float {
	f.doinit()
	originalValue := f.String()
	C.mpfr_set_prec(&f.mpfr[0], C.mpfr_prec_t(prec))
	_ = f.SetString(originalValue, 10)
	return f
}

// FitsIntmax returns true if f (rounded by rnd) fits in an intmax_t.
func (f *Float) FitsIntmax() bool {
	f.doinit()
	return C.mpfr_fits_intmax_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsSint returns true if f (rounded by rnd) fits in a signed int.
func (f *Float) FitsSint() bool {
	f.doinit()
	return C.mpfr_fits_sint_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsSlong returns true if f (rounded by rnd) fits in a signed long.
func (f *Float) FitsSlong() bool {
	f.doinit()
	return C.mpfr_fits_slong_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsSshort returns true if f (rounded by rnd) fits in a signed short.
func (f *Float) FitsSshort() bool {
	f.doinit()
	return C.mpfr_fits_sshort_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsUint returns true if f (rounded by rnd) fits in an unsigned int.
func (f *Float) FitsUint() bool {
	f.doinit()
	return C.mpfr_fits_uint_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsUintmax returns true if f (rounded by rnd) fits in a uintmax_t.
func (f *Float) FitsUintmax() bool {
	f.doinit()
	return C.mpfr_fits_uintmax_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsUlong returns true if f (rounded by rnd) fits in an unsigned long.
func (f *Float) FitsUlong() bool {
	f.doinit()
	return C.mpfr_fits_ulong_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// FitsUshort returns true if f (rounded by rnd) fits in an unsigned short.
func (f *Float) FitsUshort() bool {
	f.doinit()
	return C.mpfr_fits_ushort_p(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)) != 0
}

// ErrInvalidString is returned when mpfr_set_str fails to parse a string.
var ErrInvalidString = &FloatError{"invalid string for mpfr_set_str"}

// FloatError is a simple error type for mpfr-related errors.
type FloatError struct {
	Msg string
}

func (e *FloatError) Error() string {
	return e.Msg
}

// FromInt initializes an MPFR Float from a Go int.
func FromInt(value int) *Float {
	f := NewFloat()
	C.mpfr_set_si(&f.mpfr[0], C.long(value), C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// FromInt64 initializes an MPFR Float from a Go int64.
// TODO: needs a better implementation that doesn't rely on string conversion
func FromInt64(value int64) *Float {
	f := NewFloat()
	if value >= math.MinInt32 && value <= math.MaxInt32 {
		// Use mpfr_set_si directly for smaller values
		C.mpfr_set_si(&f.mpfr[0], C.long(value), C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Use a math/big.Int for larger values
		bigVal := big.NewInt(value)
		return FromBigInt(bigVal)
	}
	return f
}

// FromUint64 initializes an MPFR Float from a Go uint64.
// TODO: needs a better implementation that doesn't rely on string conversion
func FromUint64(value uint64) *Float {
	f := NewFloat()
	if value <= math.MaxUint32 {
		// Use mpfr_set_ui directly for smaller values
		C.mpfr_set_ui(&f.mpfr[0], C.ulong(value), C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Use a math/big.Int for larger values
		bigVal := new(big.Int).SetUint64(value)
		return FromBigInt(bigVal)
	}
	return f
}

// FromFloat64 initializes an MPFR Float from a Go float64.
func FromFloat64(value float64) *Float {
	f := NewFloat()
	C.mpfr_set_d(&f.mpfr[0], C.double(value), C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// FromBigInt initializes an MPFR Float from a math/big.Int.
// TODO: needs a better implementation that doesn't rely on string conversion
func FromBigInt(value *big.Int) *Float {
	f := NewFloat()
	if value == nil {
		C.mpfr_set_zero(&f.mpfr[0], 1) // Initialize to zero
		return f
	}

	// Convert math/big.Int to a string and parse with MPFR
	str := value.Text(10)
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.mpfr_rnd_t(f.RoundingMode)) != 0 {
		panic("FromBigInt: failed to parse big.Int")
	}

	return f
}

// FromBigFloat initializes an MPFR Float from a math/big.Float.
// TODO: needs a better implementation that doesn't rely on string conversion
func FromBigFloat(value *big.Float) *Float {
	f := NewFloat()
	if value == nil {
		C.mpfr_set_zero(&f.mpfr[0], 1) // Initialize to zero
		return f
	}

	// Convert math/big.Float to a string, then parse with MPFR
	str := value.Text('g', -1) // Decimal format
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.mpfr_rnd_t(f.RoundingMode)) != 0 {
		panic("FromBigFloat: failed to parse big.Float")
	}

	return f
}

// SetRoundMode sets the rounding mode for the Float.
func (f *Float) SetRoundMode(rnd Rnd) {
	f.RoundingMode = rnd
}

// SetInt sets the value of the Float to the specified int.
func (f *Float) SetInt(value int) *Float {
	f.doinit()
	C.mpfr_set_si(&f.mpfr[0], C.long(value), C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// SetInt64 sets the value of the Float to the specified int64.
func (f *Float) SetInt64(value int64) *Float {
	f.doinit()
	if value >= math.MinInt32 && value <= math.MaxInt32 {
		// Use mpfr_set_si directly for smaller values
		C.mpfr_set_si(&f.mpfr[0], C.long(value), C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Use a math/big.Int for larger values
		bigVal := big.NewInt(value)
		f.SetBigInt(bigVal)
	}
	return f
}

// SetUint64 sets the value of the Float to the specified uint64.
// TODO: needs a better implementation that doesn't rely on string conversion
func (f *Float) SetUint64(value uint64) *Float {
	f.doinit()
	if value <= math.MaxUint32 {
		// Use mpfr_set_ui directly for smaller values
		C.mpfr_set_ui(&f.mpfr[0], C.ulong(value), C.mpfr_rnd_t(f.RoundingMode))
	} else {
		// Use a math/big.Int for larger values
		bigVal := new(big.Int).SetUint64(value)
		f.SetBigInt(bigVal)
	}
	return f
}

func (f *Float) SetFloat64(value float64) *Float {
	f.doinit()
	C.mpfr_set_d(&f.mpfr[0], C.double(value), C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// SetBigInt sets the value of the Float to the specified math/big.Int.
// TODO: needs a better implementation that doesn't rely on string conversion
func (f *Float) SetBigInt(value *big.Int) *Float {
	f.doinit()
	if value == nil {
		C.mpfr_set_zero(&f.mpfr[0], 1) // Set to zero if nil
		return f
	}

	// Convert math/big.Int to string and set it using mpfr_set_str
	str := value.Text(10)
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.mpfr_rnd_t(f.RoundingMode)) != 0 {
		panic("SetBigInt: failed to parse big.Int")
	}
	return f
}

// SetBigFloat sets the value of the Float to the specified math/big.Float.
// TODO: needs a better implementation that doesn't rely on string conversion
func (f *Float) SetBigFloat(value *big.Float) *Float {
	f.doinit()
	if value == nil {
		C.mpfr_set_zero(&f.mpfr[0], 1) // Set to zero if nil
		return f
	}

	// Convert math/big.Float to string and set it using mpfr_set_str
	str := value.Text('g', -1)
	cstr := C.CString(str)
	defer C.free(unsafe.Pointer(cstr))

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.mpfr_rnd_t(f.RoundingMode)) != 0 {
		panic("SetBigFloat: failed to parse big.Float")
	}
	return f
}

// Int64 converts the Float to an int64.
// After the conversion, the Float is cleared to conserve memory.
func (f *Float) Int64() int64 {
	// Clean up the Float after use
	defer f.Clear()
	return int64(C.mpfr_get_si(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)))
}

// Uint64 converts the Float to a uint64.
// After the conversion, the Float is cleared to conserve memory.
func (f *Float) Uint64() uint64 {
	// Clean up the Float after use
	defer f.Clear()
	return uint64(C.mpfr_get_ui(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)))
}

// Float64 converts the Float to a float64.
// After the conversion, the Float is cleared to conserve memory.
func (f *Float) Float64() float64 {
	// Clean up the Float after use
	defer f.Clear()
	return float64(C.mpfr_get_d(&f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode)))
}

// BigInt converts the Float to a math/big.Int.
// It writes the result into the provided big.Int and clears the Float after conversion.
// TODO: needs a better implementation that doesn't rely on string conversion
func (f *Float) BigInt(result *big.Int) {
	defer f.Clear()

	var exp C.mpfr_exp_t
	cstr := C.mpfr_get_str(nil, &exp, 10, 0, &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	if cstr == nil {
		panic("BigInt: mpfr_get_str failed")
	}
	defer C.mpfr_free_str(cstr)

	mantissa := C.GoString(cstr)

	reneg := false
	if mantissa[0] == '-' {
		reneg = true
		mantissa = mantissa[1:]
	}

	// Handle cases where the exponent is larger than or within the length of the mantissa
	if int(exp) >= len(mantissa) {
		mantissa += strings.Repeat("0", int(exp)-len(mantissa))
	} else if int(exp) < len(mantissa) && exp > 0 {
		// Insert a decimal point at the correct position
		mantissa = mantissa[:int(exp)] + "." + mantissa[int(exp):]
	} else if exp < 0 {
		// Handle negative exponents: prepend zeros
		mantissa = "0." + strings.Repeat("0", -int(exp)) + mantissa
	}

	if reneg == true {
		mantissa = "-" + mantissa
	}

	result.SetString(mantissa, 10)
}

// BigFloat converts the Float to a math/big.Float.
// It writes the result into the provided big.Float and clears the Float after conversion.
// TODO: needs a better implementation that doesn't rely on string conversion
func (f *Float) BigFloat(result *big.Float) {
	defer f.Clear() // Clean up the Float after use

	var exp C.mpfr_exp_t
	cstr := C.mpfr_get_str(nil, &exp, 10, 0, &f.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	if cstr == nil {
		panic("BigFloat: mpfr_get_str failed")
	}
	defer C.mpfr_free_str(cstr)

	mantissa := C.GoString(cstr)

	reneg := false
	if mantissa[0] == '-' {
		reneg = true
		mantissa = mantissa[1:]
	}
	// Handle cases where the exponent is larger than or within the length of the mantissa
	if int(exp) >= len(mantissa) {
		mantissa += strings.Repeat("0", int(exp)-len(mantissa))
	} else if int(exp) < len(mantissa) && exp > 0 {
		// Insert a decimal point at the correct position
		mantissa = mantissa[:int(exp)] + "." + mantissa[int(exp):]
	} else if exp < 0 {
		// Handle negative exponents: prepend zeros
		mantissa = "0." + strings.Repeat("0", -int(exp)) + mantissa
	}

	if reneg == true {
		mantissa = "-" + mantissa
	}

	// Parse the formatted mantissa into a big.Float
	result.SetString(mantissa)
}
