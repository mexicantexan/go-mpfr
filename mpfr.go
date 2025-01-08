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
	"fmt"
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

// Add sets f to x + y, with the given rounding mode, and returns f.
func (f *Float) Add(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_add(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Sub sets f to x - y, with the given rounding mode, and returns f.
func (f *Float) Sub(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_sub(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Mul sets f to x * y, with the given rounding mode, and returns f.
func (f *Float) Mul(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_mul(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Div sets f to x / y, with the given rounding mode, and returns f.
func (f *Float) Div(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_div(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
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

	result := C.mpfr_div(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	if result != 0 {
		// Debug Log inexact results for debugging, but don't panic
		// fmt.Printf("Quo: inexact result with code = %d\n", result)
	}

	return f
}

// Pow sets f to x^y (x raised to the power y), with the given rounding mode, and returns f.
func (f *Float) Pow(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_pow(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Exp sets f to e^x (the exponential of x), with the given rounding mode, and returns f.
func (f *Float) Exp(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Log sets f to the natural logarithm of x (ln(x)), with the given rounding mode, and returns f.
// If x <= 0, MPFR will return NaN or -Inf depending on the input value.
func (f *Float) Log(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_log(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Cmp compares f and x and returns -1 if f < x, 0 if f == x, +1 if f > x.
func (f *Float) Cmp(x *Float) int {
	f.doinit()
	x.doinit()
	return int(C.mpfr_cmp(&f.mpfr[0], &x.mpfr[0]))
}

// Abs sets f = |x| (absolute value of x), using the specified rounding mode.
func (f *Float) Abs(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_abs(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Acos sets f = arccos(x) with rounding mode rnd.
func (f *Float) Acos(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_acos(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Acosh sets f = arcosh(x) with rounding mode rnd.
func (f *Float) Acosh(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_acosh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Agm sets f = AGM(x, y) (arithmetic-geometric mean), using rnd.
func (f *Float) Agm(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_agm(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Asin sets f = arcsin(x), using rnd.
func (f *Float) Asin(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_asin(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Asinh sets f = arsinh(x), using rnd.
func (f *Float) Asinh(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_asinh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Atan sets f = arctan(x), using rnd.
func (f *Float) Atan(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_atan(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Atan2 sets f = arctan2(y, x) = angle whose tangent is y/x, using rnd.
// The signature follows (y, x, rnd) convention of mpfr_atan2.
func (f *Float) Atan2(y, x *Float) *Float {
	y.doinit()
	x.doinit()
	f.doinit()
	C.mpfr_atan2(&f.mpfr[0], &y.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Atanh sets f = artanh(x), using rnd.
func (f *Float) Atanh(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_atanh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Cbrt sets f = cbrt(x), using rnd.
func (f *Float) Cbrt(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cbrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Sqrt sets f to the square root of x with the specified rounding mode and returns f.
func (f *Float) Sqrt(x *Float) *Float {
	x.doinit()
	f.doinit()

	// Compute the square root
	res := C.mpfr_sqrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	if res != 0 && C.mpfr_nan_p(&f.mpfr[0]) != 0 {
		panic(fmt.Sprintf("Sqrt: failed to compute square root for %s; MPFR result code: %v", x.String(), res))
	}

	return f
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
	res := C.mpfr_rootn_ui(&f.mpfr[0], &x.mpfr[0], C.ulong(k), C.mpfr_rnd_t(f.RoundingMode))
	if res != 0 && C.mpfr_nan_p(&f.mpfr[0]) != 0 {
		panic(fmt.Sprintf("Root: failed to compute the %d-th root", k))
	}

	// check if NaN
	if C.mpfr_nan_p(&f.mpfr[0]) != 0 {
		panic("Root: result is NaN")
	}

	return f
}

// Ceil sets f = ceil(x), i.e. the smallest integral value >= x.
func (f *Float) Ceil(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_ceil(&f.mpfr[0], &x.mpfr[0]) // rop = mpfr_ceil(op)
	return f
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

// Cos sets f = cos(x), using rnd.
func (f *Float) Cos(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cos(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Cosh sets f = cosh(x), using rnd.
func (f *Float) Cosh(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cosh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Cot sets f = cot(x) = 1 / tan(x), using rnd.
func (f *Float) Cot(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cot(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Coth sets f = coth(x) = 1 / tanh(x), using rnd.
func (f *Float) Coth(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_coth(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Csc sets f = csc(x) = 1 / sin(x), using rnd.
func (f *Float) Csc(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_csc(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Csch sets f = csch(x) = 1 / sinh(x), using rnd.
func (f *Float) Csch(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_csch(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Exp10 sets f = 10^x, using rnd.
func (f *Float) Exp10(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp10(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Exp2 sets f = 2^x, using rnd.
func (f *Float) Exp2(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp2(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Floor sets f = floor(x), i.e. the largest integral value <= x.
func (f *Float) Floor(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_floor(&f.mpfr[0], &x.mpfr[0])
	return f
}

// Fma sets f = (x * y) + z, with the given rounding mode, and returns f.
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

// Frac sets f to the fractional part of x, using the given rounding mode, and returns f.
// The fractional part is defined in MPFR as x - floor(x) if x ≥ 0, or x - ceil(x) if x < 0.
func (f *Float) Frac(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_frac(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// FreeCache frees internal caches used by MPFR.
func FreeCache() {
	C.mpfr_free_cache()
}

// Gamma sets f = Gamma(x) (the Gamma function of x) with rounding mode rnd,
// and returns f.
func (f *Float) Gamma(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_gamma(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
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

// Inf returns true if f is infinite, false otherwise.
func (f *Float) Inf() bool {
	f.doinit()
	return C.mpfr_inf_p(&f.mpfr[0]) != 0
}

// Inf returns true if x is infinite, false otherwise.
func Inf(x *Float) bool {
	x.doinit()
	return C.mpfr_inf_p(&x.mpfr[0]) != 0
}

// J0 sets f = J0(x) (the Bessel function of the first kind of order 0),
// using rounding mode rnd, and returns f.
func (f *Float) J0(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_j0(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// J1 sets f = J1(x) (the Bessel function of the first kind of order 1),
// using rounding mode rnd, and returns f.
func (f *Float) J1(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_j1(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Jn sets f = Jn(n, x) (the Bessel function of the first kind of order n),
// using rounding mode rnd, and returns f.
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

// Li2 sets f = li2(x) (the dilogar function), with rounding mode rnd, and returns f.
func (f *Float) Li2(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_li2(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Lngamma sets f = log|Gamma(x)| (an alias of mpfr_lgamma), and returns (sign, f).
// The difference from mpfr_lgamma might be purely naming/compatibility related,
// but MPFR treats mpfr_lngamma as a synonym for mpfr_lgamma in many versions.
func (f *Float) Lngamma(x *Float) (int, *Float) {
	x.doinit()
	f.doinit()
	sign := C.mpfr_lngamma(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return int(sign), f
}

// Max sets f to the maximum of x and y (like max(x, y)), using rnd to handle ties or special cases.
func (f *Float) Max(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_max(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Min sets f to the minimum of x and y (like min(x, y)), using rnd if needed.
func (f *Float) Min(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_min(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
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

// Modf splits x into integer and fractional parts, with rounding mode rnd.
// It returns two new *Float values: (intPart, fracPart).
//
//	x = intPart + fracPart
//
// The signs of the parts follow MPFR's definition.
func Modf(x *Float, rnd Rnd) (intPart, fracPart *Float) {
	x.doinit()
	intPart = NewFloat()
	fracPart = NewFloat()
	C.mpfr_modf(&intPart.mpfr[0], &fracPart.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return intPart, fracPart
}

// MPMemoryCleanup releases any memory that MPFR might be caching for internal purposes.
func MPMemoryCleanup() {
	C.mpfr_mp_memory_cleanup()
}

// Neg sets f = -x, with rounding mode rnd, and returns f.
func (f *Float) Neg(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_neg(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// NextAbove sets f to the next representable floating-point value above x (toward +∞).
// Then returns f.
func (f *Float) NextAbove(x *Float) *Float {
	x.doinit()
	f.doinit()
	// Copy x into f, then move f one step above.
	C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	C.mpfr_nextabove(&f.mpfr[0])
	return f
}

// NextBelow sets f to the next representable floating-point value below x (toward -∞).
// Then returns f.
func (f *Float) NextBelow(x *Float) *Float {
	x.doinit()
	f.doinit()
	// Copy x into f, then move f one step below.
	C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	C.mpfr_nextbelow(&f.mpfr[0])
	return f
}

// NextToward sets f to the next representable floating-point value from x in the direction of y.
// Then returns f.
func (f *Float) NextToward(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	C.mpfr_nexttoward(&f.mpfr[0], &y.mpfr[0])
	return f
}

// RecSqrt sets f = 1 / sqrt(x), using the specified rounding mode, and returns f.
func (f *Float) RecSqrt(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_rec_sqrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
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

// Reldiff sets f = the relative difference between x and y, i.e. |x - y| / max(|x|, |y|),
// using the specified rounding mode, and returns f.
func (f *Float) Reldiff(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_reldiff(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Remainder sets f = x - n * y, where n is an integer chosen so that f is in (-|y|/2, |y|/2].
func (f *Float) Remainder(x, y *Float) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_remainder(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
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

// Round sets f to the nearest integer value of x using the current MPFR rounding mode
// (which is normally “round to nearest, ties away from zero”).
func (f *Float) Round(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_round(&f.mpfr[0], &x.mpfr[0])
	return f
}

// RoundEven sets f = x rounded to the nearest integer, with ties to even.
func (f *Float) RoundEven(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_roundeven(&f.mpfr[0], &x.mpfr[0])
	return f
}

// Sec sets f = sec(x) = 1/cos(x), with rounding mode rnd, and returns f.
func (f *Float) Sec(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_sec(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Sech sets f = sech(x) = 1/cosh(x), with rounding mode rnd, and returns f.
func (f *Float) Sech(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_sech(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Swap exchanges the contents of f and x (their mantissa, sign, exponent, etc.).
func (f *Float) Swap(x *Float) {
	f.doinit()
	x.doinit()
	C.mpfr_swap(&f.mpfr[0], &x.mpfr[0])
}

// Tan sets f = tan(x), using rounding mode rnd, and returns f.
func (f *Float) Tan(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_tan(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Tanh sets f = tanh(x), using rounding mode rnd, and returns f.
func (f *Float) Tanh(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_tanh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Trunc sets f = the integer part of x, truncated toward zero, and returns f.
func (f *Float) Trunc(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_trunc(&f.mpfr[0], &x.mpfr[0])
	return f
}

// Y0 sets f = Y0(x) (the Bessel function of the second kind of order 0), using rnd, and returns f.
func (f *Float) Y0(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_y0(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Y1 sets f = Y1(x) (the Bessel function of the second kind of order 1), using rnd, and returns f.
func (f *Float) Y1(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_y1(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// Yn sets f = Yn(n, x) (the Bessel function of the second kind of order n),
// using rounding mode rnd, and returns f.
func (f *Float) Yn(n int, x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_yn(&f.mpfr[0], C.long(n), &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
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

// Zeta sets f = ζ(x) (the Riemann zeta function), using rounding mode rnd, and returns f.
func (f *Float) Zeta(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_zeta(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(f.RoundingMode))
	return f
}

// SetPrec sets the precision of the Float to the specified number of bits.
// This method changes the precision and clears the content of f, so the value will need to be reinitialized.
func (f *Float) SetPrec(prec uint) *Float {
	f.doinit()
	C.mpfr_set_prec(&f.mpfr[0], C.mpfr_prec_t(prec))
	f.SetFloat64(0.0)
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
