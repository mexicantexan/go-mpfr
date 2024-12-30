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
	"strconv"
	"strings"
	"unsafe"
)

// Float represents a multiple-precision floating-point number.
// The zero value for Float is *not* valid until initialized
// (similar to how the GMP code works).
// Float represents a multiple-precision floating-point number.
type Float struct {
	mpfr C.mpfr_t // Use C.mpfr_t directly (array of 1 struct)
	init bool
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
	C.mpfr_init(&f.mpfr[0]) // Pass a pointer to the first element

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

// GetFloat64 returns the float64 approximation of f, using the specified rounding mode.
func (f *Float) GetFloat64(rnd Rnd) float64 {
	f.doinit()
	return float64(C.mpfr_get_d(&f.mpfr[0], C.mpfr_rnd_t(rnd)))
}

// SetString parses a string into f.
func (f *Float) SetString(s string, base int, rnd Rnd) error {
	f.doinit()
	cstr := C.CString(s)
	defer C.free(unsafe.Pointer(cstr))
	ret := C.mpfr_set_str(&f.mpfr[0], cstr, C.int(base), C.mpfr_rnd_t(rnd))
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
	cstr := C.mpfr_get_str(nil, &exp, C.int(base), 0, &f.mpfr[0], C.MPFR_RNDN)
	if cstr == nil {
		return "<mpfr_get_str_error>"
	}
	defer C.mpfr_free_str(cstr)

	mantissa := C.GoString(cstr)
	return mantissa + "e" + strconv.FormatInt(int64(exp), 10)
}

// Copy sets f to x, copying the entire mpfr_t.
func (f *Float) Copy(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_set(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(RoundToNearest))
	return f
}

// Add sets f to x + y, with the given rounding mode, and returns f.
func (f *Float) Add(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_add(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Sub sets f to x - y, with the given rounding mode, and returns f.
func (f *Float) Sub(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_sub(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Mul sets f to x * y, with the given rounding mode, and returns f.
func (f *Float) Mul(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_mul(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Div sets f to x / y, with the given rounding mode, and returns f.
func (f *Float) Div(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_div(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Pow sets f to x^y (x raised to the power y), with the given rounding mode, and returns f.
func (f *Float) Pow(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_pow(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Exp sets f to e^x (the exponential of x), with the given rounding mode, and returns f.
func (f *Float) Exp(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Log sets f to the natural logarithm of x (ln(x)), with the given rounding mode, and returns f.
// If x <= 0, MPFR will return NaN or -Inf depending on the input value.
func (f *Float) Log(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_log(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Cmp compares f and x and returns -1 if f < x, 0 if f == x, +1 if f > x.
func (f *Float) Cmp(x *Float) int {
	f.doinit()
	x.doinit()
	return int(C.mpfr_cmp(&f.mpfr[0], &x.mpfr[0]))
}

// Abs sets f = |x| (absolute value of x), using the specified rounding mode.
func (f *Float) Abs(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_abs(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Acos sets f = arccos(x) with rounding mode rnd.
func (f *Float) Acos(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_acos(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Acosh sets f = arcosh(x) with rounding mode rnd.
func (f *Float) Acosh(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_acosh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Agm sets f = AGM(x, y) (arithmetic-geometric mean), using rnd.
func (f *Float) Agm(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_agm(&f.mpfr[0], &x.mpfr[0], &y.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Asin sets f = arcsin(x), using rnd.
func (f *Float) Asin(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_asin(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Asinh sets f = arsinh(x), using rnd.
func (f *Float) Asinh(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_asinh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Atan sets f = arctan(x), using rnd.
func (f *Float) Atan(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_atan(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Atan2 sets f = arctan2(y, x) = angle whose tangent is y/x, using rnd.
// The signature follows (y, x, rnd) convention of mpfr_atan2.
func (f *Float) Atan2(y, x *Float, rnd Rnd) *Float {
	y.doinit()
	x.doinit()
	f.doinit()
	C.mpfr_atan2(&f.mpfr[0], &y.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Atanh sets f = artanh(x), using rnd.
func (f *Float) Atanh(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_atanh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Cbrt sets f = cbrt(x), using rnd.
func (f *Float) Cbrt(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cbrt(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
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
func (f *Float) Cos(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cos(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Cosh sets f = cosh(x), using rnd.
func (f *Float) Cosh(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cosh(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Cot sets f = cot(x) = 1 / tan(x), using rnd.
func (f *Float) Cot(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_cot(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Coth sets f = coth(x) = 1 / tanh(x), using rnd.
func (f *Float) Coth(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_coth(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Csc sets f = csc(x) = 1 / sin(x), using rnd.
func (f *Float) Csc(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_csc(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Csch sets f = csch(x) = 1 / sinh(x), using rnd.
func (f *Float) Csch(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_csch(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Exp10 sets f = 10^x, using rnd.
func (f *Float) Exp10(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp10(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Exp2 sets f = 2^x, using rnd.
func (f *Float) Exp2(x *Float, rnd Rnd) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_exp2(&f.mpfr[0], &x.mpfr[0], C.mpfr_rnd_t(rnd))
	return f
}

// Floor sets f = floor(x), i.e. the largest integral value <= x.
func (f *Float) Floor(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_floor(&f.mpfr[0], &x.mpfr[0])
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
func (f *Float) FitsIntmax(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_intmax_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsSint returns true if f (rounded by rnd) fits in a signed int.
func (f *Float) FitsSint(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_sint_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsSlong returns true if f (rounded by rnd) fits in a signed long.
func (f *Float) FitsSlong(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_slong_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsSshort returns true if f (rounded by rnd) fits in a signed short.
func (f *Float) FitsSshort(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_sshort_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsUint returns true if f (rounded by rnd) fits in an unsigned int.
func (f *Float) FitsUint(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_uint_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsUintmax returns true if f (rounded by rnd) fits in a uintmax_t.
func (f *Float) FitsUintmax(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_uintmax_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsUlong returns true if f (rounded by rnd) fits in an unsigned long.
func (f *Float) FitsUlong(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_ulong_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
}

// FitsUshort returns true if f (rounded by rnd) fits in an unsigned short.
func (f *Float) FitsUshort(rnd Rnd) bool {
	f.doinit()
	return C.mpfr_fits_ushort_p(&f.mpfr[0], C.mpfr_rnd_t(rnd)) != 0
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
	C.mpfr_set_si(&f.mpfr[0], C.long(value), C.MPFR_RNDN)
	return f
}

// FromInt64 initializes an MPFR Float from a Go int64.
// TODO: needs a better implementation that doesn't rely on string conversion
func FromInt64(value int64) *Float {
	f := NewFloat()
	if value >= math.MinInt32 && value <= math.MaxInt32 {
		// Use mpfr_set_si directly for smaller values
		C.mpfr_set_si(&f.mpfr[0], C.long(value), C.MPFR_RNDN)
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
		C.mpfr_set_ui(&f.mpfr[0], C.ulong(value), C.MPFR_RNDN)
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
	C.mpfr_set_d(&f.mpfr[0], C.double(value), C.MPFR_RNDN)
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

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.MPFR_RNDN) != 0 {
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

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.MPFR_RNDN) != 0 {
		panic("FromBigFloat: failed to parse big.Float")
	}

	return f
}

// SetInt sets the value of the Float to the specified int.
func (f *Float) SetInt(value int) *Float {
	f.doinit()
	C.mpfr_set_si(&f.mpfr[0], C.long(value), C.MPFR_RNDN)
	return f
}

// SetInt64 sets the value of the Float to the specified int64.
func (f *Float) SetInt64(value int64) *Float {
	f.doinit()
	if value >= math.MinInt32 && value <= math.MaxInt32 {
		// Use mpfr_set_si directly for smaller values
		C.mpfr_set_si(&f.mpfr[0], C.long(value), C.MPFR_RNDN)
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
		C.mpfr_set_ui(&f.mpfr[0], C.ulong(value), C.MPFR_RNDN)
	} else {
		// Use a math/big.Int for larger values
		bigVal := new(big.Int).SetUint64(value)
		f.SetBigInt(bigVal)
	}
	return f
}

func (f *Float) SetFloat64(value float64) *Float {
	f.doinit()
	C.mpfr_set_d(&f.mpfr[0], C.double(value), C.MPFR_RNDN)
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

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.MPFR_RNDN) != 0 {
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

	if C.mpfr_set_str(&f.mpfr[0], cstr, 10, C.MPFR_RNDN) != 0 {
		panic("SetBigFloat: failed to parse big.Float")
	}
	return f
}

// Int64 converts the Float to an int64.
// After the conversion, the Float is cleared to conserve memory.
func (f *Float) Int64() int64 {
	// Clean up the Float after use
	defer f.Clear()
	return int64(C.mpfr_get_si(&f.mpfr[0], C.MPFR_RNDN))
}

// Uint64 converts the Float to a uint64.
// After the conversion, the Float is cleared to conserve memory.
func (f *Float) Uint64() uint64 {
	// Clean up the Float after use
	defer f.Clear()
	return uint64(C.mpfr_get_ui(&f.mpfr[0], C.MPFR_RNDN))
}

// Float64 converts the Float to a float64.
// After the conversion, the Float is cleared to conserve memory.
func (f *Float) Float64() float64 {
	// Clean up the Float after use
	defer f.Clear()
	return float64(C.mpfr_get_d(&f.mpfr[0], C.MPFR_RNDN))
}

// BigInt converts the Float to a math/big.Int.
// It writes the result into the provided big.Int and clears the Float after conversion.
// TODO: needs a better implementation that doesn't rely on string conversion
func (f *Float) BigInt(result *big.Int) {
	defer f.Clear()

	var exp C.mpfr_exp_t
	cstr := C.mpfr_get_str(nil, &exp, 10, 0, &f.mpfr[0], C.MPFR_RNDN)
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
	cstr := C.mpfr_get_str(nil, &exp, 10, 0, &f.mpfr[0], C.MPFR_RNDN)
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
