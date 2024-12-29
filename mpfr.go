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
	"runtime"
	"strconv"
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
// TODO: MPFR has more modes...
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
	f.SetFloat64(0.0, RoundToNearest)
	return f
}

// SetFloat64 sets f to the given float64 value (x), using the specified rounding mode.
func (f *Float) SetFloat64(x float64, rnd Rnd) *Float {
	f.doinit()
	C.mpfr_set_d(&f.mpfr[0], C.double(x), C.mpfr_rnd_t(rnd))
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

// Cmp compares f and x and returns -1 if f < x, 0 if f == x, +1 if f > x.
func (f *Float) Cmp(x *Float) int {
	f.doinit()
	x.doinit()
	return int(C.mpfr_cmp(&f.mpfr[0], &x.mpfr[0]))
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
