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

// We could wrap certain MPFR functions here if needed to handle
// version differences or extra logic, similar to how your GMP code
// has _mpz_* wrappers. For now, we’ll call MPFR functions directly
// from Go via cgo.
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Float represents a multiple-precision floating-point number.
// The zero value for Float is *not* valid until initialized
// (similar to how the GMP code works).
type Float struct {
	mpfr C.mpfr_t
	init bool
}

// finalizer is called by the garbage collector when there are no
// more references to this Float. It clears the mpfr_t and releases
// native memory.
func finalizer(f *Float) {
	if f.init {
		C.mpfr_clear(&f.mpfr)
		f.init = false
	}
}

// doinit initializes f.mpfr if it isn’t already initialized.
func (f *Float) doinit() {
	if f.init {
		return
	}
	f.init = true
	C.mpfr_init(&f.mpfr)
	runtime.SetFinalizer(f, finalizer)
}

// Clear deallocates the native mpfr_t. After calling Clear,
// the Float must not be used again.
//
// If you don’t call Clear manually, the Go GC will eventually
// call the finalizer to free memory.
func (f *Float) Clear() {
	finalizer(f)
}

// Rnd is the type for MPFR rounding modes.
//
// NOTE: MPFR has more modes, but here are the common ones.
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
	return f
}

// SetFloat64 sets f to the given float64 value (x), using the specified rounding mode.
func (f *Float) SetFloat64(x float64, rnd Rnd) *Float {
	f.doinit()
	C.mpfr_set_d(&f.mpfr, C.double(x), C.mpfr_rnd_t(rnd))
	return f
}

// GetFloat64 returns the float64 approximation of f,
// using the specified rounding mode.
func (f *Float) GetFloat64(rnd Rnd) float64 {
	f.doinit()
	return float64(C.mpfr_get_d(&f.mpfr, C.mpfr_rnd_t(rnd)))
}

// SetString parses a string into f. The string should
// be in a format recognized by mpfr_set_str, e.g. "3.14159" or "1.23e4".
// base can be 2..36, or 0 to let MPFR auto-detect base 10.
func (f *Float) SetString(s string, base int, rnd Rnd) error {
	f.doinit()
	cstr := C.CString(s)
	defer C.free(unsafe.Pointer(cstr))
	ret := C.mpfr_set_str(&f.mpfr, cstr, C.int(base), C.mpfr_rnd_t(rnd))
	if ret != 0 {
		// MPFR returns 0 if successful, or nonzero on parse error
		return ErrInvalidString
	}
	return nil
}

// String returns f in a (base=10) string representation.
// This is a simplistic example; real usage might consider
// precision, format, etc.
func (f *Float) String() string {
	f.doinit()
	// We’ll ask mpfr to print it with some default precision,
	// then convert it to a Go string.
	// For a more robust approach, you might want to control
	// the precision or use mpfr_get_str directly.
	const outBase = 10
	// Let’s guess a buffer size (like 128 bytes). In real usage
	// you might do something more dynamic.
	bufSize := 128
	buf := make([]byte, bufSize)
	// mpfr_sprintf: mpfr_*printf can write into a C string. We'll do that.
	//
	// int mpfr_sprintf (char *str, const char *format, ..., mpfr_t, ...)
	// A simple format might be "%.Re".
	format := C.CString("%.Re") // R = current precision, e = exponent form
	defer C.free(unsafe.Pointer(format))

	// We need a pointer to the first element of our Go byte slice.
	cbuf := (*C.char)(unsafe.Pointer(&buf[0]))

	// mpfr_sprintf returns the number of characters written (excluding null terminator).
	n := C.mpfr_sprintf(cbuf, C.size_t(bufSize), format, &f.mpfr)
	if n < 0 {
		return "<mpfr_sprintf_error>"
	}
	// Convert from C buffer to Go string up to n bytes.
	return string(buf[:n])
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

// Copy sets f to x, copying the entire mpfr_t.
func (f *Float) Copy(x *Float) *Float {
	x.doinit()
	f.doinit()
	C.mpfr_set(&f.mpfr, &x.mpfr, C.mpfr_rnd_t(RoundToNearest))
	return f
}

// Add sets f to x + y, with the given rounding mode, and returns f.
func (f *Float) Add(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_add(&f.mpfr, &x.mpfr, &y.mpfr, C.mpfr_rnd_t(rnd))
	return f
}

// Sub sets f to x - y, with the given rounding mode, and returns f.
func (f *Float) Sub(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_sub(&f.mpfr, &x.mpfr, &y.mpfr, C.mpfr_rnd_t(rnd))
	return f
}

// Mul sets f to x * y, with the given rounding mode, and returns f.
func (f *Float) Mul(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_mul(&f.mpfr, &x.mpfr, &y.mpfr, C.mpfr_rnd_t(rnd))
	return f
}

// Div sets f to x / y, with the given rounding mode, and returns f.
// If y == 0, MPFR handles division by zero by setting special values
// (Inf, NaN) according to its rules. We do not panic here.
func (f *Float) Div(x, y *Float, rnd Rnd) *Float {
	x.doinit()
	y.doinit()
	f.doinit()
	C.mpfr_div(&f.mpfr, &x.mpfr, &y.mpfr, C.mpfr_rnd_t(rnd))
	return f
}

// Cmp compares f and x and returns:
//   -1 if f <  x
//    0 if f == x
//   +1 if f >  x
func (f *Float) Cmp(x *Float) int {
	f.doinit()
	x.doinit()
	return int(C.mpfr_cmp(&f.mpfr, &x.mpfr))
}
