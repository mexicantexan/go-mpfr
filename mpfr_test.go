package mpfr_test

import (
	"mpfr"
	"testing"
)

func TestNewFloat(t *testing.T) {
	f := mpfr.NewFloat()

	// Check if it starts as 0.0
	got := f.GetFloat64(mpfr.RoundToNearest)
	want := 0.0
	if got != want {
		t.Errorf("NewFloat() = %v; want %v", got, want)
	}
}

func TestSetFloat64GetFloat64(t *testing.T) {
	f := mpfr.NewFloat()

	// Set to 3.1415
	f.SetFloat64(3.1415, mpfr.RoundToNearest)
	got := f.GetFloat64(mpfr.RoundToNearest)
	want := 3.1415
	if got != want {
		t.Errorf("SetFloat64(3.1415) / GetFloat64() = %v; want %v", got, want)
	}
}

func TestSetString(t *testing.T) {
	f := mpfr.NewFloat()

	// Valid string
	err := f.SetString("3.25", 10, mpfr.RoundToNearest)
	if err != nil {
		t.Fatalf("SetString(\"3.25\") returned error: %v", err)
	}

	got := f.GetFloat64(mpfr.RoundToNearest)
	want := 3.25
	if got != want {
		t.Errorf("SetString(\"3.25\") = %v; want %v", got, want)
	}

	// Invalid string
	err = f.SetString("not-a-number", 10, mpfr.RoundToNearest)
	if err == nil {
		t.Error("SetString(\"not-a-number\") = nil error; want non-nil error")
	}
}

func TestStringMethod(t *testing.T) {
	f := mpfr.NewFloat()
	f.SetFloat64(1.2345, mpfr.RoundToNearest)

	s := f.String()
	if s == "" {
		t.Errorf("String() returned empty; want a numeric string representation")
	}
}

func TestAdd(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(1.5, mpfr.RoundToNearest)
	y := mpfr.NewFloat().SetFloat64(2.25, mpfr.RoundToNearest)
	sum := mpfr.NewFloat().Add(x, y, mpfr.RoundToNearest)

	got := sum.GetFloat64(mpfr.RoundToNearest)
	want := 3.75
	if got != want {
		t.Errorf("1.5 + 2.25 = %v; want %v", got, want)
	}
}

func TestSub(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(5.0, mpfr.RoundToNearest)
	y := mpfr.NewFloat().SetFloat64(3.1, mpfr.RoundToNearest)
	diff := mpfr.NewFloat().Sub(x, y, mpfr.RoundToNearest)

	got := diff.GetFloat64(mpfr.RoundToNearest)
	want := 1.9
	// Floating-point inexactness is possible, so you might do an approximate check
	if got != want {
		t.Errorf("5.0 - 3.1 = %v; want %v", got, want)
	}
}

func TestMul(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(2.0, mpfr.RoundToNearest)
	y := mpfr.NewFloat().SetFloat64(3.25, mpfr.RoundToNearest)
	product := mpfr.NewFloat().Mul(x, y, mpfr.RoundToNearest)

	got := product.GetFloat64(mpfr.RoundToNearest)
	want := 6.5
	if got != want {
		t.Errorf("2.0 * 3.25 = %v; want %v", got, want)
	}
}

func TestDiv(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(10.0, mpfr.RoundToNearest)
	y := mpfr.NewFloat().SetFloat64(4.0, mpfr.RoundToNearest)
	quotient := mpfr.NewFloat().Div(x, y, mpfr.RoundToNearest)

	got := quotient.GetFloat64(mpfr.RoundToNearest)
	want := 2.5
	if got != want {
		t.Errorf("10.0 / 4.0 = %v; want %v", got, want)
	}
}

func TestCmp(t *testing.T) {
	a := mpfr.NewFloat().SetFloat64(2.0, mpfr.RoundToNearest)
	b := mpfr.NewFloat().SetFloat64(2.0, mpfr.RoundToNearest)
	c := mpfr.NewFloat().SetFloat64(3.0, mpfr.RoundToNearest)

	if a.Cmp(b) != 0 {
		t.Errorf("Cmp(2.0, 2.0) != 0; want 0 (equal)")
	}
	if a.Cmp(c) >= 0 {
		t.Errorf("Cmp(2.0, 3.0) >= 0; want < 0")
	}
	if c.Cmp(a) <= 0 {
		t.Errorf("Cmp(3.0, 2.0) <= 0; want > 0")
	}
}

func TestCopy(t *testing.T) {
	orig := mpfr.NewFloat().SetFloat64(1.23, mpfr.RoundToNearest)
	copy := mpfr.NewFloat().Copy(orig)

	if copy.Cmp(orig) != 0 {
		t.Errorf("Copy() doesn't match original: orig=%v copy=%v", orig, copy)
	}

	// Modify the original and ensure copy is unaffected.
	orig.SetFloat64(9.99, mpfr.RoundToNearest)
	if copy.Cmp(orig) == 0 {
		t.Errorf("Changing the original also changed the copy; want them independent.")
	}
}

func TestClear(t *testing.T) {
	f := mpfr.NewFloat().SetFloat64(3.14159, mpfr.RoundToNearest)
	f.Clear()
	// After Clear, f should no longer be used. A minimal test is to check we can call f.Clear again.
	// This shouldn't crash but does nothing.
	f.Clear()
	// There's not much else to test here without forcing memory checks.
}
