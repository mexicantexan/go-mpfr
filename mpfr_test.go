package mpfr_test

import (
	"fmt"
	"github.com/mexicantexan/go-mpfr"
	"math"
	"math/big"
	"strconv"
	"strings"
	"testing"
)

const eps = 1e-13

func almostEqual(got, want float64) bool {
	return math.Abs(got-want) < eps
}

// NormalizeFloatString converts a float string to a consistent format for comparison
func NormalizeFloatString(input string) (string, error) {
	// Parse the string into a float64
	num, err := strconv.ParseFloat(strings.TrimSpace(input), 64)
	if err != nil {
		return "", err
	}

	// Reformat the float64 to a standard string representation
	// %e for scientific notation, remove trailing zeros and the decimal point if unnecessary
	return fmt.Sprintf("%.10g", num), nil
}

// CompareFloatsString checks if two floating-point strings are equivalent
func CompareFloatStrings(actual, expected string) (bool, error) {
	normalizedActual, err := NormalizeFloatString(actual)
	if err != nil {
		return false, fmt.Errorf("error normalizing actual value: %v", err)
	}

	normalizedExpected, err := NormalizeFloatString(expected)
	if err != nil {
		return false, fmt.Errorf("error normalizing expected value: %v", err)
	}

	return normalizedActual == normalizedExpected, nil
}

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
	f.SetFloat64(3.1415)
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
	f.SetFloat64(1.2345)

	s := f.String()
	if s == "" {
		t.Errorf("String() returned empty; want a numeric string representation")
	}
}

func TestAdd(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(1.5)
	y := mpfr.NewFloat().SetFloat64(2.25)
	sum := mpfr.NewFloat().Add(x, y, mpfr.RoundToNearest)

	got := sum.GetFloat64(mpfr.RoundToNearest)
	want := 3.75
	if got != want {
		t.Errorf("1.5 + 2.25 = %v; want %v", got, want)
	}
}

func TestSub(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(5.0)
	y := mpfr.NewFloat().SetFloat64(3.1)
	diff := mpfr.NewFloat().Sub(x, y, mpfr.RoundToNearest)

	got := diff.GetFloat64(mpfr.RoundToNearest)
	want := 1.9
	// Floating-point inexactness is possible, so you might do an approximate check
	if got != want {
		t.Errorf("5.0 - 3.1 = %v; want %v", got, want)
	}
}

func TestMul(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(2.0)
	y := mpfr.NewFloat().SetFloat64(3.25)
	product := mpfr.NewFloat().Mul(x, y, mpfr.RoundToNearest)

	got := product.GetFloat64(mpfr.RoundToNearest)
	want := 6.5
	if got != want {
		t.Errorf("2.0 * 3.25 = %v; want %v", got, want)
	}
}

func TestDiv(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(10.0)
	y := mpfr.NewFloat().SetFloat64(4.0)
	quotient := mpfr.NewFloat().Div(x, y, mpfr.RoundToNearest)

	got := quotient.GetFloat64(mpfr.RoundToNearest)
	want := 2.5
	if got != want {
		t.Errorf("10.0 / 4.0 = %v; want %v", got, want)
	}
}

func TestQuo(t *testing.T) {
	tests := []struct {
		x, y        float64
		rnd         mpfr.Rnd
		expected    string
		shouldPanic bool
	}{
		{10, 2, mpfr.RoundToNearest, "5", false},
		{10, 3, mpfr.RoundToNearest, "3.333333333333333", false},
		{10, 3, mpfr.RoundToward0, "3.333333333333333", false},
		{10, 0, mpfr.RoundToNearest, "", true},
		{-10, 0, mpfr.RoundToNearest, "", true},
	}

	for _, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if !tt.shouldPanic {
						t.Errorf("Quo(%v, %v) unexpectedly panicked: %v", tt.x, tt.y, r)
					}
				} else if tt.shouldPanic {
					t.Errorf("Quo(%v, %v) did not panic as expected", tt.x, tt.y)
				}
			}()

			x := mpfr.FromFloat64(tt.x)
			y := mpfr.FromFloat64(tt.y)
			result := mpfr.NewFloat()

			if !tt.shouldPanic {
				result.Quo(x, y, tt.rnd)
				got := result.Float64()
				expected, _ := strconv.ParseFloat(tt.expected, 64)
				closeEnough := almostEqual(got, expected)
				if !closeEnough {
					t.Errorf("Quo(%v, %v) got %v; want %v", tt.x, tt.y, got, tt.expected)
				}
			} else {
				_ = result.Quo(x, y, tt.rnd) // Expect a panic
			}
		}()
	}
}

func TestCmp(t *testing.T) {
	a := mpfr.NewFloat().SetFloat64(2.0)
	b := mpfr.NewFloat().SetFloat64(2.0)
	c := mpfr.NewFloat().SetFloat64(3.0)

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
	orig := mpfr.NewFloat().SetFloat64(1.23)
	copy := mpfr.NewFloat().Copy(orig)

	if copy.Cmp(orig) != 0 {
		t.Errorf("Copy() doesn't match original: orig=%v copy=%v", orig, copy)
	}

	// Modify the original and ensure copy is unaffected.
	orig.SetFloat64(9.99)
	if copy.Cmp(orig) == 0 {
		t.Errorf("Changing the original also changed the copy; want them independent.")
	}
}

func TestClear(t *testing.T) {
	f := mpfr.NewFloat().SetFloat64(3.14159)
	f.Clear()
	// After Clear, f should no longer be used. A minimal test is to check we can call f.Clear again.
	// This shouldn't crash but does nothing.
	f.Clear()
	// There's not much else to test here without forcing memory checks.
}

func TestPow(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(2.0)
	y := mpfr.NewFloat().SetFloat64(3.0)

	result := mpfr.NewFloat()
	result.Pow(x, y, mpfr.RoundToNearest)

	want := 8.0 // 2^3 = 8
	got := result.GetFloat64(mpfr.RoundToNearest)
	if got != want {
		t.Errorf("Pow(2, 3) = %v; want %v", got, want)
	}
}

func TestExp(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(1.0) // e^1 = e

	result := mpfr.NewFloat()
	result.Exp(x, mpfr.RoundToNearest)

	want := 2.718281828459045 // Approximation of e
	got := result.GetFloat64(mpfr.RoundToNearest)
	if math.Abs(got-want) > 1e-14 {
		t.Errorf("Exp(1) = %v; want %v", got, want)
	}
}

func TestLog(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(2.718281828459045) // ln(e) = 1

	result := mpfr.NewFloat()
	result.Log(x, mpfr.RoundToNearest)

	want := 1.0 // ln(e) = 1
	got := result.GetFloat64(mpfr.RoundToNearest)
	if math.Abs(got-want) > 1e-14 {
		t.Errorf("Log(e) = %v; want %v", got, want)
	}
}

func TestAbs(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(-3.5)
	got := mpfr.NewFloat().Abs(x, mpfr.RoundToNearest)
	want := 3.5
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Abs(-3.5) got %v; want %v", got, want)
	}
}

func TestAcos(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(1.0)
	got := mpfr.NewFloat().Acos(x, mpfr.RoundToNearest)
	want := 0.0 // acos(1) = 0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Acos(1.0) got %v; want %v", got, want)
	}
}

func TestAcosh(t *testing.T) {
	// acosh(1) = 0
	x := mpfr.NewFloat().SetFloat64(1.0)
	got := mpfr.NewFloat().Acosh(x, mpfr.RoundToNearest)
	want := 0.0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Acosh(1.0) got %v; want %v", got, want)
	}
}

func TestAgm(t *testing.T) {
	// For x=1, y=9, AGM is about 3.9362355...(approx)
	a := mpfr.NewFloat().SetFloat64(1.0)
	b := mpfr.NewFloat().SetFloat64(9.0)
	got := mpfr.NewFloat().Agm(a, b, mpfr.RoundToNearest)

	want := 3.9362355

	gotF := got.GetFloat64(mpfr.RoundToNearest)
	if math.Abs(gotF-want) > 1e-6 {
		t.Errorf("Agm(1,9) ~ %v; want ~2.986415", gotF)
	}
}

func TestAsin(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(0.0)
	got := mpfr.NewFloat().Asin(x, mpfr.RoundToNearest)
	want := 0.0 // asin(0) = 0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Asin(0) got %v; want %v", got, want)
	}
}

func TestAsinh(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(0.0)
	got := mpfr.NewFloat().Asinh(x, mpfr.RoundToNearest)
	want := 0.0 // asinh(0) = 0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Asinh(0) got %v; want %v", got, want)
	}
}

func TestAtan(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(0.0)
	got := mpfr.NewFloat().Atan(x, mpfr.RoundToNearest)
	want := 0.0 // atan(0) = 0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Atan(0) got %v; want %v", got, want)
	}
}

func TestAtan2(t *testing.T) {
	// atan2(y=1, x=1) = pi/4 ~ 0.785398163
	y := mpfr.NewFloat().SetFloat64(1.0)
	x := mpfr.NewFloat().SetFloat64(1.0)
	got := mpfr.NewFloat().Atan2(y, x, mpfr.RoundToNearest)
	want := math.Pi / 4
	if math.Abs(got.GetFloat64(mpfr.RoundToNearest)-want) > 1e-14 {
		t.Errorf("Atan2(1,1) got %v; want %v", got, want)
	}
}

func TestAtanh(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(0.0)
	got := mpfr.NewFloat().Atanh(x, mpfr.RoundToNearest)
	want := 0.0 // atanh(0) = 0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Atanh(0) got %v; want %v", got, want)
	}
}

func TestRoots(t *testing.T) {
	tests := []struct {
		x        float64  // Input value
		rnd      mpfr.Rnd // Rounding mode
		expected string   // Expected result as a string
		method   string   // Method to test ("Cbrt" or "Sqrt")
	}{
		{8, mpfr.RoundToNearest, "2", "Cbrt"},
		{-27, mpfr.RoundToNearest, "-3", "Cbrt"},
		{9, mpfr.RoundToNearest, "3", "Sqrt"},
		{16, mpfr.RoundToNearest, "4", "Sqrt"},
	}

	for _, tt := range tests {
		x := mpfr.FromFloat64(tt.x)
		result := mpfr.NewFloat()

		switch tt.method {
		case "Cbrt":
			result.Cbrt(x, tt.rnd)
		case "Sqrt":
			result.Sqrt(x, tt.rnd)
		default:
			t.Fatalf("unknown method: %s", tt.method)
		}

		got := result.Float64()
		expected, _ := strconv.ParseFloat(tt.expected, 64)
		closeEnough := almostEqual(got, expected)
		if !closeEnough {
			t.Errorf("%s(%v) got %v; want %v", tt.method, tt.x, got, tt.expected)
		}
	}
}

func TestRootUI(t *testing.T) {
	tests := []struct {
		x           float64  // Input value
		k           uint     // Root degree
		rnd         mpfr.Rnd // Rounding mode
		expected    string   // Expected result as a string
		shouldPanic bool     // Whether the operation should panic
	}{
		{32, 5, mpfr.RoundToNearest, "2", false}, // 5th root of 32
		{8, 3, mpfr.RoundToNearest, "2", false},  // Cube root
		{16, 4, mpfr.RoundToNearest, "2", false}, // 4th root
		{27, 3, mpfr.RoundToNearest, "3", false}, // Cube root
		{10, 0, mpfr.RoundToNearest, "", true},   // Invalid k
		{-32, 2, mpfr.RoundToNearest, "", true},  // Even root of a negative number
	}

	for _, tt := range tests {
		func() {
			defer func() {
				if r := recover(); r != nil {
					if !tt.shouldPanic {
						t.Errorf("Root(%v, %d) unexpectedly panicked: %v", tt.x, tt.k, r)
					}
				} else if tt.shouldPanic {
					t.Errorf("Root(%v, %d) did not panic as expected", tt.x, tt.k)
				}
			}()

			x := mpfr.FromFloat64(tt.x)
			result := mpfr.NewFloat()

			if !tt.shouldPanic {
				result.RootUI(x, tt.k, tt.rnd)
				got := result.Float64()
				expected, _ := strconv.ParseFloat(tt.expected, 64)
				closeEnough := almostEqual(got, expected)
				if !closeEnough {
					t.Errorf("Root(%v, %d) got %v; want %v", tt.x, tt.k, got, tt.expected)
				}
			} else {
				out := result.RootUI(x, tt.k, tt.rnd)
				println(out.Float64())
			}
		}()
	}
}

func TestCeil(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(3.1)
	got := mpfr.NewFloat().Ceil(x)
	want := 4.0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Ceil(3.1) got %v; want %v", got, want)
	}
}

func TestCmpAbs(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(-3.5)
	y := mpfr.NewFloat().SetFloat64(2.0)
	res := mpfr.CmpAbs(x, y)
	// |-3.5|=3.5, |2.0|=2 => 3.5>2 => res=1
	if res != 1 {
		t.Errorf("CmpAbs(-3.5, 2.0) = %v; want 1", res)
	}
}

func TestCos(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(0.0)
	got := mpfr.NewFloat().Cos(x, mpfr.RoundToNearest)
	want := 1.0 // cos(0)=1
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Cos(0) got %v; want %v", got, want)
	}
}

func TestCosh(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(0.0)
	got := mpfr.NewFloat().Cosh(x, mpfr.RoundToNearest)
	want := 1.0 // cosh(0)=1
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Cosh(0) got %v; want %v", got, want)
	}
}

func TestCot(t *testing.T) {
	// cot(pi/4) = 1
	val := math.Pi / 4
	x := mpfr.NewFloat().SetFloat64(val)
	got := mpfr.NewFloat().Cot(x, mpfr.RoundToNearest)
	want := 1.0
	if math.Abs(got.GetFloat64(mpfr.RoundToNearest)-want) > 1e-7 {
		t.Errorf("Cot(pi/4) got %v; want %v", got, want)
	}
}

func TestCoth(t *testing.T) {
	// coth(1) ~ 1.313035285
	x := mpfr.NewFloat().SetFloat64(1.0)
	got := mpfr.NewFloat().Coth(x, mpfr.RoundToNearest)
	want := 1.313035285
	diff := math.Abs(got.GetFloat64(mpfr.RoundToNearest) - want)
	if diff > 1e-7 {
		t.Errorf("Coth(1) got %v; want ~1.313035285", got)
	}
}

func TestCsc(t *testing.T) {
	// csc(pi/2)=1
	val := math.Pi / 2
	x := mpfr.NewFloat().SetFloat64(val)
	got := mpfr.NewFloat().Csc(x, mpfr.RoundToNearest)
	want := 1.0
	if math.Abs(got.GetFloat64(mpfr.RoundToNearest)-want) > 1e-7 {
		t.Errorf("Csc(pi/2) got %v; want 1.0", got)
	}
}

func TestCsch(t *testing.T) {
	// csch(1) ~ 0.850918128
	x := mpfr.NewFloat().SetFloat64(1.0)
	got := mpfr.NewFloat().Csch(x, mpfr.RoundToNearest)
	want := 0.850918128
	diff := math.Abs(got.GetFloat64(mpfr.RoundToNearest) - want)
	if diff > 1e-7 {
		t.Errorf("Csch(1) got %v; want ~0.850918128", got)
	}
}

func TestExp10(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(2.0)
	got := mpfr.NewFloat().Exp10(x, mpfr.RoundToNearest)
	want := 100.0 // 10^2 = 100
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Exp10(2) got %v; want %v", got, want)
	}
}

func TestExp2(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(3.0)
	got := mpfr.NewFloat().Exp2(x, mpfr.RoundToNearest)
	want := 8.0 // 2^3=8
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Exp2(3) got %v; want %v", got, want)
	}
}

func TestFloor(t *testing.T) {
	x := mpfr.NewFloat().SetFloat64(3.9)
	got := mpfr.NewFloat().Floor(x)
	want := 3.0
	if !almostEqual(got.GetFloat64(mpfr.RoundToNearest), want) {
		t.Errorf("Floor(3.9) got %v; want %v", got, want)
	}
}

func TestFitsAll(t *testing.T) {
	var smallVal float64 = 123
	var bigVal float64 = 1e20

	sf := mpfr.NewFloat().SetFloat64(smallVal)
	bf := mpfr.NewFloat().SetFloat64(bigVal)

	// smallVal => 123 fits in int, long, short (assuming typical 32/64-bit)
	if !sf.FitsSint(mpfr.RoundToNearest) {
		t.Error("FitsSint(123) = false; want true")
	}
	if !sf.FitsSlong(mpfr.RoundToNearest) {
		t.Error("FitsSlong(123) = false; want true")
	}
	if !sf.FitsUshort(mpfr.RoundToNearest) {
		t.Error("FitsUshort(123) = false; want true (assuming 16-bit short, 123 fits)")
	}

	// bigVal => 1e20 typically won't fit in a 32-bit or 16-bit range
	if bf.FitsSint(mpfr.RoundToNearest) {
		t.Error("FitsSint(1e20) = true; want false")
	}
	if bf.FitsSshort(mpfr.RoundToNearest) {
		t.Error("FitsSshort(1e20) = true; want false")
	}

	// Ulong or intmax could still fail if 1e20 is beyond 64-bit range,
	// doing a naive check for now
	if bf.FitsUint(mpfr.RoundToNearest) {
		t.Error("FitsUint(1e20) = true; want false on a 32-bit system")
	}
}

func TestSetPrec(t *testing.T) {
	f := mpfr.NewFloat()
	f.SetFloat64(3.141592653589793)

	// Set precision to 128 bits
	f.SetPrec(128)
	got := f.GetFloat64(mpfr.RoundToNearest)
	want := 0.0 // After setting precision, value should be cleared

	if got != want {
		t.Errorf("SetPrec(128) got %v; want %v", got, want)
	}
}

func TestFromInt(t *testing.T) {
	f := mpfr.FromInt(-42)
	if got := f.GetFloat64(mpfr.RoundToNearest); got != -42.0 {
		t.Errorf("FromInt(-42) got %v; want -42", got)
	}
}

func TestFromInt64(t *testing.T) {
	f := mpfr.FromInt64(int64(9223372036854775807))
	if got := f.GetFloat64(mpfr.RoundToNearest); got != 9.223372036854776e+18 {
		t.Errorf("FromInt64(9223372036854775807) got %v; want 9.223372036854776e+18", got)
	}
}

func TestFromUint64(t *testing.T) {
	f := mpfr.FromUint64(uint64(18446744073709551615))
	if got := f.GetFloat64(mpfr.RoundToNearest); got != 1.8446744073709552e+19 {
		t.Errorf("FromUint64(18446744073709551615) got %v; want 1.8446744073709552e+19", got)
	}
}

func TestFromFloat64(t *testing.T) {
	f := mpfr.FromFloat64(math.Pi)
	if got := f.GetFloat64(mpfr.RoundToNearest); math.Abs(got-math.Pi) > 1e-15 {
		t.Errorf("FromFloat64(math.Pi) got %v; want %v", got, math.Pi)
	}
}

func TestFromBigInt(t *testing.T) {
	bi := big.NewInt(-1234567890123456789)
	f := mpfr.FromBigInt(bi)
	if got := f.GetFloat64(mpfr.RoundToNearest); got != -1.2345678901234568e+18 {
		t.Errorf("FromBigInt(-1234567890123456789) got %v; want -1.2345678901234568e+18", got)
	}
}

func TestFromBigFloat(t *testing.T) {
	bf := big.NewFloat(1.618033988749894)
	f := mpfr.FromBigFloat(bf)
	if got := f.GetFloat64(mpfr.RoundToNearest); math.Abs(got-1.618033988749894) > 1e-15 {
		t.Errorf("FromBigFloat(1.618033988749894) got %v; want %v", got, 1.618033988749894)
	}
}

func TestSetInt(t *testing.T) {
	f := mpfr.NewFloat()
	f.SetInt(-42)
	if got := f.GetFloat64(mpfr.RoundToNearest); got != -42.0 {
		t.Errorf("SetInt(-42) got %v; want -42", got)
	}
}

func TestSetInt64(t *testing.T) {
	f := mpfr.NewFloat()
	f.SetInt64(9223372036854775807)
	if got := f.GetFloat64(mpfr.RoundToNearest); math.Abs(got-9.223372036854776e+18) > 1e-10 {
		t.Errorf("SetInt64(9223372036854775807) got %v; want ~9.223372036854776e+18", got)
	}
}

func TestSetUint64(t *testing.T) {
	f := mpfr.NewFloat()
	f.SetUint64(18446744073709551615)
	if got := f.GetFloat64(mpfr.RoundToNearest); math.Abs(got-1.8446744073709552e+19) > 1e-10 {
		t.Errorf("SetUint64(18446744073709551615) got %v; want ~1.8446744073709552e+19", got)
	}
}

func TestSetFloat64(t *testing.T) {
	f := mpfr.NewFloat()
	f.SetFloat64(math.Pi)
	if got := f.GetFloat64(mpfr.RoundToNearest); math.Abs(got-math.Pi) > 1e-15 {
		t.Errorf("SetFloat64(math.Pi) got %v; want %v", got, math.Pi)
	}
}

func TestSetBigInt(t *testing.T) {
	bi := big.NewInt(-1234567890123456789)
	f := mpfr.NewFloat()
	f.SetBigInt(bi)
	if got := f.GetFloat64(mpfr.RoundToNearest); got != -1.2345678901234568e+18 {
		t.Errorf("SetBigInt(-1234567890123456789) got %v; want -1.2345678901234568e+18", got)
	}
}

func TestSetBigFloat(t *testing.T) {
	bf := big.NewFloat(1.618033988749894)
	f := mpfr.NewFloat()
	f.SetBigFloat(bf)
	if got := f.GetFloat64(mpfr.RoundToNearest); math.Abs(got-1.618033988749894) > 1e-15 {
		t.Errorf("SetBigFloat(1.618033988749894) got %v; want %v", got, 1.618033988749894)
	}
}

func TestInt64(t *testing.T) {
	f := mpfr.FromFloat64(123.45)
	got := f.Int64()
	want := int64(123)
	if got != want {
		t.Errorf("Int64() got %v; want %v", got, want)
	}
}

func TestUint64(t *testing.T) {
	f := mpfr.FromFloat64(123.45)
	got := f.Uint64()
	want := uint64(123)
	if got != want {
		t.Errorf("Uint64() got %v; want %v", got, want)
	}
}

func TestFloat64(t *testing.T) {
	f := mpfr.FromFloat64(math.Pi)
	got := f.Float64()
	want := math.Pi
	if math.Abs(got-want) > 1e-15 {
		t.Errorf("Float64() got %v; want %v", got, want)
	}
}

func TestBigInt(t *testing.T) {
	f := mpfr.FromFloat64(12345.67)
	result := new(big.Int)
	f.BigInt(result)
	want := big.NewInt(12345)
	if result.Cmp(want) != 0 {
		t.Errorf("BigInt() got %v; want %v", result, want)
	}

	f = mpfr.FromFloat64(0.000123456)
	result = new(big.Int)
	f.BigInt(result)
	want = big.NewInt(0) // 0 for very small fractional values
	if result.Cmp(want) != 0 {
		t.Errorf("BigInt() with small fraction got %v; want %v", result, want)
	}
}

func TestBigFloat(t *testing.T) {
	tests := []struct {
		input    float64
		expected string // Expected string representation of the big.Float
	}{
		{12345.67, "12345.67"},
		{0.000123456, "0.000123456"},
		{math.Pi, "3.1415926535897931"},
		{-math.E, "-2.718281828459045"},
	}

	bigFloatEps := big.NewFloat(1e-16)

	for _, tt := range tests {
		f := mpfr.FromFloat64(tt.input)
		result := new(big.Float)
		f.BigFloat(result)

		got := result.Text('g', -1) // Get the string representation
		diff := new(big.Float).Sub(result, big.NewFloat(tt.input))
		if diff.Cmp(bigFloatEps) > 0 {
			t.Errorf("BigFloat(%v) got %v; want %v", tt.input, got, tt.expected)
		}
	}
}
