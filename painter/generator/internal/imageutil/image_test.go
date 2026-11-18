package imageutil

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// A half-transparent pure-red pixel must be scored as red, not dark red.
func TestLoadAndPrepareUnpremultipliesAlpha(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	img.SetNRGBA(0, 0, color.NRGBA{255, 0, 0, 128})
	img.SetNRGBA(1, 0, color.NRGBA{0, 0, 255, 255})
	path := filepath.Join(t.TempDir(), "a.png")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	f.Close()

	p, err := LoadAndPrepare(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r := p.Target[0]; math.Abs(float64(r)-1) > 0.01 {
		t.Fatalf("semi-transparent red stored as r=%.3f, want 1.0", r)
	}
	if a := p.Target[3]; math.Abs(float64(a)-128.0/255) > 0.01 {
		t.Fatalf("alpha changed: %.3f", a)
	}
	if b := p.Target[6]; b != 1 {
		t.Fatalf("opaque pixel changed: b=%.3f", b)
	}
}
