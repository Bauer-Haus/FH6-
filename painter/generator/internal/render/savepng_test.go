package render

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func TestSavePNGRoundTripAndNoTempLeft(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "preview.png")
	pix := []float32{1, 0, 0, 1, 0, 0.5, 0, 1, 0, 0, 1, 1, 0.25, 0.25, 0.25, 0.5}
	for pass := 0; pass < 2; pass++ { // second pass replaces an existing file
		if err := SavePNG(path, pix, 2, 2); err != nil {
			t.Fatal(err)
		}
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	r, g, b, a := img.At(1, 0).RGBA()
	if r>>8 != 0 || g>>8 != 128 || b>>8 != 0 || a>>8 != 255 {
		t.Fatalf("pixel (1,0) = %d %d %d %d", r>>8, g>>8, b>>8, a>>8)
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("expected only the PNG in %s, found %d entries", dir, len(entries))
	}
}
