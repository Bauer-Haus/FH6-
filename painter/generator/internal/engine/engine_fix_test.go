package engine

import (
	"math"
	"testing"

	"forza-painter-geometrize-go/internal/model"
)

func ellipse(x, y, rx, ry, rot float64, a int) model.Shape {
	return model.Shape{Type: 16, Data: []float64{x, y, rx, ry, rot}, Color: []int{10, 20, 30, a}}
}

func fullMask(w, h int) []uint8 {
	m := make([]uint8, w*h)
	for i := range m {
		m[i] = 1
	}
	return m
}

// A long ellipse turned 90° is mostly outside its unrotated cx±rx, cy±ry
// box. Covering only that box with a later shape must not mark it occluded.
func TestPruneKeepsRotatedShapeVisibleOutsideUnrotatedBox(t *testing.T) {
	w, h := 100, 100
	shapes := []model.Shape{
		{Type: 1, Data: []float64{0, 0, 100, 100}, Color: []int{0, 0, 0, 255}},
		ellipse(50, 50, 40, 4, 90, 255), // vertical needle: spans y 10..90
		ellipse(50, 50, 46, 9, 0, 255),  // covers x 4..96, y 41..59 only
	}
	got := pruneOccludedShapes(shapes, w, h, fullMask(w, h))
	if len(got) != 3 {
		t.Fatalf("visible rotated ellipse was pruned: kept %d of 3 shapes", len(got))
	}
}

func TestPruneStillRemovesFullyCoveredShape(t *testing.T) {
	w, h := 100, 100
	shapes := []model.Shape{
		{Type: 1, Data: []float64{0, 0, 100, 100}, Color: []int{0, 0, 0, 255}},
		ellipse(50, 50, 40, 4, 90, 255),
		ellipse(50, 50, 48, 48, 0, 255), // circle covering the whole needle
	}
	got := pruneOccludedShapes(shapes, w, h, fullMask(w, h))
	if len(got) != 2 {
		t.Fatalf("hidden shape not pruned: kept %d, want 2", len(got))
	}
}

func TestShapeExtentsContainRotatedCorners(t *testing.T) {
	rx, ry := float32(30), float32(10)
	for _, deg := range []float64{0, 17, 45, 90, 133, 270} {
		th := deg * math.Pi / 180
		c, s := float32(math.Cos(th)), float32(math.Sin(th))
		ex, ey := shapeExtents(false, rx, ry, c, s)
		for _, k := range [][2]float32{{rx, ry}, {rx, -ry}, {-rx, ry}, {-rx, -ry}} {
			// local (xr, yr) -> world (dx, dy) is the inverse rotation used by the kernels
			dx := k[0]*c - k[1]*s
			dy := k[0]*s + k[1]*c
			if float32(math.Abs(float64(dx))) > ex+1e-3 || float32(math.Abs(float64(dy))) > ey+1e-3 {
				t.Fatalf("rect corner %v at %v° outside box (%v, %v)", k, deg, ex, ey)
			}
		}
		eex, eey := shapeExtents(true, rx, ry, c, s)
		if eex > ex+1e-3 || eey > ey+1e-3 {
			t.Fatalf("ellipse box larger than rect box at %v°", deg)
		}
	}
}
