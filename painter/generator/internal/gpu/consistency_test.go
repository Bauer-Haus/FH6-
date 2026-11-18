package gpu

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"forza-painter-geometrize-go/internal/model"
)

// Every scoring path must give a candidate the same score whether it is
// evaluated alone or inside a batch. A mismatch between the host's
// per-candidate packing and a kernel's stride breaks this for every
// candidate after the first. Skips when no OpenCL/Vulkan device exists.
func TestBatchScoresMatchSingleScores(t *testing.T) {
	loadTestShaders()
	const w, h = 64, 48
	rng := rand.New(rand.NewSource(3))
	target := make([]float32, w*h*4)
	current := make([]float32, w*h*4)
	mask := make([]uint8, w*h)
	for p := 0; p < w*h; p++ {
		x, y := p%w, p/w
		target[p*4+0] = float32(x) / w
		target[p*4+1] = float32(y) / h
		target[p*4+2] = rng.Float32()
		target[p*4+3] = 1
		for c := 0; c < 3; c++ {
			current[p*4+c] = 0.5
		}
		current[p*4+3] = 1
		mask[p] = 1
	}
	cands := make([]model.Candidate, 40)
	for i := range cands {
		cands[i] = model.Candidate{
			ShapeType: i % 3, // 0 rect, 1 ellipse, 2 triangle
			X:         4 + rng.Float32()*(w-8),
			Y:         4 + rng.Float32()*(h-8),
			RX:        2 + rng.Float32()*10,
			RY:        2 + rng.Float32()*10,
			Theta:     rng.Float32() * 360,
			A:         0.4 + rng.Float32()*0.6,
		}
	}

	type cfg struct {
		backend   string
		workGroup bool
		metric    string
	}
	for _, c := range []cfg{
		{"opencl", true, "mse"}, {"opencl", false, "mse"}, {"opencl", false, "ssim"}, {"opencl", true, "ssim"},
		{"vulkan", true, "mse"}, {"vulkan", false, "mse"},
	} {
		name := c.backend + "/" + c.metric
		if c.workGroup {
			name += "/workgroup"
		}
		t.Run(name, func(t *testing.T) {
			be, err := NewBackend(c.backend, target, current, mask, w, h, len(cands), 16)
			if err != nil {
				t.Skipf("%s unavailable: %v", c.backend, err)
			}
			defer be.Close()
			be.SetUseWorkGroupEval(c.workGroup)
			be.SetErrorMetric(c.metric)
			if c.metric == "ssim" {
				if err := be.SubmitSsimMap(); err != nil {
					t.Fatal(err)
				}
				if err := be.Flush(); err != nil {
					t.Fatal(err)
				}
			}
			batch, err := be.Evaluate(cands)
			if err != nil {
				t.Fatal(err)
			}
			for i, cand := range cands {
				single, err := be.Evaluate([]model.Candidate{cand})
				if err != nil {
					t.Fatal(err)
				}
				a, b := float64(batch[i].Score), float64(single[0].Score)
				if math.Abs(a-b) > 1e-3*math.Max(1, math.Abs(b)) {
					t.Fatalf("candidate %d (shape %d): batch score %.5f != single score %.5f", i, cand.ShapeType, a, b)
				}
			}
		})
	}
}

// A square rotated 45° reaches sqrt(2)·r from its centre. The draw step used
// the ellipse bounding box (r), so the corners were never painted.
func TestApplyDrawsRotatedRectangleCorners(t *testing.T) {
	loadTestShaders()
	const w, h = 64, 64
	target := make([]float32, w*h*4)
	current := make([]float32, w*h*4)
	mask := make([]uint8, w*h)
	for p := range mask {
		mask[p] = 1
		current[p*4+3] = 1
		target[p*4+3] = 1
	}
	for _, name := range []string{"opencl", "vulkan"} {
		t.Run(name, func(t *testing.T) {
			be, err := NewBackend(name, target, current, mask, w, h, 4, 16)
			if err != nil {
				t.Skipf("%s unavailable: %v", name, err)
			}
			defer be.Close()
			sq := model.Candidate{ShapeType: 0, X: 32, Y: 32, RX: 14, RY: 14, Theta: 45, R: 1, G: 1, B: 1, A: 1}
			if err := be.Apply(sq); err != nil {
				t.Fatal(err)
			}
			out := make([]float32, w*h*4)
			if err := be.ReadCurrent(out); err != nil {
				t.Fatal(err)
			}
			// Point 18px straight up from the centre: inside the rotated
			// square (corner at 14·sqrt(2) ≈ 19.8) but outside radius 14.
			p := (32-18)*w + 32
			if out[p*4] < 0.99 {
				t.Fatalf("rotated square corner not drawn (r=%.2f)", out[p*4])
			}
		})
	}
}

func loadTestShaders() {
	// The Vulkan backend loads SPIR-V from ./shaders next to the binary; a
	// test binary lives elsewhere, so load the repo's compiled shaders.
	for _, sh := range []struct {
		dst  *[]byte
		name string
	}{{&spvEvalV3, "eval_v3"}, {&spvEvalV4, "eval_v4"}, {&spvApply, "apply"}, {&spvErrorGrid, "error_grid"}} {
		if len(*sh.dst) == 0 {
			if b, err := os.ReadFile(filepath.Join("..", "..", "shaders", sh.name+".comp.spv")); err == nil {
				*sh.dst = b
			}
		}
	}
}
