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

// The work-group kernel (row-span loop) must score exactly the same pixels
// as the simple per-candidate kernel, for thin, rotated, clipped and
// partly off-canvas shapes of every type.
func TestWorkGroupKernelMatchesReference(t *testing.T) {
	loadTestShaders()
	const w, h = 160, 120
	rng := rand.New(rand.NewSource(11))
	target := make([]float32, w*h*4)
	current := make([]float32, w*h*4)
	mask := make([]uint8, w*h)
	for p := 0; p < w*h; p++ {
		for c := 0; c < 3; c++ {
			target[p*4+c] = rng.Float32()
			current[p*4+c] = rng.Float32()
		}
		target[p*4+3], current[p*4+3] = 1, 1
		mask[p] = 1
		if p%w < 6 { // a transparent strip on the left edge
			mask[p] = 0
		}
	}
	var cands []model.Candidate
	for i := 0; i < 400; i++ {
		rx := 0.63 + rng.Float32()*60
		ry := 0.63 + rng.Float32()*60
		if i%4 == 0 {
			ry = 0.63 + rng.Float32()*2 // needles
		}
		cands = append(cands, model.Candidate{
			ShapeType: i % 3, X: -20 + rng.Float32()*(w+40), Y: -20 + rng.Float32()*(h+40),
			RX: rx, RY: ry, Theta: rng.Float32() * 360, A: 0.3 + rng.Float32()*0.7,
		})
	}
	for _, name := range []string{"opencl", "vulkan"} {
		t.Run(name, func(t *testing.T) {
			be, err := NewBackend(name, target, current, mask, w, h, len(cands), 16)
			if err != nil {
				t.Skipf("%s unavailable: %v", name, err)
			}
			defer be.Close()
			be.SetUseWorkGroupEval(false)
			ref, err := be.Evaluate(cands)
			if err != nil {
				t.Fatal(err)
			}
			be.SetUseWorkGroupEval(true)
			got, err := be.Evaluate(cands)
			if err != nil {
				t.Fatal(err)
			}
			for i := range cands {
				a, b := float64(got[i].Score), float64(ref[i].Score)
				if math.Abs(a-b) > 2e-3*math.Max(1, math.Abs(b)) {
					t.Fatalf("candidate %d %+v: work-group %.5f vs reference %.5f", i, cands[i], a, b)
				}
			}
		})
	}
}

// With a sample budget, big shapes are scored from a sampled grid. Their
// scores must stay close to the exact ones, and small shapes (area below
// the budget) must be scored exactly.
func TestSampleBudgetApproximatesExactScore(t *testing.T) {
	loadTestShaders()
	const w, h = 400, 300
	rng := rand.New(rand.NewSource(21))
	target := make([]float32, w*h*4)
	current := make([]float32, w*h*4)
	mask := make([]uint8, w*h)
	for p := 0; p < w*h; p++ {
		x, y := float32(p%w), float32(p/w)
		target[p*4+0] = 0.5 + 0.5*float32(math.Sin(float64(x)/23))
		target[p*4+1] = 0.5 + 0.5*float32(math.Cos(float64(y)/17))
		target[p*4+2] = rng.Float32() * 0.2
		target[p*4+3] = 1
		current[p*4+0], current[p*4+1], current[p*4+2], current[p*4+3] = 0.4, 0.4, 0.4, 1
		mask[p] = 1
	}
	var cands []model.Candidate
	for i := 0; i < 200; i++ {
		r := float32(4)
		if i%2 == 0 {
			r = 40 + rng.Float32()*80 // big: sampled
		}
		cands = append(cands, model.Candidate{ShapeType: 1, X: 60 + rng.Float32()*(w-120), Y: 60 + rng.Float32()*(h-120),
			RX: r, RY: r * (0.5 + rng.Float32()), Theta: rng.Float32() * 180, A: 1})
	}
	for _, name := range []string{"opencl", "vulkan"} {
		t.Run(name, func(t *testing.T) {
			be, err := NewBackend(name, target, current, mask, w, h, len(cands), 16)
			if err != nil {
				t.Skipf("%s unavailable: %v", name, err)
			}
			defer be.Close()
			be.SetUseWorkGroupEval(true) // the budget is implemented in the work-group kernel
			exact, err := be.Evaluate(cands)
			if err != nil {
				t.Fatal(err)
			}
			exactCopy := append([]EvalResult(nil), exact...)
			be.SetSampleBudget(1024)
			approx, err := be.Evaluate(cands)
			be.SetSampleBudget(0)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for i := range cands {
				a, b := float64(approx[i].Score), float64(exactCopy[i].Score)
				rel := math.Abs(a-b) / math.Max(1, math.Abs(b))
				if i%2 == 1 && rel > 1e-4 {
					t.Fatalf("small shape %d changed: %.4f vs %.4f", i, a, b)
				}
				if rel > worst {
					worst = rel
				}
			}
			if worst > 0.10 {
				t.Fatalf("sampled score off by %.1f%% (max allowed 10%%)", worst*100)
			}
			t.Logf("worst relative error with budget 1024: %.2f%%", worst*100)
		})
	}
}
