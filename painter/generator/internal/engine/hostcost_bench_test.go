package engine

import (
	"math/rand"
	"testing"

	"forza-painter-geometrize-go/internal/gpu"
	"forza-painter-geometrize-go/internal/imageutil"
	"forza-painter-geometrize-go/internal/model"
)

// Host-side (CPU) work for one "super slow" layer, without the GPU:
// 350k random candidates, packing them, turning results into structs and
// picking the best. These run on one core while the GPU waits.
func benchPrepared() *imageutil.PreparedImage {
	w, h := 1900, 1900
	return &imageutil.PreparedImage{Width: w, Height: h}
}

func benchSampler(w, h int) *errorSampler {
	rng := rand.New(rand.NewSource(1))
	grid := make([]float32, 64*64)
	for i := range grid {
		grid[i] = rng.Float32()
	}
	return newErrorSampler(grid, 64, 64, w, h)
}

func BenchmarkRandomCandidates350k(b *testing.B) {
	p := benchPrepared()
	s := benchSampler(p.Width, p.Height)
	rng := rand.New(rand.NewSource(2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = randomCandidates(rng, p, 350000, false, false, s, 0.3)
	}
}

func BenchmarkPickBest350k(b *testing.B) {
	p := benchPrepared()
	s := benchSampler(p.Width, p.Height)
	rng := rand.New(rand.NewSource(2))
	cands := randomCandidates(rng, p, 350000, false, false, s, 0.3)
	flat := make([]float32, len(cands)*4)
	for i := range flat {
		flat[i] = rng.Float32() - 0.5
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// what WaitEval + submitAndPickBest do
		out := make([]gpu.EvalResult, len(cands))
		for j := range out {
			out[j] = gpu.EvalResult{Score: flat[j*4], R: flat[j*4+1], G: flat[j*4+2], B: flat[j*4+3]}
		}
		_ = pickBestIndex(out, cands, 10)
	}
}

func BenchmarkPack350k(b *testing.B) {
	p := benchPrepared()
	s := benchSampler(p.Width, p.Height)
	rng := rand.New(rand.NewSource(2))
	cands := randomCandidates(rng, p, 350000, false, false, s, 0.3)
	packed := make([]float32, len(cands)*7)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, c := range cands {
			k := j * 7
			packed[k], packed[k+1], packed[k+2], packed[k+3] = c.X, c.Y, c.RX, c.RY
			packed[k+4], packed[k+5], packed[k+6] = c.Theta, c.A, float32(c.ShapeType)
		}
	}
}

var _ = model.Candidate{}

// The parallel pick must choose exactly what a serial scan chooses.
func TestPickBestIndexMatchesSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	p := benchPrepared()
	cands := randomCandidates(rng, p, 50000, false, false, benchSampler(p.Width, p.Height), 0)
	res := make([]gpu.EvalResult, len(cands))
	for trial := 0; trial < 5; trial++ {
		for i := range res {
			res[i].Score = float32(rng.Intn(1000)) - 900 // many ties
		}
		want := 0
		wantAdj := res[0].Score + scaleCompatibilityPenalty(cands[0], trial)
		for i := 1; i < len(res); i++ {
			if a := res[i].Score + scaleCompatibilityPenalty(cands[i], trial); a < wantAdj {
				wantAdj, want = a, i
			}
		}
		if got := pickBestIndex(res, cands, trial); got != want {
			t.Fatalf("trial %d: parallel pick %d, serial %d", trial, got, want)
		}
	}
}

// Same seed must give the same candidates, whatever the core count.
func TestRandomCandidatesDeterministic(t *testing.T) {
	p := benchPrepared()
	s := benchSampler(p.Width, p.Height)
	a := randomCandidates(rand.New(rand.NewSource(4)), p, 100000, false, false, s, 0.5)
	b := randomCandidates(rand.New(rand.NewSource(4)), p, 100000, false, false, s, 0.5)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("candidate %d differs between runs", i)
		}
	}
}
