package output

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"forza-painter-geometrize-go/internal/model"
)

func SaveGeometry(path string, shapes []model.Shape) error {
	payload := model.Geometry{Shapes: shapes}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return WriteFileAtomic(path, data)
}

// WriteFileAtomic writes data next to path and renames it into place, so the
// desktop app polling the output folder never reads a half-written file. If
// the rename fails (on Windows the target can be briefly held open by a
// reader) it falls back to writing path directly.
func WriteFileAtomic(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*.tmp")
	if err != nil {
		return os.WriteFile(path, data, 0o644)
	}
	tmpName := tmp.Name()
	_, werr := tmp.Write(data)
	cerr := tmp.Close()
	if werr == nil && cerr == nil {
		if err := os.Rename(tmpName, path); err == nil {
			return nil
		}
	}
	os.Remove(tmpName)
	return os.WriteFile(path, data, 0o644)
}

func BuildOutputPath(basePath string, shapeCount int) string {
	ext := filepath.Ext(basePath)
	name := basePath[:len(basePath)-len(ext)]
	return fmt.Sprintf("%s.%d.json", name, shapeCount)
}

func BuildFinalOutputPath(basePath string) string {
	ext := filepath.Ext(basePath)
	name := basePath[:len(basePath)-len(ext)]
	return name + ".json"
}
