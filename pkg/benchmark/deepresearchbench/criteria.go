package deepresearchbench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// criterion is one weighted rubric line item within a RACE dimension.
type criterion struct {
	Criterion   string  `json:"criterion"`
	Explanation string  `json:"explanation"`
	Weight      float64 `json:"weight"`
}

// dimensionCriteria holds the four RACE dimensions' criterion lists for one
// task.
type dimensionCriteria struct {
	Comprehensiveness    []criterion `json:"comprehensiveness"`
	Insight              []criterion `json:"insight"`
	InstructionFollowing []criterion `json:"instruction_following"`
	Readability          []criterion `json:"readability"`
}

// dimensionWeights holds the four RACE dimensions' top-level weights, which
// upstream generates to sum to 1.0.
type dimensionWeights struct {
	Comprehensiveness    float64 `json:"comprehensiveness"`
	Insight              float64 `json:"insight"`
	InstructionFollowing float64 `json:"instruction_following"`
	Readability          float64 `json:"readability"`
}

// criteriaRow is one row of the vendored criteria.jsonl.
type criteriaRow struct {
	ID              int               `json:"id"`
	Prompt          string            `json:"prompt"`
	DimensionWeight dimensionWeights  `json:"dimension_weight"`
	Criterions      dimensionCriteria `json:"criterions"`
}

// taskCriteria is the fully-loaded per-task RACE rubric, keyed by numeric
// task ID.
type taskCriteria struct {
	Weights    dimensionWeights
	Criterions dimensionCriteria
}

// loadCriteria parses criteria.jsonl into a map keyed by numeric task ID.
// This is vendored, upstream-generated data ARIES does not control, so a
// dimension-weight sum that drifts slightly from 1.0 is tolerated rather
// than treated as a hard error; only structurally missing/duplicate/empty
// rows fail loudly, mirroring loadPrompts/loadReferenceArticles.
func loadCriteria(path string) (map[int]taskCriteria, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open criteria file: %w", err)
	}
	defer file.Close()

	criteria := make(map[int]taskCriteria, expectedTaskCount)
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var row criteriaRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("parse line %d: %w", lineNumber, err)
		}
		if len(row.Criterions.Comprehensiveness) == 0 || len(row.Criterions.Insight) == 0 ||
			len(row.Criterions.InstructionFollowing) == 0 || len(row.Criterions.Readability) == 0 {
			return nil, fmt.Errorf("line %d: id %d is missing criteria for at least one dimension", lineNumber, row.ID)
		}
		if _, duplicate := criteria[row.ID]; duplicate {
			return nil, fmt.Errorf("line %d: duplicate id %d", lineNumber, row.ID)
		}
		criteria[row.ID] = taskCriteria{Weights: row.DimensionWeight, Criterions: row.Criterions}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read criteria file: %w", err)
	}

	if len(criteria) != expectedTaskCount {
		return nil, fmt.Errorf("expected exactly %d criteria rows, found %d", expectedTaskCount, len(criteria))
	}
	for id := 1; id <= expectedTaskCount; id++ {
		if _, ok := criteria[id]; !ok {
			return nil, fmt.Errorf("missing criteria for task id %d", id)
		}
	}
	return criteria, nil
}
