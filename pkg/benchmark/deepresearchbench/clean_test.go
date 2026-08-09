package deepresearchbench

import "testing"

func TestStripCitationMarkers(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "single bracketed numeric citation",
			input: "society into 7 levels[15]",
			want:  "society into 7 levels",
		},
		{
			name:  "run of adjacent bracketed citations",
			input: "a well-known result[1][2][3].",
			want:  "a well-known result.",
		},
		{
			name:  "comma-separated bracketed citation list",
			input: "widely cited[1, 2].",
			want:  "widely cited.",
		},
		{
			name:  "OpenAI-style tool citation with trailing locator",
			input: "levels[15†L10][5†summary]",
			want:  "levels",
		},
		{
			name:  "markdown link citation is left untouched",
			input: "according to [ChinaFile](https://example.com/report)'s classification",
			want:  "according to [ChinaFile](https://example.com/report)'s classification",
		},
		{
			name:  "bare trailing number is left untouched",
			input: "divided into 7 levels 15",
			want:  "divided into 7 levels 15",
		},
		{
			name:  "ordinal-looking numbers are left untouched",
			input: "the 1st and 2nd quarters of 2024",
			want:  "the 1st and 2nd quarters of 2024",
		},
		{
			name:  "markdown numbered list marker is left untouched",
			input: "1. First item\n2. Second item",
			want:  "1. First item\n2. Second item",
		},
		{
			name:  "no citations at all",
			input: "a plain sentence with no markers",
			want:  "a plain sentence with no markers",
		},
		{
			name:  "citation at end of paragraph keeps the block break",
			input: "a claim[1]\n\n## Next section",
			want:  "a claim\n\n## Next section",
		},
		{
			name:  "empty string",
			input: "",
			want:  "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := stripCitationMarkers(test.input); got != test.want {
				t.Fatalf("stripCitationMarkers(%q) = %q, want %q", test.input, got, test.want)
			}
		})
	}
}
