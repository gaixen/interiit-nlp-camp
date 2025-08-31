//  running instructions: go run duckduckgo.go Narendra Modi := search_Results for Narendra Modi

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

var ddgURL = "https://api.duckduckgo.com/"

type NetFail struct {
	Err error
}

func (e NetFail) Error() string {
	return fmt.Sprintf("netfail: %v", e.Err)
}

func httpGetJSON(endpoint string, params map[string]string, retries int, timeout time.Duration) (map[string]interface{}, error) {
	var last error
	client := &http.Client{Timeout: timeout}
	values := url.Values{}
	for k, v := range params {
		values.Set(k, v)
	}
	fullURL := endpoint + "?" + values.Encode()
	for i := 0; i < retries; i++ {
		resp, err := client.Get(fullURL)
		if err != nil {
			last = err
			log.Printf("retry %d/%d: %v", i+1, retries, err)
			time.Sleep(time.Duration(1<<i)*time.Second + time.Millisecond*500)
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 500 {
			last = fmt.Errorf("bad upstream %d", resp.StatusCode)
			log.Printf("retry %d/%d: %v", i+1, retries, last)
			time.Sleep(time.Duration(1<<i)*time.Second + time.Millisecond*500)
			continue
		}
		var result map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			last = err
			log.Printf("retry %d/%d: %v", i+1, retries, err)
			time.Sleep(time.Duration(1<<i)*time.Second + time.Millisecond*500)
			continue
		}
		return result, nil
	}
	return nil, NetFail{last}
}

func ddgParams(q string) map[string]string {
	return map[string]string{
		"q":             q,
		"format":        "json",
		"no_html":       "1",
		"skip_disambig": "1",
		"t":             "agent",
	}
}

func ddg(q string) (map[string]interface{}, error) {
	return httpGetJSON(ddgURL, ddgParams(q), 3, 8*time.Second)
}

func flatten(rt []interface{}) []map[string]string {
	out := []map[string]string{}
	for _, item := range rt {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if topics, ok := m["Topics"].([]interface{}); ok {
			for _, t := range topics {
				topic, ok := t.(map[string]interface{})
				if ok && topic["FirstURL"] != nil && topic["Text"] != nil {
					out = append(out, map[string]string{
						"title": topic["Text"].(string),
						"url":   topic["FirstURL"].(string),
					})
				}
			}
		} else {
			if m["FirstURL"] != nil && m["Text"] != nil {
				out = append(out, map[string]string{
					"title": m["Text"].(string),
					"url":   m["FirstURL"].(string),
				})
			}
		}
	}
	seen := map[string]bool{}
	dedup := []map[string]string{}
	for _, x := range out {
		if !seen[x["url"]] {
			seen[x["url"]] = true
			dedup = append(dedup, x)
		}
	}
	return dedup
}

func pick(d map[string]interface{}) map[string]string {
	if v, ok := d["Answer"].(string); ok && v != "" {
		return map[string]string{"kind": "answer", "value": v}
	}
	if v, ok := d["Definition"].(string); ok && v != "" {
		return map[string]string{"kind": "definition", "value": v}
	}
	if v, ok := d["AbstractText"].(string); ok && v != "" {
		return map[string]string{"kind": "abstract", "value": v}
	}
	if h, ok := d["Heading"].(string); ok && h != "" {
		if a, ok := d["Abstract"].(string); ok && a != "" {
			return map[string]string{"kind": "abstract", "value": a}
		}
	}
	return nil
}

func search(q string, want int) map[string]interface{} {
	log.Printf("searching: %s", q)
	raw, err := ddg(q)
	if err != nil {
		log.Printf("ddg err: %v", err)
		raw = map[string]interface{}{}
	}
	var related []map[string]string
	if rt, ok := raw["RelatedTopics"].([]interface{}); ok {
		related = flatten(rt)
	}
	ans := pick(raw)
	if ans != nil {
		return map[string]interface{}{
			"query":   q,
			"mode":    "instant",
			"answer":  ans,
			"related": related[:min(want, len(related))],
		}
	}
	if len(related) > 0 {
		return map[string]interface{}{
			"query":   q,
			"mode":    "related_only",
			"related": related[:min(want, len(related))],
			"hint":    "try broader search",
		}
	}
	return map[string]interface{}{
		"query":   q,
		"mode":    "dry",
		"related": related[:min(want, len(related))],
		"hint":    "nothing solid",
	}
}

func asText(r map[string]interface{}) string {
	lines := []string{}
	if ans, ok := r["answer"].(map[string]string); ok {
		lines = append(lines, fmt.Sprintf("%s: %s", ans["kind"], ans["value"]))
	}
	if related, ok := r["related"].([]map[string]string); ok && len(related) > 0 {
		lines = append(lines, "related:")
		for _, x := range related {
			lines = append(lines, fmt.Sprintf("- %s -> %s", x["title"], x["url"]))
		}
	}
	if hint, ok := r["hint"].(string); ok && hint != "" {
		lines = append(lines, hint)
	}
	return strings.Join(lines, "\n")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("usage: duckduckgo <query> [--json] [--n N]")
		os.Exit(1)
	}
	args := os.Args[1:]
	asJSON := false
	want := 6
	queryParts := []string{}
	for i := 0; i < len(args); i++ {
		if args[i] == "--json" {
			asJSON = true
		} else if args[i] == "--n" && i+1 < len(args) {
			fmt.Sscanf(args[i+1], "%d", &want)
			i++
		} else {
			queryParts = append(queryParts, args[i])
		}
	}
	q := strings.Join(queryParts, " ")
	r := search(q, want)
	if asJSON {
		b, _ := json.MarshalIndent(r, "", "  ")
		fmt.Println(string(b))
	} else {
		fmt.Println(asText(r))
	}
}
