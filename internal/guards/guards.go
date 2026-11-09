// Package guards checks a translation for failures that can be detected without
// understanding it: wrong script, changed numbers, gender particles, leaked prompt text.
// Each guard is a cheap heuristic, measured against scored translations before it is trusted.
package guards

import (
	"regexp"
	"strings"
	"unicode"
)

// Request describes one translation being checked.
type Request struct {
	Source, Output string
	TargetThai     bool
	SpeakerGender  string // "male", "female" or empty when unknown
}

var (
	digits      = regexp.MustCompile(`\d{2,}`) // single digits are mostly texting shorthand: l8, 2gether
	clockWords  = regexp.MustCompile(`ทุ่ม|โมง|บ่าย|เที่ยง|ตี\d|ตี[หนึ่งสองสามสี่ห้า]`)
	thaiEvening = regexp.MustCompile(`(หนึ่ง|สอง|สาม|สี่|ห้า|[1-5])\s*ทุ่ม`)
	englishPM   = regexp.MustCompile(`(?i)\b(\d{1,2})(?::\d{2})?\s*p\.?m\b`)
	commentary  = regexp.MustCompile(`(?i)^\s*(here is|here's|translation:|the translation|sure[,!]|note:|คำแปล|แปลว่า)`)
	refusal     = regexp.MustCompile(`(?i)as an ai\b|(cannot|can't|unable to) translate`)
	markup      = regexp.MustCompile("(?i)</?think|<\\|im_|<turn\\||<\\|turn|#src_text|#guidelines|```|earlier messages in this chat")
	hedge       = regexp.MustCompile(`ครับ\s*/\s*ค[่ะ]+|ค[่ะ]+\s*/\s*ครับ|ผม\s*/\s*(ดิฉัน|ฉัน)|(ดิฉัน|ฉัน)\s*/\s*ผม`)
	englishNot  = regexp.MustCompile(`(?i)\b(not|never|no|without|cannot)\b|n't\b`)
	thaiNot     = regexp.MustCompile(`ไม่|อย่า|ห้าม|มิได้|ไร้|ปราศจาก|เปล่า`)
	femaleWords = regexp.MustCompile(`ค่ะ|ดิฉัน`)
	maleWords   = regexp.MustCompile(`ครับ|ผม(?:\s|$|จะ|ไป|มา|เอง|ก็|ไม่|อยู่|ชอบ|รัก)`)
)

func hasThai(s string) bool {
	for _, r := range s {
		if r >= 0x0E00 && r <= 0x0E7F {
			return true
		}
	}
	return false
}

func hasLetters(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

func normaliseDigits(s string) string {
	mapped := strings.Map(func(r rune) rune {
		if r >= '๐' && r <= '๙' {
			return '0' + (r - '๐')
		}
		return r
	}, s)
	return strings.ReplaceAll(mapped, ",", "")
}

func numbersChanged(source, output string) bool {
	if clockWords.MatchString(source) || clockWords.MatchString(output) {
		return false // Thai clock hours are legitimately different numbers: 3 ทุ่ม is 9 pm
	}
	in := digits.FindAllString(normaliseDigits(source), -1)
	if len(in) == 0 {
		return false
	}
	out := map[string]bool{}
	for _, n := range digits.FindAllString(normaliseDigits(output), -1) {
		out[n] = true
	}
	for _, n := range in {
		if !out[n] {
			return true
		}
	}
	return false
}

var thaiHour = map[string]string{"หนึ่ง": "7", "1": "7", "สอง": "8", "2": "8", "สาม": "9", "3": "9", "สี่": "10", "4": "10", "ห้า": "11", "5": "11"}

// eveningChanged compares Thai evening hours with English PM hours: สองทุ่ม is 8 pm.
// ponytail: evening hours only, the one clock where the words and the digits disagree most.
func eveningChanged(source, output string) bool {
	thai, english := normaliseDigits(source), normaliseDigits(output)
	if !thaiEvening.MatchString(thai) {
		thai, english = english, thai
	}
	hours := thaiEvening.FindAllStringSubmatch(thai, -1)
	pms := englishPM.FindAllStringSubmatch(english, -1)
	if len(hours) != 1 || len(pms) != 1 {
		return false // nothing to compare, or more than one time to pair up
	}
	return thaiHour[hours[0][1]] != pms[0][1]
}

func negationLost(source, output string, targetThai bool) bool {
	if targetThai {
		return englishNot.MatchString(source) && !thaiNot.MatchString(output)
	}
	return thaiNot.MatchString(source) && !englishNot.MatchString(output)
}

// soft guards are reported but never act, because they wrongly flag too many good translations.
var soft = map[string]bool{"negation": true}

// Hard keeps the failures that should cause a repair or count against an adapter.
func Hard(failed []string) []string {
	var out []string
	for _, name := range failed {
		if !soft[name] {
			out = append(out, name)
		}
	}
	return out
}

// Check returns the name of every guard the translation fails.
func Check(r Request) []string {
	var failed []string
	add := func(name string, bad bool) {
		if bad {
			failed = append(failed, name)
		}
	}
	output := strings.TrimSpace(r.Output)
	if output == "" {
		return []string{"empty"}
	}
	add("unchanged", hasLetters(r.Source) && strings.EqualFold(output, strings.TrimSpace(r.Source)))
	if hasLetters(r.Source) {
		add("no_thai", r.TargetThai && !hasThai(output))
	}
	add("thai_in_english", !r.TargetThai && hasThai(output))
	add("runaway", len([]rune(output)) > 6*len([]rune(r.Source))+60)
	add("commentary", commentary.MatchString(output))
	add("refusal", refusal.MatchString(output))
	add("markup", markup.MatchString(output))
	add("numbers", numbersChanged(r.Source, output))
	add("times", eveningChanged(r.Source, output))
	add("negation", negationLost(r.Source, output, r.TargetThai))
	if r.TargetThai {
		add("gender_hedge", hedge.MatchString(output))
		add("gender", (r.SpeakerGender == "male" && femaleWords.MatchString(output)) ||
			(r.SpeakerGender == "female" && maleWords.MatchString(output)))
	}
	return failed
}
