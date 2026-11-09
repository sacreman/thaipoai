package guards

import (
	"reflect"
	"testing"
)

func TestGuards(t *testing.T) {
	cases := []struct {
		name string
		req  Request
		want []string
	}{
		{"clean to Thai", Request{Source: "See you at 9.", Output: "เจอกันตอน 9", TargetThai: true}, nil},
		{"clean to English", Request{Source: "เจอกัน", Output: "See you."}, nil},
		{"empty", Request{Source: "hi", Output: "  ", TargetThai: true}, []string{"empty"}},
		{"not translated", Request{Source: "Hello", Output: "hello", TargetThai: true}, []string{"unchanged", "no_thai"}},
		{"Thai left in English", Request{Source: "สวัสดี", Output: "Hello สวัสดี"}, []string{"thai_in_english"}},
		{"emoji only is fine", Request{Source: "😀", Output: "😀", TargetThai: true}, nil},
		{"commentary", Request{Source: "เจอกัน", Output: "Here is the translation: see you"}, []string{"commentary"}},
		{"refusal", Request{Source: "เจอกัน", Output: "I cannot translate this."}, []string{"refusal"}},
		{"sorry is a real translation", Request{Source: "ขอโทษ", Output: "I'm sorry."}, nil},
		{"markup leak", Request{Source: "เจอกัน", Output: "</think> See you"}, []string{"markup"}},
		{"changed number", Request{Source: "It costs 450 baht.", Output: "ราคา 540 บาท", TargetThai: true}, []string{"numbers"}},
		{"texting shorthand is not a number", Request{Source: "gna be l8 2 the party, get a new 1", Output: "จะไปสายนะ", TargetThai: true}, nil},
		{"commas and Thai digits", Request{Source: "1,200 baht", Output: "๑๒๐๐ บาท", TargetThai: true}, nil},
		{"Thai clock is not a changed number", Request{Source: "เจอกัน 3 ทุ่ม", Output: "See you at 9 pm."}, nil},
		{"wrong evening hour to Thai", Request{Source: "Let's meet at 7 PM.", Output: "เจอกันตอนสองทุ่มนะ", TargetThai: true}, []string{"times"}},
		{"right evening hour to Thai", Request{Source: "Let's meet at 8 PM.", Output: "เจอกันตอนสองทุ่มนะ", TargetThai: true}, nil},
		{"wrong evening hour to English", Request{Source: "รถออกตอนสองทุ่มครึ่ง", Output: "It leaves at 10:30 PM."}, []string{"times"}},
		{"right evening hour to English", Request{Source: "รถออกตอนสองทุ่มครึ่ง", Output: "It leaves at 8:30 p.m."}, nil},
		{"negation lost to Thai", Request{Source: "I am not going.", Output: "ฉันจะไป", TargetThai: true}, []string{"negation"}},
		{"negation kept to Thai", Request{Source: "I am not going.", Output: "ฉันไม่ไป", TargetThai: true}, nil},
		{"negation lost to English", Request{Source: "ไม่ไป", Output: "I'm going."}, []string{"negation"}},
		{"hedged gender", Request{Source: "Thanks", Output: "ขอบคุณครับ/ค่ะ", TargetThai: true}, []string{"gender_hedge"}},
		{"man writing a woman's particle", Request{Source: "Thanks", Output: "ขอบคุณค่ะ", TargetThai: true, SpeakerGender: "male"}, []string{"gender"}},
		{"woman writing a man's particle", Request{Source: "Thanks", Output: "ขอบคุณครับ", TargetThai: true, SpeakerGender: "female"}, []string{"gender"}},
		{"unknown speaker has no gender check", Request{Source: "Thanks", Output: "ขอบคุณครับ", TargetThai: true}, nil},
		{"runaway", Request{Source: "Hi", Output: string(make([]rune, 100)) + "ก", TargetThai: true}, []string{"runaway"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Check(c.req); !reflect.DeepEqual(got, c.want) {
				t.Fatalf("got %v, want %v", got, c.want)
			}
		})
	}
}
