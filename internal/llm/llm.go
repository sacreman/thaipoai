// Package llm holds the prompt the model is trained and tested with, and a client for any
// OpenAI-compatible chat endpoint.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Message is one chat turn.
type Message = map[string]string

// ReferenceOnly introduces chat history. History in a system message is not translated.
const ReferenceOnly = "Earlier messages in this chat, for reference only. Never translate them:\n"

const chatSystem = "You translate chat messages between English and Thai. Translate the meaning, not word for word: " +
	"idioms, slang and sarcasm must keep their intended meaning in the target language, and match the sender's tone. " +
	"If something is truly ambiguous, keep the ambiguity and never invent facts. Output only the translation."

// ChatMessages must match chat_messages in train/examples.py, which the model is trained on.
// Notes carries optional facts about the speakers.
func ChatMessages(target, source string, context []string, notes string) []Message {
	system := chatSystem
	if notes != "" {
		system += "\n\n" + notes
	}
	if len(context) > 0 {
		system += "\n\n" + ReferenceOnly + strings.Join(context, "\n")
	}
	return []Message{
		{"role": "system", "content": system},
		{"role": "user", "content": "Translate into " + target + ":\n\n" + source},
	}
}

func genderWord(gender string) string {
	if gender == "male" {
		return "a man"
	}
	return "a woman"
}

// SpeakerNotes describes the people in a chat, for the notes argument of ChatMessages.
func SpeakerNotes(speaker, addressee string) string {
	var parts []string
	if speaker != "" {
		parts = append(parts, "The person writing this message is "+genderWord(speaker)+".")
	}
	if addressee != "" {
		parts = append(parts, "The person reading it is "+genderWord(addressee)+".")
	}
	return strings.Join(parts, " ")
}

// Client calls an OpenAI-compatible endpoint.
type Client struct {
	Endpoint, Model, APIKey string
	Temperature             float64
	NoThinking              bool
	HTTP                    *http.Client
}

// Chat returns the model's reply.
func (c Client) Chat(ctx context.Context, messages []Message) (string, error) {
	payload := map[string]any{
		"model":       c.Model,
		"messages":    messages,
		"temperature": c.Temperature,
		"max_tokens":  256,
	}
	if c.NoThinking {
		payload["thinking"] = map[string]string{"type": "disabled"}
		payload["reasoning_effort"] = "none"
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		request.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	response, err := c.HTTP.Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return "", err
	}
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d", response.StatusCode)
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "", err
	}
	if len(decoded.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	return strings.TrimSpace(decoded.Choices[0].Message.Content), nil
}
