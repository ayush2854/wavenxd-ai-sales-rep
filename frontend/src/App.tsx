import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { newSessionId, resetSession, sendMessage } from "./api";
import { LANGUAGES, listen, speak, speechInputSupported, speechOutputSupported, stopSpeaking } from "./speech";
import type { Listener } from "./speech";

type Role = "user" | "assistant";

interface Message {
  id: number;
  role: Role;
  text: string;
  error?: boolean;
}

// Demo starters: one tap sends the message. They disappear after the first real message.
const STARTERS: { label: string; text: string }[] = [
  { label: "Stent coating", text: "I want to coat stents. Viscosity around 10 cP, 3 ml/min, 10 micron droplets." },
  { label: "Fuel cell (Marathi)", text: "मला इंधन सेल कोटिंगसाठी नोझल हवा आहे" },
  { label: "Fuel cell (Hindi)", text: "मुझे फ्यूल सेल कोटिंग के लिए नोज़ल चाहिए" },
  { label: "Spray dryer", text: "We need a nozzle for an industrial spray dryer, about 100 ml/min, 20 cP." },
];

const GENERIC_ERROR =
  "Sorry, I could not reach the assistant just now. Please check that the server is running and try again.";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sessionRef = useRef<string>(newSessionId());
  const nextId = useRef(1);
  const started = useRef(false); // React StrictMode runs effects twice in development
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // voice (optional: shown only when the browser supports it)
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [micLang, setMicLang] = useState("en-IN");
  const [voiceReplies, setVoiceReplies] = useState(false);
  const [voiceNote, setVoiceNote] = useState("");
  const listenerRef = useRef<Listener | null>(null);
  const voiceRepliesRef = useRef(false);
  const micLangRef = useRef("en-IN");
  const sendRef = useRef<(text: string) => Promise<void>>(async () => undefined);
  const canListen = speechInputSupported();
  const canSpeak = speechOutputSupported();

  useEffect(() => {
    voiceRepliesRef.current = voiceReplies;
    micLangRef.current = micLang;
  }, [voiceReplies, micLang]);

  const add = useCallback((role: Role, text: string, error = false) => {
    setMessages((prev) => [...prev, { id: nextId.current++, role, text, error }]);
  }, []);

  const announce = useCallback((text: string) => {
    if (!voiceRepliesRef.current) return;
    const { voiceFound } = speak(text, micLangRef.current, () => setSpeaking(false));
    setSpeaking(true);
    setVoiceNote(voiceFound ? "" : "No installed voice was found for this language, so the browser default is used.");
  }, []);

  // The assistant speaks first: a hidden "Hello" makes it greet and introduce itself.
  const greet = useCallback(async () => {
    setLoading(true);
    try {
      const reply = await sendMessage(sessionRef.current, "Hello");
      add("assistant", reply);
      announce(reply);
    } catch {
      add("assistant", GENERIC_ERROR, true);
    } finally {
      setLoading(false);
    }
  }, [add, announce]);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    void greet();
  }, [greet]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  useEffect(() => {
    if (!loading) inputRef.current?.focus();
  }, [loading]);

  const send = useCallback(
    async (raw: string) => {
      const text = raw.trim();
      if (!text || loading) return;
      setInput("");
      add("user", text);
      setLoading(true);
      try {
        const reply = await sendMessage(sessionRef.current, text);
        add("assistant", reply);
        announce(reply);
      } catch {
        add("assistant", GENERIC_ERROR, true);
      } finally {
        setLoading(false);
      }
    },
    [add, announce, loading]
  );

  useEffect(() => {
    sendRef.current = send;
  }, [send]);

  useEffect(
    () => () => {
      listenerRef.current?.stop();
      stopSpeaking();
    },
    []
  );

  const toggleMic = () => {
    if (listening) {
      listenerRef.current?.stop();
      return;
    }
    stopSpeaking();
    setSpeaking(false);
    setVoiceNote("");
    setListening(true);
    listenerRef.current = listen(micLang, {
      onInterim: (text) => setInput(text),
      onFinal: (text) => {
        setInput("");
        void sendRef.current(text);
      },
      onEnd: () => setListening(false),
      onError: (message) => setVoiceNote(message),
    });
  };

  const stopVoice = () => {
    stopSpeaking();
    setSpeaking(false);
  };

  const newConversation = useCallback(async () => {
    if (loading) return;
    listenerRef.current?.stop();
    stopVoice();
    setVoiceNote("");
    await resetSession(sessionRef.current);
    sessionRef.current = newSessionId();
    setMessages([]);
    setInput("");
    await greet();
  }, [greet, loading]);

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Do not send while an input method (Marathi, Hindi, Tamil keyboards) is composing text.
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send(input);
    }
  };

  const showStarters = messages.length <= 1 && !loading;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            W
          </span>
          <div>
            <h1>WaveNxD Technologies</h1>
            <p>AI Sales Assistant · Ultrasonic spray nozzles</p>
          </div>
        </div>
        <button className="ghost" onClick={() => void newConversation()} disabled={loading}>
          New conversation
        </button>
      </header>

      <main className="chat" aria-live="polite">
        {messages.map((m) => (
          <div key={m.id} className={`row ${m.role}`}>
            {m.role === "assistant" && (
              <span className="avatar" aria-hidden="true">
                A
              </span>
            )}
            <div className={`bubble ${m.role}${m.error ? " error" : ""}`}>{m.text}</div>
          </div>
        ))}

        {loading && (
          <div className="row assistant">
            <span className="avatar" aria-hidden="true">
              A
            </span>
            <div className="bubble assistant typing" aria-label="Aarya is typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}

        {showStarters && (
          <div className="starters">
            {STARTERS.map((s) => (
              <button key={s.label} className="chip" onClick={() => void send(s.text)}>
                {s.label}
              </button>
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      {(canListen || canSpeak) && (
        <div className="voicebar">
          {canListen && (
            <>
              <select
                value={micLang}
                onChange={(e) => setMicLang(e.target.value)}
                disabled={listening}
                aria-label="Voice language"
              >
                {LANGUAGES.map((l) => (
                  <option key={l.code} value={l.code}>
                    {l.label}
                  </option>
                ))}
              </select>
              <button className={`mic${listening ? " on" : ""}`} onClick={toggleMic} disabled={loading}>
                {listening ? "■ Stop" : "🎤 Speak"}
              </button>
            </>
          )}
          {canSpeak && (
            <label className="toggle">
              <input
                type="checkbox"
                checked={voiceReplies}
                onChange={(e) => {
                  setVoiceReplies(e.target.checked);
                  if (!e.target.checked) stopVoice();
                }}
              />
              Read replies aloud
            </label>
          )}
          {speaking && (
            <button className="linkbtn" onClick={stopVoice}>
              Stop voice
            </button>
          )}
          {voiceNote && <span className="voicenote">{voiceNote}</span>}
        </div>
      )}

      <footer className="composer">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your message in English, Marathi, Hindi, Tamil…"
          rows={1}
          maxLength={2000}
          disabled={loading}
          aria-label="Message"
        />
        <button className="send" onClick={() => void send(input)} disabled={loading || !input.trim()}>
          Send
        </button>
      </footer>
      <p className="note">Sample conversations only: please do not enter real customer data.</p>
    </div>
  );
}