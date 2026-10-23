// Thin wrappers around the browser's Web Speech API (works in Chrome and Edge).
// Voice input needs an internet connection and microphone permission; installed voices vary by device.

interface RecognitionItem {
  transcript: string;
}
interface RecognitionResult {
  isFinal: boolean;
  [index: number]: RecognitionItem;
}
interface RecognitionEvent {
  resultIndex: number;
  results: ArrayLike<RecognitionResult>;
}
interface Recognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  maxAlternatives: number;
  onresult: ((e: RecognitionEvent) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
}
type RecognitionCtor = new () => Recognition;

function recognitionCtor(): RecognitionCtor | undefined {
  const w = window as unknown as { SpeechRecognition?: RecognitionCtor; webkitSpeechRecognition?: RecognitionCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition;
}

export const speechInputSupported = (): boolean => recognitionCtor() !== undefined;
export const speechOutputSupported = (): boolean => typeof window !== "undefined" && "speechSynthesis" in window;

export const LANGUAGES = [
  { code: "en-IN", label: "English" },
  { code: "hi-IN", label: "हिन्दी" },
  { code: "mr-IN", label: "मराठी" },
  { code: "ta-IN", label: "தமிழ்" },
];

// ---------------------------------------------------------------- voice input
export interface Listener {
  stop(): void;
}

interface ListenHandlers {
  onInterim(text: string): void;
  onFinal(text: string): void;
  onEnd(): void;
  onError(message: string): void;
}

function friendlyError(code: string): string {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone permission was blocked. Allow the microphone for this page and try again.";
    case "network":
      return "Voice input needs an internet connection.";
    case "audio-capture":
      return "No microphone was found.";
    case "language-not-supported":
      return "This browser does not support voice input for the selected language.";
    default:
      return `Voice input error (${code}).`;
  }
}

export function listen(lang: string, handlers: ListenHandlers): Listener {
  const Ctor = recognitionCtor();
  if (!Ctor) {
    handlers.onError("Voice input is not supported in this browser. Please use Chrome or Edge.");
    handlers.onEnd();
    return { stop: () => undefined };
  }
  const rec = new Ctor();
  rec.lang = lang;
  rec.interimResults = true;
  rec.continuous = false;
  rec.maxAlternatives = 1;

  let finalText = "";
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const result = e.results[i];
      if (result.isFinal) finalText += result[0].transcript;
      else interim += result[0].transcript;
    }
    handlers.onInterim((finalText + interim).trim());
  };
  rec.onerror = (e) => {
    if (e.error !== "no-speech" && e.error !== "aborted") handlers.onError(friendlyError(e.error));
  };
  rec.onend = () => {
    const text = finalText.trim();
    if (text) handlers.onFinal(text);
    handlers.onEnd();
  };
  rec.start();
  return { stop: () => rec.stop() };
}

// --------------------------------------------------------------- voice output
let token = 0; // lets a newer utterance (or stopSpeaking) silence the callbacks of an older one

function replyLanguages(text: string, preferred: string): string[] {
  if (/[\u0B80-\u0BFF]/.test(text)) return ["ta-IN"];
  if (/[\u0900-\u097F]/.test(text)) return preferred === "hi-IN" ? ["hi-IN", "mr-IN"] : ["mr-IN", "hi-IN"];
  return ["en-IN", "en-GB", "en-US"];
}

function pickVoice(candidates: string[]): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  for (const lang of candidates) {
    const wanted = lang.toLowerCase();
    const exact = voices.find((v) => v.lang.replace("_", "-").toLowerCase() === wanted);
    if (exact) return exact;
  }
  for (const lang of candidates) {
    const base = lang.split("-")[0].toLowerCase();
    const loose = voices.find((v) => v.lang.toLowerCase().startsWith(base));
    if (loose) return loose;
  }
  return undefined;
}

function cleanForSpeech(text: string): string {
  const latin = !/[\u0900-\u097F\u0B80-\u0BFF]/.test(text);
  let out = text;
  if (latin) {
    out = out
      .replace(/µm/g, " micrometres")
      .replace(/\bml\/min\b/gi, " millilitres per minute")
      .replace(/\bkHz\b/gi, " kilohertz")
      .replace(/\bcP\b/g, " centipoise");
  }
  return out.replace(/[*#`]/g, "").replace(/[ \t]+/g, " ").trim();
}

// Some browsers cut long utterances off after about 15 seconds, so speak sentence by sentence.
function chunk(text: string, max = 180): string[] {
  const sentences = text.split(/(?<=[.!?।])\s+|\n+/).filter(Boolean);
  const chunks: string[] = [];
  for (const sentence of sentences) {
    const last = chunks.length - 1;
    if (last >= 0 && chunks[last].length + sentence.length < max) chunks[last] += " " + sentence;
    else chunks.push(sentence);
  }
  return chunks;
}

export function stopSpeaking(): void {
  token++;
  if (speechOutputSupported()) window.speechSynthesis.cancel();
}

export function speak(text: string, preferred: string, onEnd: () => void): { voiceFound: boolean } {
  const synth = window.speechSynthesis;
  synth.cancel();
  const mine = ++token;
  const candidates = replyLanguages(text, preferred);
  const voice = pickVoice(candidates);
  const parts = chunk(cleanForSpeech(text));
  let index = 0;

  const done = () => {
    if (mine === token) onEnd();
  };
  const next = () => {
    if (mine !== token) return;
    if (index >= parts.length) return done();
    const utterance = new SpeechSynthesisUtterance(parts[index++]);
    utterance.lang = voice?.lang ?? candidates[0];
    if (voice) utterance.voice = voice;
    utterance.onend = next;
    utterance.onerror = done;
    synth.speak(utterance);
  };
  next();
  return { voiceFound: voice !== undefined };
}