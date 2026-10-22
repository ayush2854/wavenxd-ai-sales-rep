// In development the UI runs on :5173 and the backend on :8000.
// When the backend serves the built UI, both share one origin, so the base is empty.
const API: string =
  (import.meta.env.VITE_API_URL as string | undefined) ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

const TIMEOUT_MS = 120_000; // the backend may retry Gemini a few times

export function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export async function sendMessage(sessionId: string, message: string): Promise<string> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = (await res.json()) as { reply: string };
    return data.reply;
  } finally {
    clearTimeout(timer);
  }
}

export async function resetSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${API}/reset?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" });
  } catch {
    /* a failed reset is harmless: a new session id is used anyway */
  }
}