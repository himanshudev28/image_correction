// API client — talks ONLY to our own backend via the relative /api prefix
// (Vite proxies it to FastAPI in dev; same-origin in prod). No third-party calls.

export interface Page {
  page_id: string;
  order: number;
  mode: "color" | "gray" | "bw";
  rotation: number;
  confidence: number;
  gate_flags: string[];
  quad: number[][] | null;
  passthrough: boolean;
  low_confidence: boolean;
  preview_url: string;
}

export interface Scan {
  scan_id: string;
  status: string;
  pages: Page[];
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || `request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function createScan(files: File[]): Promise<Scan> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return json<Scan>(await fetch("/api/scans", { method: "POST", body: form }));
}

export async function getScan(id: string): Promise<Scan> {
  return json<Scan>(await fetch(`/api/scans/${id}`));
}

export async function recrop(id: string, pid: string, corners: number[][]): Promise<Page> {
  return json<Page>(
    await fetch(`/api/scans/${id}/pages/${pid}/recrop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corners }),
    })
  );
}

export async function rotate(id: string, pid: string, degrees: number): Promise<Page> {
  return json<Page>(
    await fetch(`/api/scans/${id}/pages/${pid}/rotate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ degrees }),
    })
  );
}

export async function setMode(id: string, pid: string, mode: string): Promise<Page> {
  return json<Page>(
    await fetch(`/api/scans/${id}/pages/${pid}/mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    })
  );
}

export async function deletePage(id: string, pid: string): Promise<Scan> {
  return json<Scan>(await fetch(`/api/scans/${id}/pages/${pid}`, { method: "DELETE" }));
}

export async function exportPdf(
  id: string,
  encoding: "jpeg" | "png",
  dpi: number
): Promise<{ pdf_url: string }> {
  return json<{ pdf_url: string }>(
    await fetch(`/api/scans/${id}/export`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ encoding, dpi }),
    })
  );
}

// cache-busting suffix so a re-processed preview reloads
export function previewSrc(p: Page): string {
  return `${p.preview_url}?v=${p.rotation}-${p.mode}-${p.confidence}`;
}
