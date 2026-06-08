import { useMemo, useState } from "react";
import {
  createScan,
  deletePage,
  exportPdf,
  recrop,
  rotate,
  setMode,
  type Page,
  type Scan,
} from "./api/client";
import Uploader from "./components/Uploader";
import PagePreview from "./components/PagePreview";
import CornerAdjust from "./components/CornerAdjust";

export default function App() {
  const [scan, setScan] = useState<Scan | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adjusting, setAdjusting] = useState<Page | null>(null);
  const [encoding, setEncoding] = useState<"jpeg" | "png">("jpeg");

  const lowCount = useMemo(
    () => scan?.pages.filter((p) => p.low_confidence).length ?? 0,
    [scan]
  );

  async function run<T>(fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  async function onFiles(files: File[]) {
    const result = await run(() => createScan(files));
    if (result) setScan(result);
  }

  function replacePage(updated: Page) {
    setScan((s) =>
      s ? { ...s, pages: s.pages.map((p) => (p.page_id === updated.page_id ? updated : p)) } : s
    );
  }

  async function onExport() {
    if (!scan) return;
    const r = await run(() => exportPdf(scan.scan_id, encoding, 200));
    if (r) window.open(r.pdf_url, "_blank");
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-4 py-4">
          <h1 className="text-xl font-bold text-slate-800">Consent Document Scanner</h1>
          <p className="text-sm text-slate-500">
            Upload a photo or PDF — cropping, perspective &amp; lighting are corrected
            automatically. All processing stays on our server.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-4 py-8">
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {!scan && <Uploader onFiles={onFiles} busy={busy} />}

        {scan && (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => {
                  setScan(null);
                  setError(null);
                }}
                className="rounded-lg bg-slate-100 px-4 py-2 text-sm text-slate-700 hover:bg-slate-200"
              >
                ← New scan
              </button>
              {lowCount > 0 && (
                <span className="rounded-lg bg-amber-100 px-3 py-2 text-sm text-amber-800">
                  {lowCount} page{lowCount > 1 ? "s" : ""} may need adjustment
                </span>
              )}
              <div className="ml-auto flex items-center gap-2">
                <select
                  value={encoding}
                  onChange={(e) => setEncoding(e.target.value as "jpeg" | "png")}
                  className="rounded-lg border border-slate-200 px-2 py-2 text-sm text-slate-700"
                  title="JPEG is smaller (lossy); PNG is lossless but larger"
                >
                  <option value="jpeg">JPEG (smaller)</option>
                  <option value="png">PNG (lossless)</option>
                </select>
                <button
                  disabled={busy || scan.pages.length === 0}
                  onClick={onExport}
                  className="rounded-lg bg-emerald-600 px-5 py-2 font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  Export PDF
                </button>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              {scan.pages.map((p, i) => (
                <PagePreview
                  key={p.page_id}
                  page={p}
                  index={i}
                  busy={busy}
                  onAdjust={() => setAdjusting(p)}
                  onRotate={async () => {
                    const r = await run(() => rotate(scan.scan_id, p.page_id, 90));
                    if (r) replacePage(r);
                  }}
                  onMode={async (mode) => {
                    const r = await run(() => setMode(scan.scan_id, p.page_id, mode));
                    if (r) replacePage(r);
                  }}
                  onDelete={async () => {
                    const r = await run(() => deletePage(scan.scan_id, p.page_id));
                    if (r) setScan(r);
                  }}
                />
              ))}
            </div>
          </>
        )}
      </main>

      {scan && adjusting && (
        <CornerAdjust
          scanId={scan.scan_id}
          page={adjusting}
          onCancel={() => setAdjusting(null)}
          onSave={async (corners) => {
            const r = await run(() => recrop(scan.scan_id, adjusting.page_id, corners));
            if (r) replacePage(r);
            setAdjusting(null);
          }}
        />
      )}
    </div>
  );
}
