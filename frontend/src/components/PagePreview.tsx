import type { Page } from "../api/client";
import { previewSrc } from "../api/client";

interface Props {
  page: Page;
  index: number;
  busy: boolean;
  onRotate: () => void;
  onMode: (mode: string) => void;
  onAdjust: () => void;
  onDemoire: (on: boolean) => void;
  onDelete: () => void;
}

const FLAG_LABELS: Record<string, string> = {
  blur: "Blurry",
  glare: "Glare",
  low_resolution: "Low resolution",
  boundary_not_found: "Edges not found",
  geometry_uncertain: "Edges unclear — try Adjust",
  approx_crop: "Approx. crop",
  born_digital: "Born-digital — kept as-is",
  sanity_implausible_aspect: "Odd shape",
  sanity_mostly_uniform: "Looks blank",
};

// Shows the ALREADY-CLEANED page (auto pipeline ran on upload). Manual controls
// are secondary; "Adjust" is highlighted only when confidence is low.
export default function PagePreview({
  page,
  index,
  busy,
  onRotate,
  onMode,
  onAdjust,
  onDemoire,
  onDelete,
}: Props) {
  const pct = Math.round(page.confidence * 100);
  const confColor = page.low_confidence
    ? "bg-amber-100 text-amber-800"
    : "bg-emerald-100 text-emerald-800";

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-100 px-3 py-2">
        <span className="text-sm font-medium text-slate-600">Page {index + 1}</span>
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${confColor}`}>
          {pct}% confident
        </span>
      </div>

      <div className="bg-slate-100 p-2">
        <img
          src={previewSrc(page)}
          alt={`page ${index + 1}`}
          className="mx-auto max-h-80 w-auto rounded shadow"
        />
      </div>

      {page.gate_flags.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pt-2">
          {page.gate_flags.map((f) => (
            <span
              key={f}
              className={`rounded px-1.5 py-0.5 text-xs ${
                f === "born_digital"
                  ? "bg-sky-50 text-sky-700"
                  : "bg-amber-50 text-amber-700"
              }`}
            >
              {FLAG_LABELS[f] ?? f}
            </span>
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 p-3">
        {!page.passthrough && (
          <button
            disabled={busy}
            onClick={onAdjust}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${
              page.low_confidence
                ? "bg-amber-500 text-white hover:bg-amber-600"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            Adjust
          </button>
        )}
        <button
          disabled={busy}
          onClick={onRotate}
          className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-200 disabled:opacity-50"
        >
          Rotate
        </button>
        {!page.passthrough && (
          <select
            disabled={busy}
            value={page.mode}
            onChange={(e) => onMode(e.target.value)}
            className="rounded-lg border border-slate-200 px-2 py-1.5 text-sm text-slate-700 disabled:opacity-50"
          >
            <option value="color">Color</option>
            <option value="gray">Grayscale</option>
            <option value="bw">B&amp;W</option>
          </select>
        )}
        {!page.passthrough && (
          <button
            disabled={busy}
            onClick={() => onDemoire(!page.demoire)}
            title="Remove screen-photo moiré with an ML model (slower)"
            className={`rounded-lg px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${
              page.demoire
                ? "bg-violet-600 text-white hover:bg-violet-700"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            De-moiré{page.demoire ? " ✓" : ""}
          </button>
        )}
        <button
          disabled={busy}
          onClick={onDelete}
          className="ml-auto rounded-lg px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 disabled:opacity-50"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
