import { useEffect, useRef, useState } from "react";
import type { Page } from "../api/client";

interface Props {
  scanId: string;
  page: Page;
  onCancel: () => void;
  onSave: (corners: number[][]) => void;
}

const DEFAULT_QUAD = [
  [0.08, 0.08],
  [0.92, 0.08],
  [0.92, 0.92],
  [0.08, 0.92],
];

// Manual fallback (FR-31): drag the four corners over the ORIGINAL image, then
// re-run the warp. Surfaced only when the user chooses "Adjust" (auto is primary).
export default function CornerAdjust({ scanId, page, onCancel, onSave }: Props) {
  const [corners, setCorners] = useState<number[][]>(page.quad ?? DEFAULT_QUAD);
  const [dragging, setDragging] = useState<number | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const originalUrl = `/api/scans/${scanId}/pages/${page.page_id}/preview?original=1`;

  useEffect(() => {
    function move(e: PointerEvent) {
      if (dragging === null || !boxRef.current) return;
      const r = boxRef.current.getBoundingClientRect();
      const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
      setCorners((c) => c.map((p, i) => (i === dragging ? [x, y] : p)));
    }
    function up() {
      setDragging(null);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging]);

  const poly = corners.map((p) => `${p[0] * 100},${p[1] * 100}`).join(" ");

  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/60 p-4">
      <div className="flex max-h-full w-full max-w-2xl flex-col rounded-2xl bg-white p-4">
        <h3 className="mb-2 text-lg font-semibold">Adjust corners</h3>
        <p className="mb-3 text-sm text-slate-500">
          Drag the handles to the document corners, then re-crop.
        </p>
        <div
          ref={boxRef}
          className="relative max-h-[60vh] select-none overflow-hidden rounded-lg bg-slate-100"
        >
          <img src={originalUrl} alt="original" className="block w-full" draggable={false} />
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="pointer-events-none absolute inset-0 h-full w-full"
          >
            <polygon points={poly} fill="rgba(37,99,235,0.15)" stroke="#2563eb" strokeWidth={0.5} />
          </svg>
          {corners.map((p, i) => (
            <button
              key={i}
              onPointerDown={() => setDragging(i)}
              style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%` }}
              className="absolute h-6 w-6 -translate-x-1/2 -translate-y-1/2 cursor-grab touch-none rounded-full border-2 border-white bg-blue-600 shadow active:cursor-grabbing"
            />
          ))}
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <button onClick={onCancel} className="rounded-lg px-4 py-2 text-slate-600 hover:bg-slate-100">
            Cancel
          </button>
          <button
            onClick={() => onSave(corners)}
            className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
          >
            Re-crop
          </button>
        </div>
      </div>
    </div>
  );
}
