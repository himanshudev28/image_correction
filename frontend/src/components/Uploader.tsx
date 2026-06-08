import { useRef, useState } from "react";

interface Props {
  onFiles: (files: File[]) => void;
  busy: boolean;
}

// Multi-file upload (images + PDF). Live camera is deferred to M2 (NG/Non-Goals).
export default function Uploader({ onFiles, busy }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  function pick(files: FileList | null) {
    if (!files || files.length === 0) return;
    onFiles(Array.from(files));
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        pick(e.dataTransfer.files);
      }}
      className={`rounded-2xl border-2 border-dashed p-12 text-center transition ${
        dragOver ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-white"
      }`}
    >
      <p className="text-lg font-medium text-slate-700">
        Drop a photo or PDF of a consent form
      </p>
      <p className="mt-1 text-sm text-slate-500">
        JPEG, PNG, HEIC, or PDF · cropping, straightening &amp; cleanup happen
        automatically
      </p>
      <button
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        className="mt-6 rounded-lg bg-blue-600 px-5 py-2.5 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {busy ? "Processing…" : "Choose files"}
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="image/jpeg,image/png,image/heic,image/heif,application/pdf"
        className="hidden"
        onChange={(e) => pick(e.target.files)}
      />
    </div>
  );
}
