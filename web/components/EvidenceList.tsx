"use client";

import { useState } from "react";
import type { EvidenceItem } from "@/lib/types";

export function EvidenceList({ items }: { items: EvidenceItem[] }) {
  const [open, setOpen] = useState(false);
  if (!items || items.length === 0) {
    return null;
  }
  const shown = open ? items : items.slice(0, 3);
  return (
    <div className="mt-2">
      <ul className="space-y-1">
        {shown.map((it, i) => (
          <li
            key={i}
            className="rounded border border-slate-200 bg-white/60 px-2 py-1 text-xs text-slate-600"
          >
            <span className="text-slate-800">{it.sentence}</span>
            {it.review_id ? (
              <span className="ml-1 text-[10px] text-slate-400">
                ({it.review_id})
              </span>
            ) : null}
          </li>
        ))}
      </ul>
      {items.length > 3 ? (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="mt-1 text-xs font-medium text-slate-500 underline hover:text-slate-700"
        >
          {open ? "Show less" : `Show ${items.length - 3} more`}
        </button>
      ) : null}
    </div>
  );
}
