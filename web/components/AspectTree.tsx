"use client";

import { useState } from "react";
import type { Entity, ParentAspect, ChildAspect } from "@/lib/types";
import { SentimentColumn } from "./SentimentColumn";

function hasAnyContent(child: ChildAspect): boolean {
  const s = child.summaries;
  const e = child.evidence;
  return Boolean(
    s.positive ||
      s.negative ||
      s.neutral ||
      e.positive.length ||
      e.negative.length ||
      e.neutral.length,
  );
}

function ChildCard({ child }: { child: ChildAspect }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-2">
        <div className="text-sm font-semibold text-slate-800">
          {child.scale}
        </div>
        <div className="text-[11px] uppercase tracking-wide text-slate-400">
          {child.code}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <SentimentColumn
          sentiment="positive"
          summary={child.summaries.positive}
          evidence={child.evidence.positive}
        />
        <SentimentColumn
          sentiment="negative"
          summary={child.summaries.negative}
          evidence={child.evidence.negative}
        />
      </div>
      {child.summaries.neutral || child.evidence.neutral.length ? (
        <div className="mt-3">
          <SentimentColumn
            sentiment="neutral"
            summary={child.summaries.neutral}
            evidence={child.evidence.neutral}
          />
        </div>
      ) : null}
    </div>
  );
}

function ParentSection({ parent }: { parent: ParentAspect }) {
  const [open, setOpen] = useState(true);
  const children = parent.children.filter(hasAnyContent);
  if (children.length === 0) {
    return null;
  }
  return (
    <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between rounded-t-xl bg-slate-800 px-4 py-2 text-left text-white"
      >
        <span className="font-semibold">{parent.code}</span>
        <span className="text-xs text-slate-300">
          {children.length} aspects {open ? "▾" : "▸"}
        </span>
      </button>
      {open ? (
        <div className="space-y-3 p-4">
          {parent.summary ? (
            <p className="rounded bg-slate-100 px-3 py-2 text-sm text-slate-700">
              {parent.summary}
            </p>
          ) : null}
          {children.map((child) => (
            <ChildCard key={child.code} child={child} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

export function AspectTree({ entity }: { entity: Entity }) {
  const parents = entity.parents.filter((p) =>
    p.children.some(hasAnyContent),
  );
  return (
    <div className="space-y-4">
      {entity.overall_summary ? (
        <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-indigo-500">
            Overall summary
          </div>
          <p className="text-sm text-slate-800">{entity.overall_summary}</p>
        </div>
      ) : null}
      {parents.length === 0 ? (
        <p className="text-sm text-slate-500">
          No aspect summaries available for this entity.
        </p>
      ) : (
        parents.map((parent) => (
          <ParentSection key={parent.code} parent={parent} />
        ))
      )}
    </div>
  );
}
