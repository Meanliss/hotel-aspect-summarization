"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "Paper" },
  { href: "/results", label: "Results" },
  { href: "/optimality", label: "Optimality" },
  { href: "/workflow", label: "Workflow" },
  { href: "/trace", label: "Trace" },
];

export function TabNav({ compact = false }: { compact?: boolean }) {
  const pathname = usePathname();
  return (
    <nav
      className={
        compact
          ? "flex items-center gap-1"
          : "mb-6 flex gap-1 border-b border-slate-200 pb-2"
      }
    >
      {TABS.map((tab) => {
        const active =
          tab.href === "/" ? pathname === "/" : pathname.startsWith(tab.href);
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
              active
                ? compact
                  ? "bg-[var(--primary)] text-white"
                  : "bg-[var(--primary)] text-white"
                : compact
                  ? "text-[var(--on-surface-variant)] hover:bg-[var(--surface-container)] hover:text-[var(--primary)]"
                  : "text-[var(--on-surface-variant)] hover:bg-[var(--surface-container)] hover:text-[var(--primary)]"
            }`}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
