import { TabNav } from "@/components/TabNav";

const PAGE_COPY: Record<
  string,
  { eyebrow: string; title: string; description: string }
> = {
  explore: {
    eyebrow: "Research article",
    title: "Hotel Review Summarization Report",
    description:
      "A paper-style presentation of the SPACE and HASOS experiments, with method results, threshold optimality, coverage checks, and the current evidence-backed conclusion.",
  },
  results: {
    eyebrow: "Evaluation layer",
    title: "ROUGE Method Results",
    description:
      "Compare extractive, abstractive, keyword-split, and BERT-ABSA sentiment-split methods across SPACE and HASOS.",
  },
  optimality: {
    eyebrow: "Hyperparameter study",
    title: "Threshold Optimality",
    description:
      "Grid-search evidence for the shipped evidence thresholds. Token-budget scaffolding is present, but no token cells have been run yet.",
  },
  workflow: {
    eyebrow: "Method walkthrough",
    title: "Pipeline Workflow",
    description:
      "Follow how each method operates on a representative review sample: shared SemAE evidence first, then the M1-M4 branch differences.",
  },
  trace: {
    eyebrow: "Sentence-level audit",
    title: "Pipeline Trace",
    description:
      "Inspect the actual sentences flowing through raw reviews, SemAE selection, sentiment branching, and generated method outputs.",
  },
};

export function AppShell({
  page,
  children,
}: {
  page: keyof typeof PAGE_COPY;
  children: React.ReactNode;
}) {
  const copy = PAGE_COPY[page];
  return (
    <>
      <nav className="sticky top-0 z-50 border-b border-[var(--rule)] bg-[var(--paper)]/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-4 md:px-6">
          <div className="flex items-center gap-8">
            <div className="font-headline text-xl font-bold text-[var(--ink)]">
              HotelSumm
            </div>
            <div className="hidden md:block">
              <TabNav compact />
            </div>
          </div>
          <div className="hidden items-center gap-2 border-l border-[var(--rule)] pl-4 text-xs text-[var(--muted)] sm:flex">
            <span>SPACE</span>
            <span aria-hidden="true">/</span>
            <span>HASOS</span>
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-[1200px] px-4 py-8 md:px-6 md:py-10">
        <header className="mb-8 border-b border-[var(--rule)] pb-6">
          <div className="mb-3 font-body text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
            {copy.eyebrow}
          </div>
          <h1 className="font-headline text-4xl font-bold leading-tight text-[var(--ink)] md:text-5xl">
            {copy.title}
          </h1>
          <p className="mt-4 max-w-3xl font-body text-lg leading-relaxed text-[var(--muted)]">
            {copy.description}
          </p>
        </header>

        <div className="mb-6 md:hidden">
          <TabNav />
        </div>
        {children}
      </main>
    </>
  );
}
