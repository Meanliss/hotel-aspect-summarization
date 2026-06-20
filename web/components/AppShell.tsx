import { TabNav } from "@/components/TabNav";

const PAGE_COPY: Record<string, { eyebrow: string; title: string; description: string }> = {
  explore: {
    eyebrow: "SPACE benchmark · 4 method comparison",
    title: "Architectural Analytics",
    description:
      "A synthesized perspective on hotel guest sentiment. Browse per-method overall and aspect summaries, then drill into the evidence behind every narrative.",
  },
  results: {
    eyebrow: "Evaluation layer",
    title: "ROUGE Method Results",
    description:
      "Compare extractive, abstractive, keyword-split, and BERT-ABSA sentiment-split methods across SPACE hotel summaries.",
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
      <nav className="sticky top-0 z-50 border-b border-[var(--outline-variant)] bg-[var(--surface-bright)]/95 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-[1200px] items-center justify-between px-4 md:px-6">
          <div className="flex items-center gap-8">
            <div className="font-headline text-xl font-bold text-[var(--primary)]">
              HotelInsight
            </div>
            <div className="hidden md:block">
              <TabNav compact />
            </div>
          </div>
          <div className="hidden items-center rounded-full border border-[var(--outline-variant)] bg-[var(--surface-container)] px-4 py-2 sm:flex">
            <span aria-hidden="true" className="mr-2 text-[15px] text-[var(--on-surface-variant)]">
              &#8981;
            </span>
            <input
              className="w-36 border-none bg-transparent p-0 font-body text-sm text-[var(--on-surface)] placeholder:text-[var(--on-surface-variant)] focus:ring-0 md:w-52"
              placeholder="Search sample ID..."
              type="text"
            />
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-[1200px] px-4 py-8 md:px-6 md:py-10">
        <header className="mb-10">
          <div className="mb-3 font-body text-xs font-semibold uppercase tracking-[0.18em] text-[var(--secondary)]">
            {copy.eyebrow}
          </div>
          <h1 className="font-headline text-4xl font-bold leading-tight text-[var(--primary)] md:text-5xl">
            {copy.title}
          </h1>
          <p className="mt-4 max-w-3xl font-body text-lg leading-relaxed text-[var(--on-surface-variant)]">
            {copy.description}
          </p>
        </header>

        <div className="mb-6 md:hidden">
          <TabNav />
        </div>
        <section className="rounded-xl bg-[var(--surface-container-low)] p-3 md:p-4">
          {children}
        </section>
      </main>
    </>
  );
}
