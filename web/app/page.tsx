import { TabNav } from "@/components/TabNav";
import { SpaceExplore } from "@/components/SpaceExplore";

export default function Page() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">
          Aspect-Based Sentiment Summarization for the Hotel Domain
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Aspect-based sentiment summarization on the SPACE hotel benchmark —
          browse per-method overall and aspect summaries with positive /
          negative split.
        </p>
      </header>
      <TabNav />
      <SpaceExplore />
    </div>
  );
}
