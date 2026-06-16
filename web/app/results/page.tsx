import { TabNav } from "@/components/TabNav";
import { ResultsView } from "@/components/ResultsView";

export default function ResultsPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">
          HASOS Sentiment Summarization
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          ROUGE comparison of 4 methods (extractive / abstractive / sentiment-split)
          on SPACE and HASOS benchmarks.
        </p>
      </header>
      <TabNav />
      <ResultsView />
    </div>
  );
}
