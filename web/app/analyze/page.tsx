import { TabNav } from "@/components/TabNav";
import { AnalyzeView } from "@/components/AnalyzeView";

export default function AnalyzePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">
          HASOS Sentiment Summarization
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Paste hotel reviews and run the model to get aspect summaries.
        </p>
      </header>
      <TabNav />
      <AnalyzeView />
    </div>
  );
}
