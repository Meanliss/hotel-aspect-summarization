import { TabNav } from "@/components/TabNav";
import { ExploreView } from "@/components/ExploreView";

export default function Page() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">
          HASOS Sentiment Summarization
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Aspect-based opinion summaries with positive / negative split, ranked
          from SemAE evidence.
        </p>
      </header>
      <TabNav />
      <ExploreView />
    </div>
  );
}
