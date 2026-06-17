import { TabNav } from "@/components/TabNav";
import { WorkflowView } from "@/components/WorkflowView";

export default function WorkflowPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900">
          Aspect-Based Sentiment Summarization for the Hotel Domain
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          Follow how each method really operates on River Hotel: shared SemAE
          evidence first, then the M1-M4 branch differences.
        </p>
      </header>
      <TabNav />
      <WorkflowView />
    </div>
  );
}
