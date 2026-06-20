import { AppShell } from "@/components/AppShell";
import { ResultsView } from "@/components/ResultsView";

export default function ResultsPage() {
  return (
    <AppShell page="results">
      <ResultsView />
    </AppShell>
  );
}
