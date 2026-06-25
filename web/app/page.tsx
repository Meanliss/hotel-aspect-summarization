import { AppShell } from "@/components/AppShell";
import { ResearchPaperView } from "@/components/ResearchPaperView";

export default function Page() {
  return (
    <AppShell page="explore">
      <ResearchPaperView />
    </AppShell>
  );
}
