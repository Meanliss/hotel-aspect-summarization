import { AppShell } from "@/components/AppShell";
import { WorkflowView } from "@/components/WorkflowView";

export default function WorkflowPage() {
  return (
    <AppShell page="workflow">
      <WorkflowView />
    </AppShell>
  );
}
