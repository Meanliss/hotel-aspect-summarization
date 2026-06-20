import { AppShell } from "@/components/AppShell";
import { PipelineTraceView } from "@/components/PipelineTraceView";

export default function TracePage() {
  return <AppShell page="trace"><PipelineTraceView /></AppShell>;
}
