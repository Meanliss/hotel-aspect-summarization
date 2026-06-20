import { AppShell } from "@/components/AppShell";
import { HotelNarrativeDashboard } from "@/components/HotelNarrativeDashboard";

export default function Page() {
  return (
    <AppShell page="explore">
      <HotelNarrativeDashboard />
    </AppShell>
  );
}
