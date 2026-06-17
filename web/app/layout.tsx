import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aspect-Based Sentiment Summarization for the Hotel Domain",
  description:
    "Aspect-based sentiment summarization for the hotel domain — 4-method comparison on the SPACE benchmark.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
