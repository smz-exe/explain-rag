"use client";

import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

interface ResearchLayoutProps {
  /** Left panel content (papers list) */
  papersPanel: ReactNode;
  /** Right panel content (query/answer) */
  queryPanel: ReactNode;
  /** Optional: 3D visualization overlay (desktop only) */
  visualization?: ReactNode;
  /** Additional class names */
  className?: string;
}

/**
 * Split-panel layout for the Research Atlas page.
 *
 * Desktop: Papers panel (left) | Query panel (right)
 * Mobile: Stacked vertically (query first, then papers)
 *
 * 3D visualization is overlaid in the center on desktop, hidden on mobile.
 */
export function ResearchLayout({
  papersPanel,
  queryPanel,
  visualization,
  className,
}: ResearchLayoutProps) {
  return (
    <div className={cn("bg-atlas-background min-h-screen", className)}>
      {/* Mobile layout: stacked */}
      <div className="flex flex-col lg:hidden">
        {/* Query panel first on mobile for better UX */}
        <div className="border-atlas-border border-b">
          <div className="max-h-[60vh] overflow-y-auto p-4">{queryPanel}</div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">{papersPanel}</div>
      </div>

      {/* Desktop layout: split panels with optional center visualization */}
      {/* min-h-0 on grid items prevents content from expanding grid cells beyond h-screen */}
      <div className="hidden h-screen lg:grid lg:grid-cols-[320px_1fr_400px]">
        {/* Left panel: Papers */}
        <aside className="border-atlas-border bg-atlas-surface flex min-h-0 flex-col border-r">
          <div className="border-atlas-border shrink-0 border-b px-4 py-3">
            <h2 className="text-atlas-text-primary text-sm font-semibold tracking-wide uppercase">
              Papers
            </h2>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">{papersPanel}</div>
        </aside>

        {/* Center: 3D Visualization (or empty state) */}
        <main className="bg-atlas-background relative min-h-0 overflow-hidden">
          {visualization ? (
            <div className="absolute inset-0">{visualization}</div>
          ) : (
            <div className="text-atlas-text-secondary flex h-full items-center justify-center text-center">
              <div>
                <p className="text-sm">3D visualization will appear here</p>
                <p className="text-xs opacity-60">
                  once papers are added to the collection
                </p>
              </div>
            </div>
          )}
        </main>

        {/* Right panel: Query */}
        <aside className="border-atlas-border bg-atlas-surface flex min-h-0 flex-col border-l">
          <div className="border-atlas-border shrink-0 border-b px-4 py-3">
            <h2 className="text-atlas-text-primary text-sm font-semibold tracking-wide uppercase">
              Query
            </h2>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">{queryPanel}</div>
        </aside>
      </div>
    </div>
  );
}
