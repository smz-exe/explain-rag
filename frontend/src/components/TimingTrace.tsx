"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight, Clock } from "lucide-react";
import { useState } from "react";
import type { ExplanationTrace } from "@/api/model";

interface TimingTraceProps {
  trace: ExplanationTrace;
}

export function TimingTrace({ trace }: TimingTraceProps) {
  const [isOpen, setIsOpen] = useState(false);

  const formatTime = (ms: number | null) => {
    if (ms === null) return "-";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const timings = [
    { label: "Embedding", value: trace.embedding_time_ms },
    { label: "Retrieval", value: trace.retrieval_time_ms },
    { label: "Reranking", value: trace.reranking_time_ms },
    { label: "Generation", value: trace.generation_time_ms },
    { label: "Faithfulness", value: trace.faithfulness_time_ms },
  ].filter((t): t is { label: string; value: number } => t.value != null);

  // Calculate percentage of total for visualization
  const maxTime = Math.max(...timings.map((t) => t.value ?? 0));

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Clock className="h-4 w-4" />
            Timing
          </CardTitle>
          <span className="font-mono text-sm">
            {formatTime(trace.total_time_ms)}
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <CollapsibleTrigger className="text-muted-foreground hover:text-foreground flex items-center gap-2 text-sm">
            {isOpen ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            {isOpen ? "Hide" : "Show"} breakdown
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4 space-y-2">
            {timings.map(({ label, value }) => (
              <div key={label} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">{label}</span>
                  <span className="font-mono">{formatTime(value)}</span>
                </div>
                <div className="bg-muted h-2 overflow-hidden rounded-full">
                  <div
                    className="bg-primary h-full rounded-full transition-all"
                    style={{
                      width: `${((value ?? 0) / maxTime) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}
