"use client";

import { forwardRef } from "react";
import { cn } from "@/lib/utils";
import {
  QueryInput,
  AnswerDisplay,
  ChunksPanel,
  FaithfulnessReport,
  TimingTrace,
  ErrorDisplay,
  AnswerSkeleton,
  ChunksSkeleton,
  FaithfulnessSkeleton,
  TimingSkeleton,
} from "@/components";
import type { QueryInputHandle } from "@/components/QueryInput";
import type { QueryResponse } from "@/api/model";

interface QueryPanelProps {
  /** Current query response */
  response: QueryResponse | null;
  /** Whether a query is in progress */
  isLoading: boolean;
  /** Error from query mutation */
  error: Error | null;
  /** Currently highlighted chunk ID */
  highlightedChunkId?: string;
  /** Prefilled question (e.g., from "Ask about this paper") */
  prefilledQuestion?: string;
  /** Submit handler */
  onSubmit: (query: {
    question: string;
    top_k: number;
    enable_reranking: boolean;
  }) => void;
  /** Retry handler */
  onRetry: () => void;
  /** Citation click handler */
  onCitationClick: (chunkId: string) => void;
  className?: string;
}

/**
 * Query panel containing input, answer display, chunks, and analysis.
 * Designed for the right side of the Research Atlas layout.
 */
export const QueryPanel = forwardRef<QueryInputHandle, QueryPanelProps>(
  function QueryPanel(
    {
      response,
      isLoading,
      error,
      highlightedChunkId,
      prefilledQuestion,
      onSubmit,
      onRetry,
      onCitationClick,
      className,
    },
    ref
  ) {
    return (
      <div className={cn("flex flex-col gap-4 p-4", className)}>
        {/* Query Input */}
        <section>
          <QueryInput
            ref={ref}
            onSubmit={onSubmit}
            isLoading={isLoading}
            defaultQuestion={prefilledQuestion}
          />
        </section>

        {/* Error State */}
        {error && (
          <section>
            <ErrorDisplay error={error} onRetry={onRetry} />
          </section>
        )}

        {/* Loading State */}
        {isLoading && (
          <div className="space-y-4">
            <AnswerSkeleton />
            <FaithfulnessSkeleton />
            <ChunksSkeleton />
            <TimingSkeleton />
          </div>
        )}

        {/* Results */}
        {response && !isLoading && (
          <div className="space-y-4">
            {/* Answer */}
            <section>
              <AnswerDisplay
                queryId={response.query_id}
                question={response.question}
                answer={response.answer}
                citations={response.citations}
                onCitationClick={onCitationClick}
              />
            </section>

            {/* Faithfulness Report */}
            <section>
              <FaithfulnessReport faithfulness={response.faithfulness} />
            </section>

            {/* Retrieved Chunks */}
            <section>
              <ChunksPanel
                chunks={response.retrieved_chunks}
                highlightedChunkId={highlightedChunkId}
              />
            </section>

            {/* Timing Trace */}
            <section>
              <TimingTrace trace={response.trace} />
            </section>
          </div>
        )}

        {/* Empty State */}
        {!response && !isLoading && !error && (
          <div className="text-atlas-text-secondary py-12 text-center">
            <p className="mb-1 text-sm">Ask a question about the papers</p>
            <p className="text-xs opacity-60">
              Your answer will appear here with citations and analysis
            </p>
            <p className="mt-4 text-xs opacity-40">
              Press <kbd className="bg-atlas-background rounded px-1 py-0.5 font-mono">/</kbd> to focus search
            </p>
          </div>
        )}
      </div>
    );
  }
);
