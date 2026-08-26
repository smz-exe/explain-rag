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
import type { QueryCreatedResponse } from "@/api/model";
import { useGetFaithfulnessQueryQueryIdFaithfulnessGet } from "@/api/queries/query/query";

interface QueryPanelProps {
  /** Current query response, including the token authorizing reads of it */
  response: QueryCreatedResponse | null;
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
    // Verification is deferred server-side: poll until it completes.
    // Reading the report needs the capability token issued with the query.
    const shareToken = response?.share_token;
    const verificationPending = response?.faithfulness_status === "pending";
    const faithfulnessPoll = useGetFaithfulnessQueryQueryIdFaithfulnessGet(
      response?.query_id ?? "",
      {
        query: {
          enabled: !!response && !!shareToken && verificationPending,
          refetchInterval: (query) => {
            const data = query.state.data;
            const done = data?.status === 200 && data.data.status !== "pending";
            return done ? false : 2000;
          },
        },
        request: shareToken
          ? { headers: { Authorization: `Bearer ${shareToken}` } }
          : undefined,
      }
    );
    const polled =
      faithfulnessPoll.data?.status === 200 ? faithfulnessPoll.data.data : null;
    const faithfulness = response?.faithfulness ?? polled?.faithfulness ?? null;
    const faithfulnessStatus = response?.faithfulness
      ? "completed"
      : (polled?.status ?? response?.faithfulness_status);

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
                shareToken={response.share_token}
                question={response.question}
                answer={response.answer}
                citations={response.citations}
                onCitationClick={onCitationClick}
              />
            </section>

            {/* Faithfulness Report */}
            <section>
              <FaithfulnessReport
                faithfulness={faithfulness}
                status={faithfulnessStatus}
              />
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
              Press{" "}
              <kbd className="bg-atlas-background rounded px-1 py-0.5 font-mono">
                /
              </kbd>{" "}
              to focus search
            </p>
          </div>
        )}
      </div>
    );
  }
);
