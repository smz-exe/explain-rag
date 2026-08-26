"use client";

import { forwardRef, useEffect, useState } from "react";
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

/** Gap between deferred-verification polls. */
const POLL_INTERVAL_MS = 2000;
/** How long to wait for a deferred verification before declaring it failed. */
const POLL_TIMEOUT_MS = 120_000;

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
    const queryId = response?.query_id;

    // Verification runs as a best-effort background task, so a query can stay
    // "pending" forever if the server process died before finishing it. Bound
    // the wait instead of spinning indefinitely.
    const [pollTimedOut, setPollTimedOut] = useState(false);
    useEffect(() => {
      setPollTimedOut(false);
      if (!queryId || !verificationPending) return;
      const timer = setTimeout(() => setPollTimedOut(true), POLL_TIMEOUT_MS);
      return () => clearTimeout(timer);
    }, [queryId, verificationPending]);

    const faithfulnessPoll = useGetFaithfulnessQueryQueryIdFaithfulnessGet(
      queryId ?? "",
      {
        query: {
          enabled:
            !!queryId && !!shareToken && verificationPending && !pollTimedOut,
          refetchInterval: (query) => {
            const data = query.state.data;
            if (data?.status === 200 && data.data.status !== "pending") {
              return false;
            }
            // A failing poll (expired token, query gone) must not retry forever.
            if (query.state.status === "error") return false;
            return POLL_INTERVAL_MS;
          },
        },
        request: shareToken
          ? { headers: { Authorization: `Bearer ${shareToken}` } }
          : undefined,
      }
    );
    const polled =
      faithfulnessPoll.data?.status === 200 ? faithfulnessPoll.data.data : null;
    // Resolve to a definite outcome rather than an endless spinner.
    const pollGaveUp =
      verificationPending && (faithfulnessPoll.isError || pollTimedOut);

    const faithfulness = response?.faithfulness ?? polled?.faithfulness ?? null;
    const faithfulnessStatus = response?.faithfulness
      ? "completed"
      : pollGaveUp
        ? "failed"
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
