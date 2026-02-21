"use client";

import { useState } from "react";
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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// Generated types from orval
import type { QueryResponse } from "@/api/model";
import { useQueryQueryPost } from "@/api/queries/query/query";

export default function Home() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string>();

  const queryMutation = useQueryQueryPost({
    mutation: {
      onSuccess: (result) => {
        if (result.status === 200) {
          setResponse(result.data);
        }
      },
    },
  });

  const handleSubmit = (query: {
    question: string;
    top_k: number;
    enable_reranking: boolean;
  }) => {
    queryMutation.mutate({ data: query });
  };

  const handleCitationClick = (chunkId: string) => {
    setHighlightedChunkId(chunkId);
    // Scroll to the chunk
    const element = document.getElementById(`chunk-${chunkId}`);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  return (
    <main className="bg-background min-h-screen">
      <div className="container mx-auto max-w-6xl px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">ExplainRAG</h1>
          <p className="text-muted-foreground">
            Explainable RAG for Academic Papers
          </p>
        </header>

        <div className="mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Ask a Question</CardTitle>
            </CardHeader>
            <CardContent>
              <QueryInput
                onSubmit={handleSubmit}
                isLoading={queryMutation.isPending}
              />
            </CardContent>
          </Card>
        </div>

        {queryMutation.isError && (
          <div className="mb-8">
            <ErrorDisplay
              error={
                queryMutation.error instanceof Error
                  ? queryMutation.error
                  : new Error("An unexpected error occurred")
              }
              onRetry={() => queryMutation.reset()}
            />
          </div>
        )}

        {queryMutation.isPending && (
          <div className="grid gap-4 md:grid-cols-2 md:gap-6 lg:grid-cols-3">
            <div className="space-y-4 md:col-span-1 md:space-y-6 lg:col-span-2">
              <AnswerSkeleton />
              <ChunksSkeleton />
            </div>
            <div className="space-y-4 md:space-y-6">
              <FaithfulnessSkeleton />
              <TimingSkeleton />
            </div>
          </div>
        )}

        {response && !queryMutation.isPending && (
          <div className="grid gap-4 md:grid-cols-2 md:gap-6 lg:grid-cols-3">
            <div className="space-y-4 md:col-span-1 md:space-y-6 lg:col-span-2">
              <AnswerDisplay
                question={response.question}
                answer={response.answer}
                citations={response.citations}
                onCitationClick={handleCitationClick}
              />
              <ChunksPanel
                chunks={response.retrieved_chunks}
                highlightedChunkId={highlightedChunkId}
              />
            </div>
            <div className="space-y-4 md:space-y-6">
              <FaithfulnessReport faithfulness={response.faithfulness} />
              <TimingTrace trace={response.trace} />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
