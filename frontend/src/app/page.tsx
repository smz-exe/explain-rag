"use client";

import { useState } from "react";
import {
  QueryInput,
  AnswerDisplay,
  ChunksPanel,
  FaithfulnessReport,
  TimingTrace,
} from "@/components";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

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
          <Card className="border-destructive mb-8">
            <CardContent className="pt-6">
              <p className="text-destructive">
                Error:{" "}
                {queryMutation.error instanceof Error
                  ? queryMutation.error.message
                  : "An error occurred"}
              </p>
            </CardContent>
          </Card>
        )}

        {queryMutation.isPending && (
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <Skeleton className="h-6 w-24" />
              </CardHeader>
              <CardContent className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
              </CardContent>
            </Card>
          </div>
        )}

        {response && !queryMutation.isPending && (
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="space-y-6 lg:col-span-2">
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
            <div className="space-y-6">
              <FaithfulnessReport faithfulness={response.faithfulness} />
              <TimingTrace trace={response.trace} />
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
