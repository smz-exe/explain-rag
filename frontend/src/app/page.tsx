"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import {
  ResearchLayout,
  PapersPanel,
  QueryPanel,
  EmbeddingSpace,
} from "@/components";
import { visualization } from "@/lib/design-tokens";
import type { QueryCoords, QueryInputHandle } from "@/components";
import type { QueryResponse } from "@/api/model";
import { useQueryQueryPost } from "@/api/queries/query/query";
import { useEmbeddingSpace } from "@/hooks/useEmbeddingSpace";

export default function Home() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string>();
  const [prefilledQuestion, setPrefilledQuestion] = useState<string>();
  const [selectedPaperId, setSelectedPaperId] = useState<string>();

  const queryInputRef = useRef<QueryInputHandle>(null);

  // Fetch embedding space data (papers with coordinates + clusters)
  const { papers, clusters, isLoading: isLoadingSpace } = useEmbeddingSpace();

  const queryMutation = useQueryQueryPost({
    mutation: {
      onSuccess: (result) => {
        if (result.status === 200) {
          setResponse(result.data);
        }
      },
    },
  });

  // Keyboard shortcut: / to focus search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input or textarea
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      if (e.key === "/") {
        e.preventDefault();
        queryInputRef.current?.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleSubmit = useCallback(
    (query: { question: string; top_k: number; enable_reranking: boolean }) => {
      setPrefilledQuestion(undefined);
      queryMutation.mutate({ data: query });
    },
    [queryMutation]
  );

  const handleCitationClick = (chunkId: string) => {
    setHighlightedChunkId(chunkId);
    requestAnimationFrame(() => {
      const element = document.getElementById(`chunk-${chunkId}`);
      element?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  // Extract retrieved paper IDs from query response
  const queryCoords: QueryCoords | undefined = useMemo(() => {
    if (!response?.retrieved_chunks?.length) return undefined;

    // Get unique paper IDs from retrieved chunks
    const retrievedPaperIds = [
      ...new Set(response.retrieved_chunks.map((chunk) => chunk.paper_id)),
    ];

    // Find centroid of retrieved papers for query visualization
    const retrievedPapers = papers.filter((p) =>
      retrievedPaperIds.includes(p.paper_id)
    );

    if (retrievedPapers.length === 0) return undefined;

    // Calculate centroid of retrieved papers as query position
    const centroid: [number, number, number] = [0, 0, 0];
    for (const paper of retrievedPapers) {
      centroid[0] += paper.coords[0];
      centroid[1] += paper.coords[1];
      centroid[2] += paper.coords[2];
    }
    centroid[0] /= retrievedPapers.length;
    centroid[1] /= retrievedPapers.length;
    centroid[2] /= retrievedPapers.length;

    // Offset so query point floats above the centroid of retrieved papers
    centroid[1] += visualization.queryPoint.verticalOffset;

    return {
      coords: centroid,
      retrievedPaperIds,
    };
  }, [response, papers]);

  return (
    <ResearchLayout
      papersPanel={
        <PapersPanel
          papers={papers}
          clusters={clusters}
          isLoading={isLoadingSpace}
          selectedPaperId={selectedPaperId}
          onPaperSelect={(paperId) => {
            setSelectedPaperId(paperId);
          }}
          onAskAboutPaper={(paper) => {
            setPrefilledQuestion(
              `What is the main contribution of "${paper.title}"?`
            );
            // Focus the query input after setting the question
            requestAnimationFrame(() => {
              queryInputRef.current?.focus();
            });
          }}
        />
      }
      queryPanel={
        <QueryPanel
          ref={queryInputRef}
          response={response}
          isLoading={queryMutation.isPending}
          error={
            queryMutation.isError
              ? queryMutation.error instanceof Error
                ? queryMutation.error
                : new Error("An unexpected error occurred")
              : null
          }
          highlightedChunkId={highlightedChunkId}
          prefilledQuestion={prefilledQuestion}
          onSubmit={handleSubmit}
          onRetry={() => queryMutation.reset()}
          onCitationClick={handleCitationClick}
        />
      }
      visualization={
        <EmbeddingSpace
          papers={papers}
          selectedPaperId={selectedPaperId}
          queryCoords={queryCoords}
          isLoading={isLoadingSpace}
          onPaperClick={(paperId) => {
            setSelectedPaperId(paperId);
          }}
          onPaperHover={() => {
            // Could highlight in sidebar on hover
          }}
        />
      }
    />
  );
}
