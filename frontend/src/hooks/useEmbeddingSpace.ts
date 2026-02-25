"use client";

import { useMemo } from "react";
import {
  useGetEmbeddingsPapersEmbeddingsGet,
  useGetClustersPapersClustersGet,
} from "@/api/queries/coordinates/coordinates";
import type { Paper, Cluster } from "@/components";

/**
 * Hook to fetch and transform embedding space data for visualization.
 * Combines papers with coordinates and cluster information.
 */
export function useEmbeddingSpace() {
  const embeddingsQuery = useGetEmbeddingsPapersEmbeddingsGet();
  const clustersQuery = useGetClustersPapersClustersGet();

  // Transform API data to component-compatible format
  const papers: Paper[] = useMemo(() => {
    if (embeddingsQuery.data?.status !== 200) return [];

    return embeddingsQuery.data.data.papers.map((p) => ({
      paper_id: p.paper_id,
      arxiv_id: p.arxiv_id,
      title: p.title,
      coords: p.coords,
      // Convert null cluster_id to -1 (noise/unclustered)
      cluster_id: p.cluster_id ?? -1,
      chunk_count: p.chunk_count,
    }));
  }, [embeddingsQuery.data]);

  const clusters: Cluster[] = useMemo(() => {
    if (clustersQuery.data?.status !== 200) return [];

    return clustersQuery.data.data.clusters.map((c) => ({
      id: c.id,
      label: c.label,
      paper_ids: c.paper_ids,
    }));
  }, [clustersQuery.data]);

  const computedAt = useMemo(() => {
    if (embeddingsQuery.data?.status !== 200) return null;
    return embeddingsQuery.data.data.computed_at;
  }, [embeddingsQuery.data]);

  return {
    papers,
    clusters,
    computedAt,
    isLoading: embeddingsQuery.isLoading || clustersQuery.isLoading,
    isError: embeddingsQuery.isError || clustersQuery.isError,
    error: embeddingsQuery.error || clustersQuery.error,
    refetch: () => {
      embeddingsQuery.refetch();
      clustersQuery.refetch();
    },
  };
}
