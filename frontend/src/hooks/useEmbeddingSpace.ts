"use client";

import { useMemo } from "react";
import {
  useGetEmbeddingsPapersEmbeddingsGet,
  useGetClustersPapersClustersGet,
} from "@/api/queries/coordinates/coordinates";
import type { Paper, Cluster } from "@/components";

/**
 * Normalize coordinates to center them around the origin.
 * UMAP output can have arbitrary coordinates, so we:
 * 1. Calculate the centroid of all points
 * 2. Subtract the centroid from each point
 * 3. Scale to fit within a reasonable range
 */
function normalizeCoordinates(
  rawPapers: Array<{
    paper_id: string;
    arxiv_id: string;
    title: string;
    coords: [number, number, number];
    cluster_id: number | null;
    chunk_count: number;
  }>
): Paper[] {
  if (rawPapers.length === 0) return [];

  // Calculate centroid
  const centroid: [number, number, number] = [0, 0, 0];
  for (const paper of rawPapers) {
    centroid[0] += paper.coords[0];
    centroid[1] += paper.coords[1];
    centroid[2] += paper.coords[2];
  }
  centroid[0] /= rawPapers.length;
  centroid[1] /= rawPapers.length;
  centroid[2] /= rawPapers.length;

  // Calculate max distance from centroid for scaling
  let maxDist = 0;
  for (const paper of rawPapers) {
    const dx = paper.coords[0] - centroid[0];
    const dy = paper.coords[1] - centroid[1];
    const dz = paper.coords[2] - centroid[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    maxDist = Math.max(maxDist, dist);
  }

  // Scale factor to fit within a radius of ~3 units (camera is at [5,5,5])
  const targetRadius = 3;
  const scale = maxDist > 0 ? targetRadius / maxDist : 1;

  // Normalize each paper's coordinates
  return rawPapers.map((p) => ({
    paper_id: p.paper_id,
    arxiv_id: p.arxiv_id,
    title: p.title,
    coords: [
      (p.coords[0] - centroid[0]) * scale,
      (p.coords[1] - centroid[1]) * scale,
      (p.coords[2] - centroid[2]) * scale,
    ] as [number, number, number],
    cluster_id: p.cluster_id ?? -1,
    chunk_count: p.chunk_count,
  }));
}

/**
 * Hook to fetch and transform embedding space data for visualization.
 * Combines papers with coordinates and cluster information.
 */
export function useEmbeddingSpace() {
  const embeddingsQuery = useGetEmbeddingsPapersEmbeddingsGet();
  const clustersQuery = useGetClustersPapersClustersGet();

  // Transform API data to component-compatible format with normalized coordinates
  const papers: Paper[] = useMemo(() => {
    if (embeddingsQuery.data?.status !== 200) return [];

    const rawPapers = embeddingsQuery.data.data.papers.map((p) => ({
      paper_id: p.paper_id,
      arxiv_id: p.arxiv_id,
      title: p.title,
      coords: p.coords as [number, number, number],
      cluster_id: p.cluster_id,
      chunk_count: p.chunk_count,
    }));

    return normalizeCoordinates(rawPapers);
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
