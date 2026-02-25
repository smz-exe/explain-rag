"use client";

import { useMemo } from "react";
import { Line } from "@react-three/drei";
import { hexToThreeColor, visualization } from "@/lib/design-tokens";
import type { Paper } from "./PapersPanel";

interface ConnectionLinesProps {
  queryCoords: [number, number, number];
  papers: Paper[];
  connectedPaperIds: string[];
}

/**
 * Lines connecting the query point to retrieved papers.
 * Visual representation of the retrieval relationship.
 */
export function ConnectionLines({
  queryCoords,
  papers,
  connectedPaperIds,
}: ConnectionLinesProps) {
  // Build coordinate map only from paper IDs and coords (not entire paper objects)
  const paperCoordsMap = useMemo(() => {
    return new Map(papers.map((p) => [p.paper_id, p.coords]));
  }, [papers]);

  const lines = useMemo(() => {
    return connectedPaperIds.flatMap((paperId, index) => {
      const coords = paperCoordsMap.get(paperId);
      if (!coords) return [];

      return [{
        paperId,
        points: [queryCoords, coords] as [
          [number, number, number],
          [number, number, number]
        ],
        // Fade opacity based on rank (first result = most opaque)
        opacity: Math.max(0.2, 1 - index * 0.1),
      }];
    });
  }, [queryCoords, paperCoordsMap, connectedPaperIds]);

  const lineColor = useMemo(
    () => hexToThreeColor(visualization.connectionLine.color),
    []
  );

  return (
    <group>
      {lines.map((line) => (
        <Line
          key={line.paperId}
          points={line.points}
          color={lineColor}
          lineWidth={1.5}
          opacity={line.opacity * visualization.connectionLine.opacity}
          transparent
          dashed
          dashSize={0.1}
          gapSize={0.05}
        />
      ))}
    </group>
  );
}
