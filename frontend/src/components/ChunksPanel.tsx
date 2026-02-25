"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChevronDown,
  ChevronRight,
  ArrowUp,
  ArrowDown,
  Minus,
} from "lucide-react";
import { useState } from "react";
import type { RetrievedChunk } from "@/api/model";

interface ChunksPanelProps {
  chunks: RetrievedChunk[];
  highlightedChunkId?: string;
}

function getRankChangeDisplay(
  originalRank: number,
  rank: number,
  rerankScore?: number | null
) {
  // Only show rank change when reranking was applied
  if (rerankScore == null) {
    return null;
  }

  const change = originalRank - rank;

  if (change > 0) {
    // Promoted (moved up in ranking)
    return {
      Icon: ArrowUp,
      text: `+${change}`,
      className:
        "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
      tooltip: `Promoted ${change} position${change > 1 ? "s" : ""} by reranking (was #${originalRank})`,
    };
  } else if (change < 0) {
    // Demoted (moved down in ranking)
    return {
      Icon: ArrowDown,
      text: `${change}`,
      className: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
      tooltip: `Demoted ${Math.abs(change)} position${Math.abs(change) > 1 ? "s" : ""} by reranking (was #${originalRank})`,
    };
  } else {
    // No change
    return {
      Icon: Minus,
      text: "=",
      className:
        "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
      tooltip: `Rank unchanged by reranking (stayed at #${originalRank})`,
    };
  }
}

export function ChunksPanel({ chunks, highlightedChunkId }: ChunksPanelProps) {
  const [openChunks, setOpenChunks] = useState<Set<string>>(new Set());

  const toggleChunk = (chunkId: string) => {
    const newOpen = new Set(openChunks);
    if (newOpen.has(chunkId)) {
      newOpen.delete(chunkId);
    } else {
      newOpen.add(chunkId);
    }
    setOpenChunks(newOpen);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">
          Retrieved Chunks ({chunks.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {chunks.map((chunk) => (
          <Collapsible
            key={chunk.chunk_id}
            open={openChunks.has(chunk.chunk_id)}
            onOpenChange={() => toggleChunk(chunk.chunk_id)}
          >
            <div
              id={`chunk-${chunk.chunk_id}`}
              className={`rounded-lg border p-3 transition-colors ${
                highlightedChunkId === chunk.chunk_id
                  ? "border-primary bg-primary/5"
                  : ""
              }`}
            >
              <CollapsibleTrigger className="flex w-full flex-col gap-2 text-left">
                <div className="flex items-start gap-2">
                  {openChunks.has(chunk.chunk_id) ? (
                    <ChevronDown className="mt-0.5 h-4 w-4 shrink-0" />
                  ) : (
                    <ChevronRight className="mt-0.5 h-4 w-4 shrink-0" />
                  )}
                  <span className="shrink-0 font-medium">#{chunk.rank}</span>
                  <span className="text-muted-foreground text-sm wrap-break-word">
                    {chunk.paper_title}
                  </span>
                </div>
                <div className="ml-6 flex flex-wrap items-center gap-1.5">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" className="cursor-help">
                        sim: {chunk.similarity_score.toFixed(3)}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs">
                      <p className="font-medium">Similarity Score</p>
                      <p className="text-muted-foreground text-xs">
                        Cosine similarity between query and chunk embeddings.
                        Range: 0–1, higher = more similar.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                  {chunk.rerank_score != null && (
                    <>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Badge variant="secondary" className="cursor-help">
                            rerank: {chunk.rerank_score.toFixed(3)}
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-xs">
                          <p className="font-medium">Rerank Score</p>
                          <p className="text-muted-foreground text-xs">
                            Cross-encoder relevance score. Analyzes query+chunk
                            together for more accurate ranking. Higher = more
                            relevant.
                          </p>
                        </TooltipContent>
                      </Tooltip>
                      {(() => {
                        const rankChange = getRankChangeDisplay(
                          chunk.original_rank,
                          chunk.rank,
                          chunk.rerank_score
                        );
                        if (!rankChange) return null;
                        const { Icon, text, className, tooltip } = rankChange;
                        return (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge
                                className={`cursor-help border-0 ${className}`}
                              >
                                <Icon className="mr-0.5 h-3 w-3" />
                                {text}
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent className="max-w-xs">
                              <p className="font-medium">Rank Change</p>
                              <p className="text-muted-foreground text-xs">
                                {tooltip}
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        );
                      })()}
                    </>
                  )}
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-3">
                <p className="text-muted-foreground text-sm whitespace-pre-wrap">
                  {chunk.content}
                </p>
              </CollapsibleContent>
            </div>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  );
}
