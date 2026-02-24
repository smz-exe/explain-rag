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
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type { RetrievedChunk } from "@/api/model";

interface ChunksPanelProps {
  chunks: RetrievedChunk[];
  highlightedChunkId?: string;
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
              <CollapsibleTrigger className="flex w-full flex-col gap-2 text-left sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                  {openChunks.has(chunk.chunk_id) ? (
                    <ChevronDown className="h-4 w-4 shrink-0" />
                  ) : (
                    <ChevronRight className="h-4 w-4 shrink-0" />
                  )}
                  <span className="font-medium">#{chunk.rank}</span>
                  <span className="text-muted-foreground max-w-30 truncate text-sm sm:max-w-50 lg:max-w-75">
                    {chunk.paper_title}
                  </span>
                </div>
                <div className="ml-6 flex flex-wrap items-center gap-1 sm:ml-0 sm:gap-2">
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
