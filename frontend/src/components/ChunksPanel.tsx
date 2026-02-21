"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
              <CollapsibleTrigger className="flex w-full items-center justify-between text-left">
                <div className="flex items-center gap-2">
                  {openChunks.has(chunk.chunk_id) ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                  <span className="font-medium">#{chunk.rank}</span>
                  <span className="text-muted-foreground max-w-50 truncate text-sm">
                    {chunk.paper_title}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline">
                    sim: {chunk.similarity_score.toFixed(3)}
                  </Badge>
                  {chunk.rerank_score != null && (
                    <Badge variant="secondary">
                      rerank: {chunk.rerank_score.toFixed(3)}
                    </Badge>
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
