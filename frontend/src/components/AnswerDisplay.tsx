"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ExportButton } from "@/components/ExportButton";
import type { Citation } from "@/api/model";

interface AnswerDisplayProps {
  queryId?: string;
  question: string;
  answer: string;
  citations: Citation[];
  onCitationClick?: (chunkId: string) => void;
}

export function AnswerDisplay({
  queryId,
  question,
  answer,
  citations,
  onCitationClick,
}: AnswerDisplayProps) {
  // Parse answer to find citation markers like [1], [2], etc.
  const renderAnswerWithCitations = () => {
    const parts = answer.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const citationIndex = parseInt(match[1], 10) - 1;
        const citation = citations[citationIndex];
        if (citation) {
          return (
            <Badge
              key={index}
              variant="secondary"
              className="hover:bg-primary hover:text-primary-foreground mx-0.5 cursor-pointer"
              onClick={() => {
                if (onCitationClick && citation.chunk_ids[0]) {
                  onCitationClick(citation.chunk_ids[0]);
                }
              }}
            >
              {part}
            </Badge>
          );
        }
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Answer</CardTitle>
          {queryId && <ExportButton queryId={queryId} />}
        </div>
        <p className="text-muted-foreground text-sm">{question}</p>
      </CardHeader>
      <CardContent>
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {renderAnswerWithCitations()}
        </div>
      </CardContent>
    </Card>
  );
}
