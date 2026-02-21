"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

interface QueryInputProps {
  onSubmit: (query: {
    question: string;
    top_k: number;
    enable_reranking: boolean;
  }) => void;
  isLoading?: boolean;
}

export function QueryInput({ onSubmit, isLoading = false }: QueryInputProps) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(10);
  const [enableReranking, setEnableReranking] = useState(false);

  const handleSubmit = (e: React.SyntheticEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (question.trim()) {
      onSubmit({
        question: question.trim(),
        top_k: topK,
        enable_reranking: enableReranking,
      });
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="question">Question</Label>
        <Textarea
          id="question"
          placeholder="Ask a question about the ingested papers..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          disabled={isLoading}
        />
      </div>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="max-w-xs flex-1 space-y-2">
          <Label htmlFor="top-k">Top K: {topK}</Label>
          <Slider
            id="top-k"
            min={1}
            max={50}
            step={1}
            value={[topK]}
            onValueChange={([value]) => setTopK(value)}
            disabled={isLoading}
          />
        </div>

        <div className="flex items-center gap-2">
          <Switch
            id="reranking"
            checked={enableReranking}
            onCheckedChange={setEnableReranking}
            disabled={isLoading}
          />
          <Label htmlFor="reranking">Enable Reranking</Label>
        </div>

        <Button
          type="submit"
          disabled={isLoading || !question.trim()}
          className="w-full sm:w-auto"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Querying...
            </>
          ) : (
            "Ask"
          )}
        </Button>
      </div>
    </form>
  );
}
