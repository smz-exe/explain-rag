"use client";

import {
  useState,
  useEffect,
  useRef,
  useImperativeHandle,
  forwardRef,
} from "react";
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
  /** Prefilled question text */
  defaultQuestion?: string;
}

export interface QueryInputHandle {
  focus: () => void;
}

export const QueryInput = forwardRef<QueryInputHandle, QueryInputProps>(
  function QueryInput(
    { onSubmit, isLoading = false, defaultQuestion = "" },
    ref
  ) {
    const [question, setQuestion] = useState(defaultQuestion);
    const [topK, setTopK] = useState(10);
    const [enableReranking, setEnableReranking] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Expose focus method to parent
    useImperativeHandle(ref, () => ({
      focus: () => {
        textareaRef.current?.focus();
      },
    }));

    // Sync with defaultQuestion prop changes
    useEffect(() => {
      if (defaultQuestion) {
        setQuestion(defaultQuestion);
      }
    }, [defaultQuestion]);

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
          <div className="flex items-center justify-between">
            <Label htmlFor="question">Question</Label>
            <kbd className="text-atlas-text-secondary bg-atlas-background hidden rounded px-1.5 py-0.5 font-mono text-xs sm:inline-block">
              /
            </kbd>
          </div>
          <Textarea
            ref={textareaRef}
            id="question"
            name="question"
            placeholder="Ask a question about the ingested papers..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            disabled={isLoading}
          />
        </div>

        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="max-w-xs flex-1 space-y-2">
            <Label id="top-k-label">Top K: {topK}</Label>
            <Slider
              aria-labelledby="top-k-label"
              name="top-k"
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
              name="reranking"
              checked={enableReranking}
              onCheckedChange={setEnableReranking}
              disabled={isLoading}
              aria-labelledby="reranking-label"
            />
            <Label id="reranking-label">Enable Reranking</Label>
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
);
