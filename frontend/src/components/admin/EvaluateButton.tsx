"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Loader2,
  BarChart2,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useEvaluateQueryEvaluationQueryQueryIdPost } from "@/api/queries/admin/admin";
import { APIError } from "@/api/custom-fetch";
import type { EvaluationResult } from "@/api/model";

interface EvaluateButtonProps {
  queryId: string;
}

function MetricBadge({ label, value }: { label: string; value: number }) {
  const percentage = Math.round(value * 100);
  const color =
    percentage >= 80
      ? "text-green-600 dark:text-green-400"
      : percentage >= 60
        ? "text-yellow-600 dark:text-yellow-400"
        : "text-red-600 dark:text-red-400";

  const Icon =
    percentage >= 80 ? CheckCircle : percentage >= 60 ? AlertCircle : XCircle;

  return (
    <div className="flex items-center justify-between rounded border p-2">
      <span className="text-muted-foreground text-sm">{label}</span>
      <div className={cn("flex items-center gap-1 font-medium", color)}>
        <Icon className="h-4 w-4" />
        {percentage}%
      </div>
    </div>
  );
}

export function EvaluateButton({ queryId }: EvaluateButtonProps) {
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<EvaluationResult | null>(null);

  const { mutate, isPending, error } =
    useEvaluateQueryEvaluationQueryQueryIdPost({
      mutation: {
        onSuccess: (response) => {
          if (response.status === 200) {
            setResult(response.data);
          }
        },
      },
    });

  const handleEvaluate = () => {
    mutate({ queryId, data: null });
  };

  const getErrorMessage = (): string | null => {
    if (!error) return null;
    if (error instanceof APIError) {
      if (error.status === 401)
        return "Authentication required. Please log in again.";
      if (error.status === 404) return "Query not found.";
      return error.detail || error.message;
    }
    return "Failed to run evaluation. Please try again.";
  };

  const errorMessage = getErrorMessage();

  return (
    <Dialog
      open={open}
      onOpenChange={(isOpen) => {
        setOpen(isOpen);
        if (!isOpen) {
          setResult(null); // Clear cache on close to ensure fresh evaluation
        }
      }}
    >
      <DialogTrigger asChild>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            if (!result) {
              handleEvaluate();
            }
          }}
          disabled={isPending}
        >
          {isPending ? (
            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
          ) : (
            <BarChart2 className="mr-1 h-3 w-3" />
          )}
          Evaluate
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>RAGAS Evaluation</DialogTitle>
          <DialogDescription>
            Quality metrics for query {queryId.slice(0, 8)}...
          </DialogDescription>
        </DialogHeader>

        {isPending && (
          <div className="flex flex-col items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
            <p className="text-muted-foreground mt-2 text-sm">
              Running RAGAS evaluation...
            </p>
            <p className="text-muted-foreground text-xs">
              This may take a moment
            </p>
          </div>
        )}

        {errorMessage && (
          <div className="rounded bg-red-50 p-4 text-sm text-red-600 dark:bg-red-900/20">
            {errorMessage}
          </div>
        )}

        {result && !isPending && (
          <div className="space-y-3">
            <MetricBadge
              label="Faithfulness"
              value={result.metrics.faithfulness}
            />
            <MetricBadge
              label="Answer Relevancy"
              value={result.metrics.answer_relevancy}
            />
            <MetricBadge
              label="Context Precision"
              value={result.metrics.context_precision}
            />
            {result.metrics.context_recall > 0 && (
              <MetricBadge
                label="Context Recall"
                value={result.metrics.context_recall}
              />
            )}

            <div className="text-muted-foreground border-t pt-2 text-xs">
              Evaluated in {result.evaluation_time_ms.toFixed(0)}ms
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
