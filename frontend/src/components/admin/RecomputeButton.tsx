"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, RefreshCw, CheckCircle } from "lucide-react";
import { useRecomputeEmbeddingsAdminPapersRecomputeEmbeddingsPost } from "@/api/queries/admin/admin";
import { useQueryClient } from "@tanstack/react-query";
import { APIError } from "@/api/custom-fetch";
import type { RecomputeResponse } from "@/api/model";

export function RecomputeButton() {
  const [result, setResult] = useState<RecomputeResponse | null>(null);
  const queryClient = useQueryClient();

  const { mutate, isPending, error } =
    useRecomputeEmbeddingsAdminPapersRecomputeEmbeddingsPost({
      mutation: {
        onSuccess: (response) => {
          if (response.status === 200) {
            setResult(response.data);
            // Invalidate embeddings query to refresh the visualization
            queryClient.invalidateQueries({ queryKey: ["/papers/embeddings"] });
            queryClient.invalidateQueries({ queryKey: ["/papers/clusters"] });
          }
        },
      },
    });

  const handleRecompute = () => {
    setResult(null);
    mutate();
  };

  const getErrorMessage = (): string | null => {
    if (!error) return null;
    if (error instanceof APIError) {
      if (error.status === 401)
        return "Authentication required. Please log in again.";
      return error.detail || error.message;
    }
    return "Failed to recompute embeddings. Please try again.";
  };

  const errorMessage = getErrorMessage();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Embedding Visualization</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-muted-foreground text-sm">
          Recompute 3D coordinates and clusters for the paper visualization.
          This runs UMAP dimensionality reduction and HDBSCAN clustering.
        </p>

        {!isPending && result && !errorMessage && (
          <div className="flex items-center gap-2 rounded bg-green-50 p-3 text-sm text-green-600 dark:bg-green-900/20">
            <CheckCircle className="h-4 w-4 flex-shrink-0" />
            <span>
              Processed {result.papers_processed} papers into{" "}
              {result.clusters_found} clusters ({result.time_ms.toFixed(0)}ms)
            </span>
          </div>
        )}

        {!isPending && errorMessage && (
          <div className="rounded bg-red-50 p-3 text-sm text-red-600 dark:bg-red-900/20">
            {errorMessage}
          </div>
        )}

        <Button
          onClick={handleRecompute}
          disabled={isPending}
          className="w-full"
        >
          {isPending ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Recomputing...
            </>
          ) : (
            <>
              <RefreshCw className="mr-2 h-4 w-4" />
              Recompute Embeddings
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
}
