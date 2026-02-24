"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Plus } from "lucide-react";
import { useIngestPapersIngestPost } from "@/api/queries/ingestion/ingestion";
import { useQueryClient } from "@tanstack/react-query";

export function IngestForm() {
  const [arxivId, setArxivId] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const ingestMutation = useIngestPapersIngestPost();
  const queryClient = useQueryClient();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!arxivId.trim()) return;

    setSuccessMessage("");

    try {
      const result = await ingestMutation.mutateAsync({
        data: { arxiv_ids: [arxivId.trim()] },
      });
      setArxivId("");

      // Invalidate papers and stats queries to refresh the data
      queryClient.invalidateQueries({ queryKey: ["/papers"] });
      queryClient.invalidateQueries({ queryKey: ["/stats"] });

      // Check if response is successful (has ingested property)
      if ("ingested" in result.data) {
        const ingested = result.data.ingested;
        if (ingested.length > 0) {
          setSuccessMessage(`Ingested: ${ingested[0].title}`);
        }
      }
    } catch (err) {
      console.error("Ingestion failed:", err);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ingest Paper</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="arxiv-id">arXiv ID</Label>
            <Input
              id="arxiv-id"
              placeholder="e.g., 1706.03762"
              value={arxivId}
              onChange={(e) => setArxivId(e.target.value)}
              disabled={ingestMutation.isPending}
            />
          </div>

          {successMessage && (
            <div className="rounded bg-green-50 p-2 text-sm text-green-600 dark:bg-green-900/20">
              {successMessage}
            </div>
          )}

          {ingestMutation.isError && (
            <div className="rounded bg-red-50 p-2 text-sm text-red-600 dark:bg-red-900/20">
              Ingestion failed. Please check the arXiv ID.
            </div>
          )}

          <Button
            type="submit"
            disabled={!arxivId.trim() || ingestMutation.isPending}
            className="w-full"
          >
            {ingestMutation.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Ingesting...
              </>
            ) : (
              <>
                <Plus className="mr-2 h-4 w-4" />
                Ingest Paper
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
