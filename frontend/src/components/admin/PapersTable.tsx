"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Trash2, Loader2 } from "lucide-react";
import {
  useListPapersPapersGet,
  useDeletePaperPapersPaperIdDelete,
} from "@/api/queries/papers/papers";

export function PapersTable() {
  const { data, isLoading, error, refetch } = useListPapersPapersGet();
  const deleteMutation = useDeletePaperPapersPaperIdDelete();
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleDelete = async (paperId: string) => {
    try {
      await deleteMutation.mutateAsync({ paperId });
      setConfirmDelete(null);
      refetch();
    } catch (err) {
      console.error("Failed to delete:", err);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Papers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Papers</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-red-500">Failed to load papers</div>
        </CardContent>
      </Card>
    );
  }

  const papers = data?.data?.papers ?? [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Papers ({papers.length})</CardTitle>
      </CardHeader>
      <CardContent>
        {papers.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No papers ingested yet. Use the form above to add papers.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Title</TableHead>
                <TableHead className="w-28">arXiv ID</TableHead>
                <TableHead className="w-20 text-right">Chunks</TableHead>
                <TableHead className="w-24">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {papers.map((paper) => (
                <TableRow key={paper.paper_id}>
                  <TableCell className="font-medium">
                    <span className="line-clamp-1">
                      {paper.title}
                    </span>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    {paper.arxiv_id}
                  </TableCell>
                  <TableCell className="text-right">
                    {paper.chunk_count}
                  </TableCell>
                  <TableCell>
                    {confirmDelete === paper.paper_id ? (
                      <div className="flex gap-1">
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => handleDelete(paper.paper_id)}
                          disabled={deleteMutation.isPending}
                        >
                          {deleteMutation.isPending ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            "Yes"
                          )}
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => setConfirmDelete(null)}
                        >
                          No
                        </Button>
                      </div>
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setConfirmDelete(paper.paper_id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
