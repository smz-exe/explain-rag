"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { useListQueriesQueryListGet } from "@/api/queries/query/query";
import { EvaluateButton } from "./EvaluateButton";

export function QueriesTable() {
  const { data, isLoading, error } = useListQueriesQueryListGet({ limit: 10 });

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return (
        date.toLocaleDateString(undefined, {
          month: "short",
          day: "numeric",
        }) +
        " " +
        date.toLocaleTimeString(undefined, {
          hour: "2-digit",
          minute: "2-digit",
        })
      );
    } catch {
      return isoString.slice(0, 16);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Recent Queries</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
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
          <CardTitle>Recent Queries</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-red-500">Failed to load queries</div>
        </CardContent>
      </Card>
    );
  }

  const queries = data?.data && "queries" in data.data ? data.data.queries : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Queries</CardTitle>
      </CardHeader>
      <CardContent>
        {queries.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No queries yet. Use the main app to ask questions.
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Question</TableHead>
                <TableHead className="w-32">Time</TableHead>
                <TableHead className="w-24">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {queries.map((query) => (
                <TableRow key={query.query_id}>
                  <TableCell>
                    <div className="line-clamp-1 font-medium">
                      {query.question}
                    </div>
                    <div className="text-muted-foreground line-clamp-1 text-xs">
                      {query.answer_preview}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-sm">
                    {formatDate(query.created_at)}
                  </TableCell>
                  <TableCell>
                    <EvaluateButton queryId={query.query_id} />
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
