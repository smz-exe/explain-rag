"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle } from "lucide-react";

interface ErrorFallbackProps {
  /** Message to show; a generic line is used when absent. */
  message?: string;
  /** Invoked by the retry button. */
  onRetry: () => void;
}

/**
 * Shared presentation for an unrecoverable render error.
 *
 * Used by both the React class boundary and Next's route-segment error page,
 * so a crash looks the same wherever it is caught.
 */
export function ErrorFallback({ message, onRetry }: ErrorFallbackProps) {
  return (
    <Card className="border-destructive" role="alert">
      <CardHeader>
        <CardTitle className="text-destructive flex items-center gap-2">
          <AlertTriangle className="h-5 w-5" />
          Something went wrong
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-muted-foreground mb-4">
          {message || "An unexpected error occurred"}
        </p>
        <Button onClick={onRetry}>Try Again</Button>
      </CardContent>
    </Card>
  );
}
