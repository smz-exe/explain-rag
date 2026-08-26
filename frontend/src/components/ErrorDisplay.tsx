"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle, RefreshCw } from "lucide-react";
import { APIError } from "@/api/custom-fetch";

interface ErrorDisplayProps {
  error: Error;
  onRetry?: () => void;
}

export function ErrorDisplay({ error, onRetry }: ErrorDisplayProps) {
  // An APIError means the server answered, so it is never a connection
  // problem. Matching on message text misread real server errors as network
  // failures whenever the backend's detail happened to mention "connect".
  const isNetworkError = !(error instanceof APIError);

  return (
    <Alert variant="destructive">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>
        {isNetworkError ? "Connection Error" : "Request Failed"}
      </AlertTitle>
      <AlertDescription className="mt-2">
        <p className="mb-3">
          {isNetworkError
            ? "Could not connect to the backend. Please check that the server is running."
            : error.message}
        </p>
        {onRetry && (
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Try Again
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
}
