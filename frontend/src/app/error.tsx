"use client";

import { ErrorFallback } from "@/components/ErrorFallback";

/**
 * Route-segment error page.
 *
 * Without this file a render error anywhere in the tree falls through to
 * Next's default screen, which is a blank page in production.
 */
export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="mx-auto max-w-xl p-6">
      <ErrorFallback message={error.message} onRetry={reset} />
    </div>
  );
}
