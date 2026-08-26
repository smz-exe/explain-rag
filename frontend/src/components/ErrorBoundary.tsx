"use client";

import { Component, ReactNode } from "react";
import { ErrorFallback } from "@/components/ErrorFallback";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <ErrorFallback
            message={this.state.error?.message}
            onRetry={() => this.setState({ hasError: false, error: undefined })}
          />
        )
      );
    }
    return this.props.children;
  }
}
