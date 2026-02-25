"use client";

import { Suspense, Component, type ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { cn } from "@/lib/utils";
import { colors, visualization } from "@/lib/design-tokens";
import { PaperPoints } from "./PaperPoints";
import { QueryPoint } from "./QueryPoint";
import { ConnectionLines } from "./ConnectionLines";
import type { Paper } from "./PapersPanel";

/**
 * Error boundary for Three.js Canvas.
 * Catches WebGL errors, shader compilation failures, etc.
 */
interface CanvasErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface CanvasErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class CanvasErrorBoundary extends Component<
  CanvasErrorBoundaryProps,
  CanvasErrorBoundaryState
> {
  constructor(props: CanvasErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): CanvasErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="flex h-full items-center justify-center">
            <div className="text-atlas-text-secondary text-center">
              <CanvasErrorIcon />
              <p className="mt-3 text-sm font-medium">
                Unable to load 3D visualization
              </p>
              <p className="mt-1 text-xs opacity-60">
                Your browser may not support WebGL
              </p>
            </div>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

function CanvasErrorIcon() {
  return (
    <svg
      className="text-atlas-border mx-auto h-12 w-12"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
      />
    </svg>
  );
}

export interface QueryCoords {
  coords: [number, number, number];
  retrievedPaperIds: string[];
}

interface EmbeddingSpaceProps {
  papers: Paper[];
  /** Currently selected paper ID (from sidebar click) */
  selectedPaperId?: string;
  /** Query coordinates after search */
  queryCoords?: QueryCoords;
  /** Loading state */
  isLoading?: boolean;
  /** Callback when a paper point is clicked */
  onPaperClick?: (paperId: string) => void;
  /** Callback when hovering over a paper */
  onPaperHover?: (paperId: string | null) => void;
  className?: string;
}

/**
 * 3D embedding space visualization using Three.js.
 * Shows papers as spheres positioned by their UMAP coordinates,
 * with query point and connection lines after search.
 */
export function EmbeddingSpace({
  papers,
  selectedPaperId,
  queryCoords,
  isLoading,
  onPaperClick,
  onPaperHover,
  className,
}: EmbeddingSpaceProps) {
  // Loading state
  if (isLoading) {
    return (
      <div className={cn("flex h-full items-center justify-center", className)}>
        <div className="text-atlas-text-secondary text-center">
          <LoadingSpinner />
          <p className="mt-3 text-sm">Loading embedding space...</p>
        </div>
      </div>
    );
  }

  // Empty state
  if (papers.length === 0) {
    return (
      <div className={cn("flex h-full items-center justify-center", className)}>
        <div className="text-atlas-text-secondary text-center">
          <EmptyStateIcon />
          <p className="mt-3 text-sm font-medium">
            No papers in the collection yet
          </p>
          <p className="mt-1 text-xs opacity-60">
            Check back soon to explore the research landscape
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("h-full w-full", className)}>
      <CanvasErrorBoundary>
        <Canvas>
          <Suspense fallback={null}>
            <PerspectiveCamera
              makeDefault
              position={visualization.camera.defaultPosition}
              fov={visualization.camera.fov}
              near={visualization.camera.near}
              far={visualization.camera.far}
            />

            <OrbitControls
              enableDamping
              dampingFactor={0.05}
              minDistance={2}
              maxDistance={20}
            />

            {/* Ambient light for base visibility */}
            <ambientLight intensity={0.6} />

            {/* Directional light for depth */}
            <directionalLight position={[10, 10, 5]} intensity={0.8} />

            {/* Grid helper for spatial reference */}
            <gridHelper
              args={[10, 10, colors.border, colors.border]}
              position={[0, -2, 0]}
            />

            {/* Paper points */}
            <PaperPoints
              papers={papers}
              selectedPaperId={selectedPaperId}
              highlightedPaperIds={queryCoords?.retrievedPaperIds}
              onPaperClick={onPaperClick}
              onPaperHover={onPaperHover}
            />

            {/* Query point (appears after search) */}
            {queryCoords && (
              <>
                <QueryPoint coords={queryCoords.coords} />
                <ConnectionLines
                  queryCoords={queryCoords.coords}
                  papers={papers}
                  connectedPaperIds={queryCoords.retrievedPaperIds}
                />
              </>
            )}
          </Suspense>
        </Canvas>
      </CanvasErrorBoundary>

      {/* Controls hint */}
      <div className="text-atlas-text-secondary pointer-events-none absolute bottom-4 left-4 text-xs opacity-50">
        Drag to rotate · Scroll to zoom
      </div>
    </div>
  );
}

function LoadingSpinner() {
  return (
    <svg
      className="text-atlas-text-secondary mx-auto h-8 w-8 animate-spin"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

function EmptyStateIcon() {
  return (
    <svg
      className="text-atlas-border mx-auto h-16 w-16"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
      />
    </svg>
  );
}
