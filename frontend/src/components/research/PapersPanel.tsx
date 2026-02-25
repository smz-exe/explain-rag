"use client";

import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { getClusterColor } from "@/lib/design-tokens";

export interface Paper {
  paper_id: string;
  arxiv_id: string;
  title: string;
  coords: [number, number, number];
  cluster_id: number;
  chunk_count: number;
}

export interface Cluster {
  id: number;
  label: string;
  paper_ids: string[];
}

interface PapersPanelProps {
  papers: Paper[];
  clusters: Cluster[];
  /** Currently selected paper ID */
  selectedPaperId?: string;
  /** Callback when a paper is clicked */
  onPaperSelect?: (paperId: string) => void;
  /** Callback when "Ask about this paper" is clicked */
  onAskAboutPaper?: (paper: Paper) => void;
  /** Loading state */
  isLoading?: boolean;
  className?: string;
}

/**
 * Papers panel with collapsible cluster sections.
 * Shows papers grouped by semantic clusters with chunk counts.
 */
export function PapersPanel({
  papers,
  clusters,
  selectedPaperId,
  onPaperSelect,
  onAskAboutPaper,
  isLoading,
  className,
}: PapersPanelProps) {
  // Create a stable key from cluster IDs to detect when clusters change
  const clusterKey = useMemo(
    () => clusters.map((c) => c.id).join(","),
    [clusters]
  );

  const [expandedClusters, setExpandedClusters] = useState<Set<number>>(
    () => new Set(clusters.map((c) => c.id))
  );
  const [prevClusterKey, setPrevClusterKey] = useState(clusterKey);

  // Sync expanded clusters when clusters prop changes (React-recommended pattern)
  if (clusterKey !== prevClusterKey) {
    setPrevClusterKey(clusterKey);
    setExpandedClusters(new Set(clusters.map((c) => c.id)));
  }

  const toggleCluster = (clusterId: number) => {
    setExpandedClusters((prev) => {
      const next = new Set(prev);
      if (next.has(clusterId)) {
        next.delete(clusterId);
      } else {
        next.add(clusterId);
      }
      return next;
    });
  };

  // Group papers by cluster
  const papersByCluster = new Map<number, Paper[]>();
  const unclusteredPapers: Paper[] = [];

  for (const paper of papers) {
    if (paper.cluster_id < 0) {
      unclusteredPapers.push(paper);
    } else {
      const existing = papersByCluster.get(paper.cluster_id) ?? [];
      papersByCluster.set(paper.cluster_id, [...existing, paper]);
    }
  }

  if (isLoading) {
    return (
      <div className={cn("p-4", className)}>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="animate-pulse">
              <div className="bg-atlas-border mb-2 h-4 w-24 rounded" />
              <div className="space-y-2 pl-3">
                <div className="bg-atlas-border h-12 rounded" />
                <div className="bg-atlas-border h-12 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (papers.length === 0) {
    return (
      <div className={cn("p-4", className)}>
        <div className="text-atlas-text-secondary py-8 text-center">
          <p className="mb-1 text-sm">No papers in the collection yet</p>
          <p className="text-xs opacity-60">
            Papers will appear here as the reading list grows
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("p-2", className)}>
      <div className="mb-2 px-2">
        <p className="text-atlas-text-secondary text-xs">
          {papers.length} paper{papers.length !== 1 ? "s" : ""} in{" "}
          {clusters.length} cluster{clusters.length !== 1 ? "s" : ""}
        </p>
      </div>

      <div className="space-y-1">
        {clusters.map((cluster) => {
          const clusterPapers = papersByCluster.get(cluster.id) ?? [];
          if (clusterPapers.length === 0) return null;

          const isExpanded = expandedClusters.has(cluster.id);
          const clusterColor = getClusterColor(cluster.id);

          return (
            <Collapsible
              key={cluster.id}
              open={isExpanded}
              onOpenChange={() => toggleCluster(cluster.id)}
            >
              <CollapsibleTrigger asChild>
                <button
                  className="hover:bg-atlas-background flex w-full items-center gap-2 rounded px-2 py-1.5 text-left transition-colors"
                  type="button"
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? "Collapse" : "Expand"} cluster: ${cluster.label}`}
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: clusterColor }}
                    aria-hidden="true"
                  />
                  <span className="text-atlas-text-primary flex-1 truncate text-sm font-medium">
                    {cluster.label}
                  </span>
                  <span className="text-atlas-text-secondary text-xs">
                    {clusterPapers.length}
                  </span>
                  <ChevronIcon isOpen={isExpanded} />
                </button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="space-y-0.5 py-1 pl-5">
                  {clusterPapers.map((paper) => (
                    <PaperItem
                      key={paper.paper_id}
                      paper={paper}
                      isSelected={selectedPaperId === paper.paper_id}
                      onSelect={onPaperSelect}
                      onAsk={onAskAboutPaper}
                    />
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          );
        })}

        {/* Unclustered papers */}
        {unclusteredPapers.length > 0 && (
          <Collapsible
            open={expandedClusters.has(-1)}
            onOpenChange={() => toggleCluster(-1)}
          >
            <CollapsibleTrigger asChild>
              <button
                className="hover:bg-atlas-background flex w-full items-center gap-2 rounded px-2 py-1.5 text-left transition-colors"
                type="button"
                aria-expanded={expandedClusters.has(-1)}
                aria-label={`${expandedClusters.has(-1) ? "Collapse" : "Expand"} unclustered papers`}
              >
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: getClusterColor(-1) }}
                  aria-hidden="true"
                />
                <span className="text-atlas-text-secondary flex-1 truncate text-sm">
                  Unclustered
                </span>
                <span className="text-atlas-text-secondary text-xs">
                  {unclusteredPapers.length}
                </span>
                <ChevronIcon isOpen={expandedClusters.has(-1)} />
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="space-y-0.5 py-1 pl-5">
                {unclusteredPapers.map((paper) => (
                  <PaperItem
                    key={paper.paper_id}
                    paper={paper}
                    isSelected={selectedPaperId === paper.paper_id}
                    onSelect={onPaperSelect}
                    onAsk={onAskAboutPaper}
                  />
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}

interface PaperItemProps {
  paper: Paper;
  isSelected?: boolean;
  onSelect?: (paperId: string) => void;
  onAsk?: (paper: Paper) => void;
}

function PaperItem({ paper, isSelected, onSelect, onAsk }: PaperItemProps) {
  return (
    <div
      className={cn(
        "group rounded px-2 py-1.5 transition-colors",
        isSelected
          ? "bg-atlas-accent/5 border-atlas-accent border-l-2"
          : "hover:bg-atlas-background"
      )}
    >
      <button
        type="button"
        className="w-full text-left"
        onClick={() => onSelect?.(paper.paper_id)}
      >
        <p
          className={cn(
            "line-clamp-2 text-sm",
            isSelected
              ? "text-atlas-text-primary font-medium"
              : "text-atlas-text-primary"
          )}
        >
          {paper.title}
        </p>
        <div className="mt-0.5 flex items-center gap-2">
          <span className="text-atlas-text-secondary font-mono text-xs">
            {paper.arxiv_id}
          </span>
          <span className="text-atlas-text-secondary text-xs opacity-60">
            {paper.chunk_count} chunks
          </span>
        </div>
      </button>
      {onAsk && (
        <Button
          variant="ghost"
          size="sm"
          className="text-atlas-text-secondary hover:text-atlas-accent mt-1 h-6 px-2 text-xs opacity-0 transition-opacity group-hover:opacity-100"
          onClick={(e) => {
            e.stopPropagation();
            onAsk(paper);
          }}
        >
          Ask about this paper
        </Button>
      )}
    </div>
  );
}

function ChevronIcon({ isOpen }: { isOpen: boolean }) {
  return (
    <svg
      className={cn(
        "text-atlas-text-secondary h-4 w-4 transition-transform",
        isOpen && "rotate-90"
      )}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  );
}
