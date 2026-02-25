"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Plus, Search } from "lucide-react";
import { useIngestPapersIngestPost } from "@/api/queries/ingestion/ingestion";
import { searchPapersPapersSearchGet } from "@/api/queries/papers/papers";
import { useQueryClient } from "@tanstack/react-query";

interface PaperSearchResult {
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  url: string;
}

export function IngestForm() {
  const [arxivId, setArxivId] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PaperSearchResult[]>([]);
  const [selectedPapers, setSelectedPapers] = useState<string[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState("");

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

      queryClient.invalidateQueries({ queryKey: ["/papers"] });
      queryClient.invalidateQueries({ queryKey: ["/stats"] });

      if ("ingested" in result.data) {
        const ingested = result.data.ingested;
        if (ingested.length > 0) {
          setSuccessMessage(`Ingested: ${ingested[0].title}`);
        }
      }
    } catch {
      // Error is handled by ingestMutation.isError state
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim() || searchQuery.length < 2) return;

    setIsSearching(true);
    setSearchError("");
    setSearchResults([]);
    setSelectedPapers([]);

    try {
      const response = await searchPapersPapersSearchGet({
        query: searchQuery,
        max_results: 5,
      });

      if (response.status === 200) {
        setSearchResults(response.data.papers);
      } else {
        setSearchError("Search failed. Please try again.");
      }
    } catch (err) {
      if (err instanceof Error && err.message.includes("401")) {
        setSearchError("Authentication required. Please log in again.");
      } else {
        setSearchError("Failed to search arXiv. Please try again.");
      }
    } finally {
      setIsSearching(false);
    }
  };

  const handleTogglePaper = (arxivId: string) => {
    setSelectedPapers((prev) =>
      prev.includes(arxivId)
        ? prev.filter((id) => id !== arxivId)
        : [...prev, arxivId]
    );
  };

  const handleIngestSelected = async () => {
    if (selectedPapers.length === 0) return;

    setSuccessMessage("");

    try {
      const result = await ingestMutation.mutateAsync({
        data: { arxiv_ids: selectedPapers },
      });

      queryClient.invalidateQueries({ queryKey: ["/papers"] });
      queryClient.invalidateQueries({ queryKey: ["/stats"] });

      if ("ingested" in result.data) {
        const ingested = result.data.ingested;
        if (ingested.length > 0) {
          setSuccessMessage(
            `Ingested ${ingested.length} paper(s): ${ingested.map((p) => p.title).join(", ")}`
          );
        }
      }

      setSearchResults([]);
      setSelectedPapers([]);
      setSearchQuery("");
    } catch {
      // Error is handled by ingestMutation.isError state
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ingest Papers</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="arxiv-id">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="arxiv-id">By arXiv ID</TabsTrigger>
            <TabsTrigger value="search">Search arXiv</TabsTrigger>
          </TabsList>

          <TabsContent value="arxiv-id" className="mt-4">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="arxiv-id">arXiv ID</Label>
                <Input
                  id="arxiv-id"
                  name="arxiv-id"
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
          </TabsContent>

          <TabsContent value="search" className="mt-4">
            <div className="space-y-4">
              <div className="flex gap-2">
                <Input
                  id="arxiv-search"
                  name="arxiv-search"
                  placeholder="Search arXiv (e.g., transformer attention)"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleSearch();
                    }
                  }}
                  disabled={isSearching}
                />
                <Button
                  type="button"
                  onClick={handleSearch}
                  disabled={isSearching || searchQuery.length < 2}
                >
                  {isSearching ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Search className="h-4 w-4" />
                  )}
                </Button>
              </div>

              {searchError && (
                <div className="rounded bg-red-50 p-2 text-sm text-red-600 dark:bg-red-900/20">
                  {searchError}
                </div>
              )}

              {searchResults.length > 0 && (
                <div className="space-y-3">
                  <Label>Select papers to ingest:</Label>
                  {searchResults.map((paper) => (
                    <div
                      key={paper.arxiv_id}
                      className="flex items-start gap-3 rounded border p-3"
                    >
                      <Checkbox
                        id={paper.arxiv_id}
                        checked={selectedPapers.includes(paper.arxiv_id)}
                        onCheckedChange={() =>
                          handleTogglePaper(paper.arxiv_id)
                        }
                      />
                      <div className="flex-1 space-y-1">
                        <label
                          htmlFor={paper.arxiv_id}
                          className="cursor-pointer text-sm leading-none font-medium"
                        >
                          {paper.title}
                        </label>
                        <p className="text-muted-foreground text-xs">
                          {paper.authors.slice(0, 3).join(", ")}
                          {paper.authors.length > 3 &&
                            ` +${paper.authors.length - 3} more`}
                        </p>
                        <p className="text-muted-foreground mt-1 text-xs">
                          {paper.abstract}
                        </p>
                      </div>
                    </div>
                  ))}

                  {successMessage && (
                    <div className="rounded bg-green-50 p-2 text-sm text-green-600 dark:bg-green-900/20">
                      {successMessage}
                    </div>
                  )}

                  {ingestMutation.isError && (
                    <div className="rounded bg-red-50 p-2 text-sm text-red-600 dark:bg-red-900/20">
                      Ingestion failed. Please try again.
                    </div>
                  )}

                  <Button
                    onClick={handleIngestSelected}
                    disabled={
                      selectedPapers.length === 0 || ingestMutation.isPending
                    }
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
                        Ingest {selectedPapers.length} Selected
                      </>
                    )}
                  </Button>
                </div>
              )}

              {!isSearching && searchResults.length === 0 && searchQuery && (
                <p className="text-muted-foreground text-center text-sm">
                  No results. Try a different search term.
                </p>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
