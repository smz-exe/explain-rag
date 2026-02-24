"use client";

import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { getQueryExportUrl } from "@/api/custom-fetch";

interface ExportButtonProps {
  queryId: string;
}

export function ExportButton({ queryId }: ExportButtonProps) {
  const handleExport = () => {
    window.open(getQueryExportUrl(queryId), "_blank");
  };

  return (
    <Button variant="outline" size="sm" onClick={handleExport}>
      <Download className="mr-2 h-4 w-4" />
      Export
    </Button>
  );
}
