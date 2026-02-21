"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ChevronDown,
  ChevronRight,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import { useState } from "react";
import type { FaithfulnessResult } from "@/api/model";

interface FaithfulnessReportProps {
  faithfulness: FaithfulnessResult;
}

const verdictConfig: Record<
  string,
  {
    icon: typeof CheckCircle;
    color: string;
    bg: string;
    badge: string;
  }
> = {
  supported: {
    icon: CheckCircle,
    color: "text-green-600",
    bg: "bg-green-50",
    badge: "bg-green-100 text-green-800",
  },
  unsupported: {
    icon: XCircle,
    color: "text-red-600",
    bg: "bg-red-50",
    badge: "bg-red-100 text-red-800",
  },
  partial: {
    icon: AlertCircle,
    color: "text-yellow-600",
    bg: "bg-yellow-50",
    badge: "bg-yellow-100 text-yellow-800",
  },
};

export function FaithfulnessReport({ faithfulness }: FaithfulnessReportProps) {
  const [isOpen, setIsOpen] = useState(false);

  const scorePercent = Math.round(faithfulness.score * 100);
  const scoreColor =
    scorePercent >= 80
      ? "text-green-600"
      : scorePercent >= 60
        ? "text-yellow-600"
        : "text-red-600";

  const supportedCount = faithfulness.claims.filter(
    (c) => c.verdict === "supported"
  ).length;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Faithfulness Report</CardTitle>
          <Badge variant="outline" className={scoreColor}>
            {scorePercent}%
          </Badge>
        </div>
        <p className="text-muted-foreground text-sm">
          {supportedCount} of {faithfulness.claims.length} claims supported
        </p>
      </CardHeader>
      <CardContent>
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <CollapsibleTrigger className="text-muted-foreground hover:text-foreground flex items-center gap-2 text-sm">
            {isOpen ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
            {isOpen ? "Hide" : "Show"} claim details
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4 space-y-3">
            {faithfulness.claims.map((claim, index) => {
              const config =
                verdictConfig[claim.verdict] || verdictConfig.partial;
              const Icon = config.icon;

              return (
                <Alert key={index} className={config.bg}>
                  <Icon className={`h-4 w-4 ${config.color}`} />
                  <AlertDescription>
                    <div className="space-y-2">
                      <div className="flex items-start justify-between gap-2">
                        <p className="text-sm font-medium">{claim.claim}</p>
                        <Badge className={config.badge}>{claim.verdict}</Badge>
                      </div>
                      <p className="text-muted-foreground text-sm">
                        {claim.reasoning}
                      </p>
                    </div>
                  </AlertDescription>
                </Alert>
              );
            })}
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
}
