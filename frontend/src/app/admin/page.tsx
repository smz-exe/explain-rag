"use client";

import { StatsCards } from "@/components/admin/StatsCards";
import { PapersTable } from "@/components/admin/PapersTable";
import { QueriesTable } from "@/components/admin/QueriesTable";
import { IngestForm } from "@/components/admin/IngestForm";
import { RecomputeButton } from "@/components/admin/RecomputeButton";

export default function AdminPage() {
  return (
    <div className="container mx-auto max-w-6xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-bold">Admin Dashboard</h1>

      <div className="space-y-6">
        {/* System Metrics */}
        <StatsCards />

        {/* Ingest Form and Recompute Button */}
        <div className="grid gap-6 md:grid-cols-2">
          <IngestForm />
          <RecomputeButton />
        </div>

        {/* Papers and Queries Tables */}
        <div className="grid gap-6 lg:grid-cols-2">
          <PapersTable />
          <QueriesTable />
        </div>
      </div>
    </div>
  );
}
