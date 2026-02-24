"use client";

import { StatsCards } from "@/components/admin/StatsCards";
import { PapersTable } from "@/components/admin/PapersTable";
import { QueriesTable } from "@/components/admin/QueriesTable";
import { IngestForm } from "@/components/admin/IngestForm";

export default function AdminPage() {
  return (
    <div className="container mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Admin Dashboard</h1>

      <div className="space-y-6">
        {/* System Metrics */}
        <StatsCards />

        {/* Ingest Form */}
        <div className="max-w-md">
          <IngestForm />
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
