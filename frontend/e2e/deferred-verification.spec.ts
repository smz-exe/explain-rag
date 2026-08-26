import { test, expect, type Page } from "@playwright/test";

/**
 * Deferred faithfulness verification is the production default: POST /query
 * returns faithfulness_status "pending" with no report, and the UI polls
 * GET /query/{id}/faithfulness until it resolves.
 */

const mockPapersResponse = { papers: [], total: 0 };
const mockClustersResponse = { clusters: [] };

const pendingQueryResponse = {
  query_id: "deferred-uuid-1",
  share_token: "test-share-token",
  question: "What is attention?",
  answer: "Attention is a mechanism [1] that allows models to focus.",
  citations: [
    { claim: "Attention mechanism", chunk_ids: ["chunk-1"], confidence: 0.9 },
  ],
  retrieved_chunks: [
    {
      chunk_id: "chunk-1",
      paper_id: "paper-1",
      paper_title: "Attention Is All You Need",
      content: "The Transformer uses self-attention.",
      similarity_score: 0.95,
      rerank_score: null,
      original_rank: 1,
      rank: 1,
    },
  ],
  // The defining difference from the legacy contract:
  faithfulness: null,
  faithfulness_status: "pending",
  trace: {
    embedding_time_ms: 50,
    retrieval_time_ms: 100,
    reranking_time_ms: null,
    generation_time_ms: 3000,
    faithfulness_time_ms: null,
    total_time_ms: 3150,
  },
};

const completedReport = {
  query_id: "deferred-uuid-1",
  status: "completed",
  faithfulness: {
    score: 0.9,
    claims: [
      {
        claim: "Attention is a mechanism",
        verdict: "supported",
        evidence_chunk_ids: ["chunk-1"],
        reasoning: "Directly stated in the retrieved chunks",
      },
    ],
  },
  faithfulness_time_ms: 1800,
};

async function stubAtlas(page: Page) {
  await page.route("**/papers/coordinates", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockPapersResponse),
    });
  });
  await page.route("**/papers/clusters", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(mockClustersResponse),
    });
  });
  await page.route("**/query", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(pendingQueryResponse),
    });
  });
}

async function ask(page: Page) {
  await page.goto("/");
  await page.getByRole("textbox").first().fill("What is attention?");
  await page.getByRole("button", { name: /ask/i }).first().click();
  await expect(page.getByText("Answer", { exact: true }).first()).toBeVisible({
    timeout: 10000,
  });
}

test.describe("Deferred faithfulness verification", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    await stubAtlas(page);
  });

  test("shows the answer immediately with verification still pending", async ({
    page,
  }) => {
    await page.route("**/query/*/faithfulness", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          query_id: "deferred-uuid-1",
          status: "pending",
          faithfulness: null,
          faithfulness_time_ms: null,
        }),
      });
    });

    await ask(page);

    await expect(
      page.getByText(/Attention is a mechanism/).first()
    ).toBeVisible();
    await expect(page.getByText(/Verifying the answer/i).first()).toBeVisible();
  });

  test("polls until verification completes, then renders the report", async ({
    page,
  }) => {
    let polls = 0;
    await page.route("**/query/*/faithfulness", async (route) => {
      polls += 1;
      const body =
        polls < 2
          ? {
              query_id: "deferred-uuid-1",
              status: "pending",
              faithfulness: null,
              faithfulness_time_ms: null,
            }
          : completedReport;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    });

    await ask(page);

    await expect(page.getByText(/Verifying the answer/i).first()).toBeVisible();

    await expect(page.getByText(/90%/).first()).toBeVisible({ timeout: 15000 });
    await expect(
      page.getByText(/1 of 1 claims supported/).first()
    ).toBeVisible();
    expect(polls).toBeGreaterThan(1);
  });

  test("sends the capability token on every poll", async ({ page }) => {
    const authHeaders: (string | undefined)[] = [];
    await page.route("**/query/*/faithfulness", async (route) => {
      authHeaders.push(route.request().headers()["authorization"]);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(completedReport),
      });
    });

    await ask(page);
    await expect(page.getByText(/90%/).first()).toBeVisible({ timeout: 15000 });

    expect(authHeaders.length).toBeGreaterThan(0);
    expect(authHeaders.every((h) => h === "Bearer test-share-token")).toBe(true);
  });

  test("reports a failed verification instead of spinning forever", async ({
    page,
  }) => {
    await page.route("**/query/*/faithfulness", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          query_id: "deferred-uuid-1",
          status: "failed",
          faithfulness: null,
          faithfulness_time_ms: null,
        }),
      });
    });

    await ask(page);

    await expect(
      page.getByText(/Verification unavailable/i).first()
    ).toBeVisible({ timeout: 15000 });
  });

  test("stops polling when the poll itself keeps failing", async ({ page }) => {
    let polls = 0;
    await page.route("**/query/*/faithfulness", async (route) => {
      polls += 1;
      await route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Access token expired" }),
      });
    });

    await ask(page);

    await expect(
      page.getByText(/Verification unavailable/i).first()
    ).toBeVisible({ timeout: 20000 });

    const pollsWhenSettled = polls;
    await page.waitForTimeout(5000);
    expect(polls).toBe(pollsWhenSettled);
  });
});
