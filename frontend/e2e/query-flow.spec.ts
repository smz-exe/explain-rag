import { test, expect } from "@playwright/test";

// Mock data for embedding space
const mockEmbeddingsResponse = {
  papers: [
    {
      paper_id: "paper-1",
      arxiv_id: "2301.00001",
      title: "Attention Is All You Need",
      coords: [0.5, 0.3, 0.2],
      cluster_id: 0,
      chunk_count: 15,
    },
  ],
  computed_at: "2024-01-01T00:00:00Z",
};

const mockClustersResponse = {
  clusters: [
    {
      id: 0,
      label: "Transformer Models",
      paper_ids: ["paper-1"],
    },
  ],
};

const mockQueryResponse = {
  query_id: "test-uuid-123",
  question: "What is attention?",
  answer: "Attention is a mechanism [1] that allows models to focus [2].",
  citations: [
    {
      claim: "Attention mechanism",
      chunk_ids: ["chunk-1"],
      confidence: 0.95,
    },
    { claim: "Focus capability", chunk_ids: ["chunk-2"], confidence: 0.88 },
  ],
  retrieved_chunks: [
    {
      chunk_id: "chunk-1",
      rank: 1,
      content: "The attention mechanism allows the model to focus on...",
      paper_title: "Attention Is All You Need",
      similarity_score: 0.92,
      rerank_score: 0.95,
    },
    {
      chunk_id: "chunk-2",
      rank: 2,
      content: "Self-attention allows the model to attend to different...",
      paper_title: "Attention Is All You Need",
      similarity_score: 0.85,
      rerank_score: null,
    },
  ],
  faithfulness: {
    score: 0.9,
    claims: [
      {
        claim: "Attention is a mechanism",
        verdict: "supported",
        reasoning: "Directly stated in the retrieved chunks",
      },
    ],
  },
  trace: {
    embedding_time_ms: 50,
    retrieval_time_ms: 100,
    reranking_time_ms: 200,
    generation_time_ms: 3000,
    faithfulness_time_ms: 2000,
    total_time_ms: 5350,
  },
};

test.describe("Query Flow", () => {
  // Use mobile viewport by default where only mobile layout is visible
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    // Mock embedding space APIs
    await page.route("**/papers/embeddings", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockEmbeddingsResponse),
      });
    });

    await page.route("**/papers/clusters", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockClustersResponse),
      });
    });

    // Mock the query API
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });
  });

  test("should submit query and display results", async ({ page }) => {
    await page.goto("/");

    // Fill in question (use first() as both layouts are in DOM)
    await page.getByRole("textbox").first().fill("What is attention?");

    // Submit
    await page.getByRole("button", { name: /ask/i }).first().click();

    // Wait for results - use exact match and first()
    await expect(page.getByText("Answer", { exact: true }).first()).toBeVisible(
      {
        timeout: 10000,
      }
    );
    await expect(
      page.getByText(/Attention is a mechanism/).first()
    ).toBeVisible();

    // Check chunks panel
    await expect(page.getByText(/Retrieved Chunks/).first()).toBeVisible();
    // Use first() since multiple chunks have the same paper title
    await expect(
      page.getByText(/Attention Is All You Need/).first()
    ).toBeVisible();

    // Check faithfulness report
    await expect(page.getByText(/Faithfulness Report/).first()).toBeVisible();
    await expect(page.getByText(/90%/).first()).toBeVisible();

    // Check timing
    await expect(page.getByText(/Timing/).first()).toBeVisible();
  });

  test("should show loading state during query", async ({ page }) => {
    // Slow down the response significantly
    await page.route("**/query", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").first().fill("Test question");
    await page.getByRole("button", { name: /ask/i }).first().click();

    // Check loading state - button should show querying and be disabled
    await expect(page.getByText(/Querying/i).first()).toBeVisible({
      timeout: 1000,
    });
  });

  test("should expand chunks when clicked", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("textbox").first().fill("What is attention?");
    await page.getByRole("button", { name: /ask/i }).first().click();

    await expect(page.getByText(/Retrieved Chunks/).first()).toBeVisible({
      timeout: 10000,
    });

    // Click to expand first chunk
    await page.getByText("#1").first().click();

    // Check content is visible
    await expect(
      page.getByText(/The attention mechanism allows/).first()
    ).toBeVisible();
  });

  test("should highlight chunk when citation clicked", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("textbox").first().fill("What is attention?");
    await page.getByRole("button", { name: /ask/i }).first().click();

    await expect(page.getByText("Answer", { exact: true }).first()).toBeVisible(
      {
        timeout: 10000,
      }
    );

    // Find and click the [1] citation badge (it's a Badge component with text [1])
    const citationBadge = page.getByText("[1]", { exact: true }).first();
    await citationBadge.click();

    // The chunk should be highlighted (has border-primary class)
    const chunk = page.locator('[id="chunk-chunk-1"]').first();
    await expect(chunk).toHaveClass(/border-primary/);
  });
});
