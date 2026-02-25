import { test, expect } from "@playwright/test";

// Mock data for embedding space
const mockEmbeddingsResponse = {
  papers: [
    {
      paper_id: "paper-1",
      arxiv_id: "2301.00001",
      title: "Test Paper",
      coords: [0.5, 0.3, 0.2],
      cluster_id: 0,
      chunk_count: 10,
    },
  ],
  computed_at: "2024-01-01T00:00:00Z",
};

const mockClustersResponse = {
  clusters: [
    {
      id: 0,
      label: "Test Cluster",
      paper_ids: ["paper-1"],
    },
  ],
};

const mockQueryResponse = {
  query_id: "test",
  question: "test",
  answer: "test answer with citation [1]",
  citations: [{ claim: "test", chunk_ids: ["chunk-1"], confidence: 0.9 }],
  retrieved_chunks: [
    {
      chunk_id: "chunk-1",
      paper_id: "paper-1",
      rank: 1,
      content: "Test content",
      paper_title: "Test Paper",
      similarity_score: 0.9,
      rerank_score: null,
    },
  ],
  faithfulness: {
    score: 0.9,
    claims: [{ claim: "test", verdict: "supported", reasoning: "test" }],
  },
  trace: { total_time_ms: 1000 },
};

test.describe("Responsive Design - Mobile", () => {
  test.use({ viewport: { width: 390, height: 844 } }); // iPhone 12 dimensions

  test.beforeEach(async ({ page }) => {
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
  });

  test("should display stacked layout on mobile", async ({ page }) => {
    await page.goto("/");

    // Query input should be visible (use first() as both layouts are in DOM)
    await expect(page.getByRole("textbox").first()).toBeVisible();

    // Controls should not overflow
    await expect(page.getByText(/Top K:/).first()).toBeVisible();
    await expect(page.getByRole("switch").first()).toBeVisible();

    // Button should be visible
    const askButton = page.getByRole("button", { name: /ask/i }).first();
    await expect(askButton).toBeVisible();

    // Papers should be visible below
    await expect(page.getByText("Test Cluster").first()).toBeVisible();
  });

  test("should display results correctly on mobile", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").first().fill("test");
    await page.getByRole("button", { name: /ask/i }).first().click();

    // Results should be visible - use exact match and first()
    await expect(page.getByText("Answer", { exact: true }).first()).toBeVisible(
      {
        timeout: 5000,
      }
    );

    // All sections should be visible (stacked vertically)
    await expect(page.getByText(/Retrieved Chunks/).first()).toBeVisible();
    await expect(page.getByText(/Faithfulness Report/).first()).toBeVisible();
    await expect(page.getByText(/Timing/).first()).toBeVisible();
  });
});

test.describe("Responsive Design - Tablet", () => {
  test.use({ viewport: { width: 768, height: 1024 } });

  test.beforeEach(async ({ page }) => {
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
  });

  test("should show stacked layout on tablet (below lg breakpoint)", async ({
    page,
  }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").first().fill("test");
    await page.getByRole("button", { name: /ask/i }).first().click();

    // Results should be visible - use exact match and first()
    await expect(page.getByText("Answer", { exact: true }).first()).toBeVisible(
      {
        timeout: 5000,
      }
    );

    // All sections should be rendered
    await expect(page.getByText(/Retrieved Chunks/).first()).toBeVisible();
    await expect(page.getByText(/Faithfulness Report/).first()).toBeVisible();
  });
});

test.describe("Responsive Design - Desktop", () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  // Skip on mobile projects - these tests require desktop viewport
  test.beforeEach(async ({ page }, testInfo) => {
    if (testInfo.project.name.toLowerCase().includes("mobile")) {
      testInfo.skip();
      return;
    }

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
  });

  test("should show 3-column layout on desktop", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });

    await page.goto("/");

    // Check for panel headings (desktop layout - these only exist in desktop layout)
    await expect(page.getByRole("heading", { name: "Papers" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Query" })).toBeVisible();

    // Center visualization area should exist
    const main = page.locator("main");
    const box = await main.boundingBox();
    expect(box?.width).toBeGreaterThan(400);

    // Use last() as desktop layout is second in DOM
    await page.getByRole("textbox").last().fill("test");
    await page.getByRole("button", { name: /ask/i }).last().click();

    // Results should be visible - use last() for desktop layout
    await expect(page.getByText("Answer", { exact: true }).last()).toBeVisible({
      timeout: 5000,
    });
  });

  test("should show papers panel on left side", async ({ page }) => {
    await page.goto("/");

    // Papers panel should show cluster (use last() as desktop layout is second in DOM)
    await expect(page.getByText("Test Cluster").last()).toBeVisible();
    await expect(page.getByText("Test Paper").last()).toBeVisible();
  });
});
