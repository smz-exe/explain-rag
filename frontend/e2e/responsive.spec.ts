import { test, expect } from "@playwright/test";

const mockQueryResponse = {
  query_id: "test",
  question: "test",
  answer: "test answer with citation [1]",
  citations: [{ claim: "test", chunk_ids: ["chunk-1"], confidence: 0.9 }],
  retrieved_chunks: [
    {
      chunk_id: "chunk-1",
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

  test("should display form correctly on mobile", async ({ page }) => {
    await page.goto("/");

    // Heading should be visible
    await expect(
      page.getByRole("heading", { name: "ExplainRAG" })
    ).toBeVisible();

    // Query input should be visible
    await expect(page.getByRole("textbox")).toBeVisible();

    // Controls should not overflow
    await expect(page.getByText(/Top K:/)).toBeVisible();
    await expect(page.getByRole("switch")).toBeVisible();

    // Button should be visible
    const askButton = page.getByRole("button", { name: /ask/i });
    await expect(askButton).toBeVisible();
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
    await page.getByRole("textbox").fill("test");
    await page.getByRole("button", { name: /ask/i }).click();

    // Results should be visible - use exact match
    await expect(
      page.getByText("Answer", { exact: true })
    ).toBeVisible({ timeout: 5000 });

    // All sections should be visible (stacked vertically)
    await expect(page.getByText(/Retrieved Chunks/)).toBeVisible();
    await expect(page.getByText(/Faithfulness Report/)).toBeVisible();
    await expect(page.getByText(/Timing/)).toBeVisible();
  });
});

test.describe("Responsive Design - Tablet", () => {
  test.use({ viewport: { width: 768, height: 1024 } });

  test("should show layout on tablet", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").fill("test");
    await page.getByRole("button", { name: /ask/i }).click();

    // Results should be visible - use exact match
    await expect(
      page.getByText("Answer", { exact: true })
    ).toBeVisible({ timeout: 5000 });

    // All sections should be rendered
    await expect(page.getByText(/Retrieved Chunks/)).toBeVisible();
    await expect(page.getByText(/Faithfulness Report/)).toBeVisible();
  });
});

test.describe("Responsive Design - Desktop", () => {
  test.use({ viewport: { width: 1280, height: 800 } });

  test("should show 3-column layout on desktop", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockQueryResponse),
      });
    });

    await page.goto("/");

    // Container should have reasonable width
    const main = page.locator("main");
    const box = await main.boundingBox();
    expect(box?.width).toBeGreaterThan(1000);

    await page.getByRole("textbox").fill("test");
    await page.getByRole("button", { name: /ask/i }).click();

    // Results should be visible - use exact match
    await expect(
      page.getByText("Answer", { exact: true })
    ).toBeVisible({ timeout: 5000 });
  });
});
