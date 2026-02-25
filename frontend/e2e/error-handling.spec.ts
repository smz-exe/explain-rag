import { test, expect } from "@playwright/test";

// Mock data for embedding space
const mockEmbeddingsResponse = {
  papers: [],
  computed_at: null,
};

const mockClustersResponse = {
  clusters: [],
};

test.describe("Error Handling", () => {
  // Use mobile viewport where only mobile layout is visible
  test.use({ viewport: { width: 390, height: 844 } });

  test.beforeEach(async ({ page }) => {
    // Mock embedding space APIs (empty state)
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
  test("should display error when API returns 500", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Internal server error" }),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").fill("test question");
    await page.getByRole("button", { name: /ask/i }).click();

    // Error message should be visible (use first() as layout renders both mobile and desktop)
    await expect(page.getByText(/Request Failed/i).first()).toBeVisible({
      timeout: 5000,
    });
  });

  test("should display network error when server unreachable", async ({
    page,
  }) => {
    await page.route("**/query", async (route) => {
      await route.abort("failed");
    });

    await page.goto("/");
    await page.getByRole("textbox").fill("test question");
    await page.getByRole("button", { name: /ask/i }).click();

    // Network error message should appear - check for Connection Error title
    await expect(page.getByText(/Connection Error/i).first()).toBeVisible({
      timeout: 5000,
    });
  });

  test("should show retry button on error", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Server error" }),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").fill("test");
    await page.getByRole("button", { name: /ask/i }).click();

    // Retry button should be visible (use first() as layout renders both mobile and desktop)
    await expect(
      page.getByRole("button", { name: /try again/i }).first()
    ).toBeVisible({
      timeout: 5000,
    });
  });

  test("should clear error on retry", async ({ page }) => {
    let requestCount = 0;

    await page.route("**/query", async (route) => {
      requestCount++;
      if (requestCount === 1) {
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Server error" }),
        });
      } else {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            query_id: "test",
            question: "test",
            answer: "Success!",
            citations: [],
            retrieved_chunks: [],
            faithfulness: { score: 1, claims: [] },
            trace: { total_time_ms: 100 },
          }),
        });
      }
    });

    await page.goto("/");
    await page.getByRole("textbox").fill("test");
    await page.getByRole("button", { name: /ask/i }).click();

    // First should error
    await expect(page.getByText(/Request Failed/i).first()).toBeVisible({
      timeout: 5000,
    });

    // Click retry (use first() as layout renders both mobile and desktop)
    await page
      .getByRole("button", { name: /try again/i })
      .first()
      .click();

    // Error should be cleared (we're back to initial state, need to submit again)
    await expect(page.getByText(/Request Failed/i).first()).not.toBeVisible();

    // Submit again
    await page.getByRole("button", { name: /ask/i }).first().click();

    // Now should succeed
    await expect(page.getByText("Success!").first()).toBeVisible({
      timeout: 5000,
    });
  });

  test("should handle validation error (422)", async ({ page }) => {
    await page.route("**/query", async (route) => {
      await route.fulfill({
        status: 422,
        contentType: "application/json",
        body: JSON.stringify({
          detail: "Question must not be empty",
        }),
      });
    });

    await page.goto("/");
    await page.getByRole("textbox").fill("x"); // minimal input to pass frontend validation
    await page.getByRole("button", { name: /ask/i }).click();

    // Validation error should be shown (use first() as layout renders both mobile and desktop)
    await expect(
      page.getByText(/Question must not be empty/i).first()
    ).toBeVisible({
      timeout: 5000,
    });
  });
});
