import { test, expect } from "@playwright/test";

test.describe("Error Handling", () => {
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

    // Error message should be visible
    await expect(page.getByText(/Request Failed/i)).toBeVisible({
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
    await expect(page.getByText(/Connection Error/i)).toBeVisible({
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

    // Retry button should be visible
    await expect(
      page.getByRole("button", { name: /try again/i })
    ).toBeVisible({ timeout: 5000 });
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
    await expect(page.getByText(/Request Failed/i)).toBeVisible({
      timeout: 5000,
    });

    // Click retry
    await page.getByRole("button", { name: /try again/i }).click();

    // Error should be cleared (we're back to initial state, need to submit again)
    await expect(page.getByText(/Request Failed/i)).not.toBeVisible();

    // Submit again
    await page.getByRole("button", { name: /ask/i }).click();

    // Now should succeed
    await expect(page.getByText("Success!")).toBeVisible({ timeout: 5000 });
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

    // Validation error should be shown
    await expect(page.getByText(/Question must not be empty/i)).toBeVisible({
      timeout: 5000,
    });
  });
});
