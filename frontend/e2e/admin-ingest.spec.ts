import { test, expect, type Page } from "@playwright/test";

/**
 * Admin ingest form error handling.
 *
 * Auth is stubbed at the API level so these run without real credentials: the
 * point under test is how the form reacts to API failures, not the login flow
 * (covered in auth.spec.ts).
 */

async function stubAdminSession(page: Page) {
  await page.route("**/auth/me", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ username: "admin", is_admin: true }),
    });
  });
  await page.route("**/stats", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        paper_count: 0,
        chunk_count: 0,
        query_count: 0,
        avg_faithfulness: null,
      }),
    });
  });
  await page.route("**/papers", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ papers: [], total: 0 }),
    });
  });
  await page.route("**/query/list*", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ queries: [], total: 0 }),
    });
  });
}

async function search(page: Page) {
  await page.goto("/admin");
  // The search field lives behind the second tab; the default tab is by-ID.
  await page.getByRole("tab", { name: /search arxiv/i }).first().click();
  await page.getByPlaceholder(/search arxiv/i).first().fill("attention");
  await page.getByRole("button", { name: /search arxiv/i }).first().click();
}

test.describe("Admin ingest form", () => {
  test.beforeEach(async ({ page }) => {
    await stubAdminSession(page);
  });

  test("tells the user to log in again when the session expired", async ({
    page,
  }) => {
    // FastAPI's 401 body is {"detail": "Not authenticated"} — the string "401"
    // appears nowhere in it, so matching on the message text never fires.
    await page.route("**/papers/search*", async (route) => {
      await route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Not authenticated" }),
      });
    });

    await search(page);

    await expect(
      page.getByText(/authentication required/i).first()
    ).toBeVisible({ timeout: 10000 });
  });

  test("shows a generic message for non-auth failures", async ({ page }) => {
    await page.route("**/papers/search*", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "arXiv unreachable" }),
      });
    });

    await search(page);

    await expect(page.getByText(/failed to search arxiv/i).first()).toBeVisible(
      { timeout: 10000 }
    );
  });
});
