import { test, expect } from "@playwright/test";

test.describe("Home Page", () => {
  test("should display the main heading and query form", async ({ page }) => {
    await page.goto("/");

    await expect(
      page.getByRole("heading", { name: "ExplainRAG" })
    ).toBeVisible();
    await expect(
      page.getByText("Explainable RAG for Academic Papers")
    ).toBeVisible();
    await expect(page.getByRole("textbox")).toBeVisible();
    await expect(page.getByRole("button", { name: /ask/i })).toBeVisible();
  });

  test("should have query controls", async ({ page }) => {
    await page.goto("/");

    // Top K slider
    await expect(page.getByText(/Top K:/)).toBeVisible();

    // Reranking toggle
    await expect(page.getByRole("switch")).toBeVisible();
    await expect(page.getByText(/Enable Reranking/)).toBeVisible();
  });

  test("Ask button should be disabled when input is empty", async ({
    page,
  }) => {
    await page.goto("/");

    const askButton = page.getByRole("button", { name: /ask/i });
    await expect(askButton).toBeDisabled();

    // Type something
    await page.getByRole("textbox").fill("test question");
    await expect(askButton).toBeEnabled();

    // Clear input
    await page.getByRole("textbox").fill("");
    await expect(askButton).toBeDisabled();
  });

  test("should toggle reranking switch", async ({ page }) => {
    await page.goto("/");

    const rerankingSwitch = page.getByRole("switch");
    await expect(rerankingSwitch).not.toBeChecked();

    await rerankingSwitch.click();
    await expect(rerankingSwitch).toBeChecked();

    await rerankingSwitch.click();
    await expect(rerankingSwitch).not.toBeChecked();
  });
});
