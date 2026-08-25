import { test, expect } from "@playwright/test";

/**
 * Admin auth flow: login, dashboard access, logout, and route guarding.
 *
 * These tests need a backend whose admin credentials are known to the test
 * runner, so they are skipped unless both env vars are set:
 *
 *   E2E_ADMIN_USERNAME / E2E_ADMIN_PASSWORD
 *
 * Local run (backend with throwaway credentials, then the spec):
 *
 *   # generate a hash for a throwaway password
 *   uv run python -c "import bcrypt; print(bcrypt.hashpw(b'e2e-password', bcrypt.gensalt(12)).decode())"
 *   ADMIN_USERNAME=e2e-admin ADMIN_PASSWORD_HASH='<hash>' PRELOAD_MODELS=false \
 *     uv run uvicorn src.main:app --port 8000
 *   E2E_ADMIN_USERNAME=e2e-admin E2E_ADMIN_PASSWORD=e2e-password \
 *     NEXT_PUBLIC_API_URL=http://localhost:8000 pnpm test:e2e e2e/auth.spec.ts
 */
const username = process.env.E2E_ADMIN_USERNAME;
const password = process.env.E2E_ADMIN_PASSWORD;

test.describe("Admin auth flow", () => {
  test.skip(
    !username || !password,
    "E2E_ADMIN_USERNAME / E2E_ADMIN_PASSWORD not set",
  );

  test("rejects invalid credentials", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Username").fill("nobody");
    await page.getByLabel("Password").fill("definitely-wrong");
    await page.getByRole("button", { name: "Sign In" }).click();

    await expect(page.getByText(/invalid|failed|unauthorized/i)).toBeVisible();
    await expect(page).toHaveURL(/\/login/);
  });

  test("logs in, shows the dashboard, and logs out", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Username").fill(username!);
    await page.getByLabel("Password").fill(password!);
    await page.getByRole("button", { name: "Sign In" }).click();

    await expect(page).toHaveURL(/\/admin/);
    await expect(
      page.getByRole("heading", { name: "Admin Dashboard" }),
    ).toBeVisible();
    await expect(page.getByText(username!)).toBeVisible();

    await page.getByRole("button", { name: "Logout" }).click();
    await expect(page).toHaveURL(/\/login/);
  });

  test("redirects unauthenticated visitors away from /admin", async ({
    page,
  }) => {
    await page.goto("/admin");
    await expect(page).toHaveURL(/\/login/);
  });
});
