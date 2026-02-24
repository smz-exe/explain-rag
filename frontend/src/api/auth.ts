/**
 * Authentication API functions.
 * These are manually created (not orval-generated) for direct control over auth flow.
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface LoginRequest {
  username: string;
  password: string;
}

export interface UserInfo {
  username: string;
  is_admin: boolean;
}

export class AuthError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "AuthError";
    this.status = status;
  }
}

/**
 * Login with username and password.
 * Sets httpOnly cookie on success.
 */
export async function login(data: LoginRequest): Promise<void> {
  const response = await fetch(`${BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
    credentials: "include",
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new AuthError(
      errorData.detail || "Login failed",
      response.status
    );
  }
}

/**
 * Logout and clear the auth cookie.
 */
export async function logout(): Promise<void> {
  const response = await fetch(`${BASE_URL}/auth/logout`, {
    method: "POST",
    credentials: "include",
  });

  if (!response.ok) {
    throw new AuthError("Logout failed", response.status);
  }
}

/**
 * Get the current authenticated user.
 * Throws if not authenticated.
 */
export async function getMe(): Promise<UserInfo> {
  const response = await fetch(`${BASE_URL}/auth/me`, {
    credentials: "include",
  });

  if (!response.ok) {
    throw new AuthError("Not authenticated", response.status);
  }

  return response.json();
}
