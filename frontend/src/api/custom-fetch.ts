const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class APIError extends Error {
  status: number;
  detail?: string;

  constructor(message: string, status: number, detail?: string) {
    super(message);
    this.name = "APIError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Custom fetch wrapper for orval-generated API clients.
 * Handles base URL prefixing and response parsing.
 */
export const customFetch = async <T>(
  url: string,
  options?: RequestInit
): Promise<T> => {
  let response: Response;

  try {
    response = await fetch(`${BASE_URL}${url}`, {
      ...options,
      credentials: "include", // Include cookies for auth
    });
  } catch {
    throw new Error(
      "Failed to connect to server. Please check that the backend is running."
    );
  }

  if (!response.ok) {
    let errorMessage = "API request failed";
    let detail: string | undefined;

    try {
      const errorData = await response.json();
      errorMessage =
        errorData.message ||
        errorData.detail ||
        `Request failed with status ${response.status}`;
      detail = errorData.detail;
    } catch {
      errorMessage = `Request failed with status ${response.status}`;
    }

    throw new APIError(errorMessage, response.status, detail);
  }

  const data = await response.json();

  // Return in orval's expected format with data and status
  return {
    data,
    status: response.status,
    headers: response.headers,
  } as T;
};
