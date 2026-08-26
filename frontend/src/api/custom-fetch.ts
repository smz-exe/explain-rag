const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Get the base API URL.
 * Use this for constructing URLs that bypass the fetch wrapper (e.g., file downloads).
 */
export const getApiBaseUrl = () => BASE_URL;

/**
 * Download a query export as a Markdown file.
 *
 * Reading a stored query requires the capability token issued with it. The
 * token is sent as an Authorization header rather than a query parameter, so
 * it never lands in browser history, referrers, or server access logs — which
 * rules out `window.open` and means the file is fetched and saved from a blob.
 */
export const downloadQueryExport = async (
  queryId: string,
  shareToken: string
): Promise<void> => {
  let response: Response;

  try {
    response = await fetch(`${BASE_URL}/query/${queryId}/export`, {
      credentials: "include",
      headers: { Authorization: `Bearer ${shareToken}` },
    });
  } catch {
    throw new Error(
      "Failed to connect to server. Please check that the backend is running."
    );
  }

  if (!response.ok) {
    throw new APIError(
      response.status === 401 || response.status === 403
        ? "This export link has expired. Run the query again to download it."
        : `Export failed with status ${response.status}`,
      response.status
    );
  }

  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `query-${queryId.slice(0, 8)}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
};

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
