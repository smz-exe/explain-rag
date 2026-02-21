const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Custom fetch wrapper for orval-generated API clients.
 * Handles base URL prefixing and response parsing.
 */
export const customFetch = async <T>(
  url: string,
  options?: RequestInit
): Promise<T> => {
  const response = await fetch(`${BASE_URL}${url}`, {
    ...options,
  });

  if (!response.ok) {
    let errorMessage = "API request failed";
    try {
      const errorData = await response.json();
      errorMessage = errorData.message || errorData.detail || errorMessage;
    } catch {
      // If parsing fails, use default message
    }
    throw new Error(errorMessage);
  }

  const data = await response.json();

  // Return in orval's expected format with data and status
  return {
    data,
    status: response.status,
    headers: response.headers,
  } as T;
};
