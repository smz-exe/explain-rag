import { defineConfig } from "orval";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default defineConfig({
  explainrag: {
    input: {
      target: `${BASE_URL}/openapi.json`,
    },
    output: {
      mode: "tags-split",
      target: "./src/api/queries",
      schemas: "./src/api/model",
      client: "react-query",
      override: {
        mutator: {
          path: "./src/api/custom-fetch.ts",
          name: "customFetch",
        },
        // No query override: orval's defaults generate useQuery hooks for GET
        // and useMutation hooks for non-GET. Setting useQuery+useMutation
        // globally inverts that (mutation takes precedence for GET, query for
        // non-GET) since orval 8.x — which breaks every call site here.
      },
    },
  },
});
