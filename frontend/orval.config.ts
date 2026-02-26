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
        query: {
          useQuery: true,
          useMutation: true,
        },
      },
    },
  },
});
