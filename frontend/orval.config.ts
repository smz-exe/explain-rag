import { defineConfig } from "orval";

export default defineConfig({
  explainrag: {
    input: {
      target: "http://localhost:8000/openapi.json",
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
