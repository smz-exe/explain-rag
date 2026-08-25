"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import {
  getGetCurrentUserAuthMeGetQueryKey,
  useGetCurrentUserAuthMeGet,
  useLoginAuthLoginPost,
  useLogoutAuthLogoutPost,
} from "@/api/queries/auth/auth";
import type { LoginRequest, UserResponse } from "@/api/model";

export interface UseAuthReturn {
  user: UserResponse | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (data: LoginRequest) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
}

/**
 * Hook for managing authentication state.
 *
 * Built on the orval-generated React Query hooks: the current-user query is
 * shared through the query cache, so every component using this hook sees the
 * same auth state without duplicate requests.
 */
export function useAuth(): UseAuthReturn {
  const router = useRouter();
  const queryClient = useQueryClient();

  // custom-fetch throws on non-200 (e.g. 401 when logged out), which lands
  // the query in its error state — treated as "not authenticated"
  const meQuery = useGetCurrentUserAuthMeGet({
    query: { retry: false },
  });

  const user: UserResponse | null =
    meQuery.data?.status === 200 ? meQuery.data.data : null;

  const loginMutation = useLoginAuthLoginPost();
  const logoutMutation = useLogoutAuthLogoutPost();

  const invalidateAuth = async () => {
    await queryClient.invalidateQueries({
      queryKey: getGetCurrentUserAuthMeGetQueryKey(),
    });
  };

  const checkAuth = async () => {
    await meQuery.refetch();
  };

  const login = async (data: LoginRequest) => {
    await loginMutation.mutateAsync({ data });
    await invalidateAuth();
    router.push("/admin");
  };

  const logout = async () => {
    await logoutMutation.mutateAsync();
    await invalidateAuth();
    router.push("/login");
  };

  return {
    user,
    isLoading: meQuery.isLoading,
    isAuthenticated: !!user,
    login,
    logout,
    checkAuth,
  };
}
