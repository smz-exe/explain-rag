"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  loginAuthLoginPost,
  logoutAuthLogoutPost,
  getCurrentUserAuthMeGet,
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
 */
export function useAuth(): UseAuthReturn {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  const fetchCurrentUser =
    useCallback(async (): Promise<UserResponse | null> => {
      try {
        const response = await getCurrentUserAuthMeGet();
        // custom-fetch throws on non-200, so we can safely narrow
        return response.status === 200 ? response.data : null;
      } catch {
        return null;
      }
    }, []);

  const checkAuth = useCallback(async () => {
    const currentUser = await fetchCurrentUser();
    setUser(currentUser);
    setIsLoading(false);
  }, [fetchCurrentUser]);

  useEffect(() => {
    // State updates happen in the promise callback, never synchronously
    // in the effect body (react-hooks/set-state-in-effect)
    let cancelled = false;
    fetchCurrentUser().then((currentUser) => {
      if (cancelled) return;
      setUser(currentUser);
      setIsLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [fetchCurrentUser]);

  const login = async (data: LoginRequest) => {
    await loginAuthLoginPost(data);
    await checkAuth();
    router.push("/admin");
  };

  const logout = async () => {
    await logoutAuthLogoutPost();
    setUser(null);
    router.push("/login");
  };

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    logout,
    checkAuth,
  };
}
