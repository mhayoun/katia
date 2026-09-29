import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

/**
 * Google Sign-In with the `drive.file` scope, plus automatic access-token
 * refresh. Google access tokens expire after ~1h; we keep the refresh token and
 * mint a fresh access token server-side whenever the current one is near expiry,
 * so Drive calls from the browser keep working through a long session.
 */
async function refreshAccessToken(token: any) {
  try {
    const res = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: process.env.AUTH_GOOGLE_ID!,
        client_secret: process.env.AUTH_GOOGLE_SECRET!,
        grant_type: "refresh_token",
        refresh_token: token.refreshToken,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw data;
    return {
      ...token,
      accessToken: data.access_token,
      expiresAt: Math.floor(Date.now() / 1000) + (data.expires_in ?? 3600),
      // Google may or may not return a new refresh token.
      refreshToken: data.refresh_token ?? token.refreshToken,
      error: undefined,
    };
  } catch {
    return { ...token, error: "RefreshFailed" };
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Google({
      authorization: {
        params: {
          scope:
            "openid email profile https://www.googleapis.com/auth/drive.file",
          access_type: "offline",
          prompt: "consent",
        },
      },
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      // Initial sign-in: store tokens + expiry.
      if (account) {
        token.accessToken = account.access_token;
        token.refreshToken = account.refresh_token;
        token.expiresAt = account.expires_at;
        return token;
      }
      // Still valid (with a 60s safety margin)? Keep it.
      if (
        token.expiresAt &&
        Date.now() < (token.expiresAt as number) * 1000 - 60_000
      ) {
        return token;
      }
      // Expired → refresh if we can.
      if (token.refreshToken) return await refreshAccessToken(token);
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken as string | undefined;
      (session as any).error = (token as any).error;
      return session;
    },
  },
});
