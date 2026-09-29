import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MyPhotos",
  description: "Connectez-vous avec Google et visualisez vos photos depuis votre Drive.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
