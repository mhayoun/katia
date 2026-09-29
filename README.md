# Katia

Outil d'**upload de photos** : chaque client se connecte avec **Google**, puis
téléverse ses photos (par exemple celles qu'il a téléchargées depuis un groupe
Facebook). Les fichiers sont stockés sur **Vercel Blob**, cloisonnés par client.

## Pourquoi un upload plutôt qu'une récupération Facebook ?

Meta a **supprimé l'API Groupes le 22 avril 2024** : aucune application tierce
ne peut plus lire le contenu (photos, publications) d'un groupe Facebook, même
pour le membre lui-même. Le seul chemin légal et fiable est donc que le client
télécharge ses photos depuis Facebook, puis les dépose ici.

## Flux

1. **Se connecter avec Google** (identité du client) — Auth.js.
2. **Téléverser** une ou plusieurs photos → stockées sur Vercel Blob sous un
   préfixe propre au client (`uploads/<hash-email>/…`).
3. **Galerie** : le client voit ses photos, peut en supprimer.

## Pile technique

- Next.js 15 (App Router) + TypeScript
- Auth.js (NextAuth v5) — Google Sign-In
- Vercel Blob (`@vercel/blob`) — stockage des fichiers

## Fichiers clés

| Fichier | Rôle |
|---|---|
| `auth.ts` | Config Auth.js (Google) |
| `app/page.tsx` | Page : login → upload → galerie |
| `components/UploadPanel.tsx` | Sélection + envoi des fichiers |
| `components/Gallery.tsx` | Affichage + suppression |
| `app/api/upload/route.ts` | `POST` (upload) et `DELETE` (suppression) vers Blob |
| `lib/uploads.ts` | Préfixe par utilisateur + listing Blob |

## Développement local

```bash
cp .env.example .env         # puis renseigner AUTH_GOOGLE_ID/SECRET
npx auth secret              # génère AUTH_SECRET
vercel env pull .env.local   # récupère BLOB_READ_WRITE_TOKEN
npm install
npm run dev                  # http://localhost:3000
```

URI de redirection Google (déjà configurée) :
`http://localhost:3000/api/auth/callback/google`

## Limites

- **4,5 Mo par fichier** (limite des Vercel Functions). Les photos sont envoyées
  une par une. Pour lever cette limite, migrer vers l'upload client
  (`@vercel/blob/client` + `handleUpload`).
- Le store Blob est **public** : les URLs sont non devinables (hash) mais
  accessibles par quiconque possède le lien. Pour un accès strictement privé,
  créer le store avec `--access private` et servir les fichiers via une route
  authentifiée.

## Déploiement

```bash
vercel --prod
```

Toutes les variables (`AUTH_SECRET`, `AUTH_GOOGLE_ID/SECRET`,
`BLOB_READ_WRITE_TOKEN`) sont déjà configurées sur le projet Vercel.
