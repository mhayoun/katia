import { auth, signIn, signOut } from "@/auth";
import PhotoApp from "@/components/PhotoApp";

export default async function Home() {
  const session = await auth();

  return (
    <main className="container">
      <h1>MyPhotos</h1>
      <p className="subtitle">
        Connectez-vous avec Google, puis visualisez vos photos directement depuis
        votre Google Drive.
      </p>

      {!session?.user ? (
        <SignedOut />
      ) : (
        <SignedIn
          email={session.user.email ?? ""}
          name={session.user.name ?? ""}
          image={session.user.image ?? ""}
          accessToken={session.accessToken}
        />
      )}
    </main>
  );
}

function SignedOut() {
  return (
    <div className="card">
      <div className="step">
        <span className="num">1</span> Se connecter
      </div>
      <p className="hint" style={{ marginTop: 0 }}>
        Commencez par vous identifier avec votre compte Google.
      </p>
      <form
        action={async () => {
          "use server";
          await signIn("google", { redirectTo: "/" });
        }}
      >
        <button className="btn btn-google" type="submit">
          Se connecter avec Google
        </button>
      </form>
    </div>
  );
}

function SignedIn({
  email,
  name,
  image,
  accessToken,
}: {
  email: string;
  name: string;
  image: string;
  accessToken?: string;
}) {
  return (
    <>
      <div className="card">
        <div className="row">
          <div className="user">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            {image ? <img className="avatar" src={image} alt="" /> : null}
            <div>
              <div style={{ fontWeight: 600 }}>{name || email}</div>
              <div className="hint" style={{ marginTop: 0 }}>
                {email}
              </div>
            </div>
          </div>
          <form
            action={async () => {
              "use server";
              await signOut({ redirectTo: "/" });
            }}
          >
            <button className="btn btn-ghost" type="submit">
              Se déconnecter
            </button>
          </form>
        </div>
      </div>

      <PhotoApp accessToken={accessToken} />
    </>
  );
}
