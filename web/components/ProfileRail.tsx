import { FACTS, PROFILE } from "@/lib/profile";

/**
 * A profile column for themes that want one. It is always in the DOM and every
 * other theme hides it with CSS (.profile-rail), which keeps a theme change a
 * pure style change: no layout is chosen in JavaScript, so there is nothing to
 * mismatch between the server render and the browser.
 */
export function ProfileRail() {
  return (
    <aside className="profile-rail" aria-label={`About ${PROFILE.name}`}>
      <div className="profile-card">
        <span className="profile-monogram" aria-hidden>
          {PROFILE.initials}
        </span>
        <p className="profile-name">{PROFILE.name}</p>
        <p className="profile-role">{PROFILE.role}</p>
      </div>
      <ul className="profile-tiles">
        {FACTS.map((fact) => (
          <li key={fact.id} className="profile-tile" data-fact={fact.id}>
            {fact.label}
          </li>
        ))}
      </ul>
      <a className="profile-link" href={PROFILE.linkedinUrl} target="_blank" rel="noopener noreferrer">
        {PROFILE.linkedinLabel}
      </a>
    </aside>
  );
}
