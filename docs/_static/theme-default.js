// Default the Furo theme to dark mode on a visitor's first load.
//
// Furo persists the chosen theme in localStorage under "theme" (values
// "light" / "dark" / "auto") and, when unset, falls back to "auto" (follow the
// OS). We prime it to "dark" on the very first visit so the site opens in dark
// mode regardless of OS preference. A returning visitor who has toggled to
// "light" or "auto" keeps their choice — we only act when nothing is stored.
(function () {
  function stored() {
    try {
      return localStorage.getItem("theme");
    } catch (e) {
      return null;
    }
  }

  // Prime storage synchronously so Furo's own init reads "dark".
  if (stored() === null) {
    try {
      localStorage.setItem("theme", "dark");
    } catch (e) {}
  }

  function applyDark() {
    if (stored() === "dark" && document.body) {
      document.body.dataset.theme = "dark";
    }
  }

  // Force the body attribute too, in case Furo relies on the OS media query
  // for the unset case. Runs before and/or after Furo's own DOMContentLoaded
  // init; either ordering ends up dark.
  if (document.readyState !== "loading") {
    applyDark();
  } else {
    document.addEventListener("DOMContentLoaded", applyDark);
  }
})();
