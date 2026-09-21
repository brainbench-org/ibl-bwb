/**
 * Start the primary sidebar collapsed, except on the API reference, and remember
 * the reader's choice.
 *
 * The theme only ever collapses on click, so the initial state is set here: a
 * class on <html> paints the collapsed geometry before first paint, then the
 * theme's own `pst-squeeze` class takes over and animates every later toggle.
 */

(function () {
  const KEY = "bwb-sidebar-collapsed";

  const read = () => {
    try {
      return localStorage.getItem(KEY);
    } catch (e) {
      return null;
    }
  };

  // The API reference is a long generated module list, and the section nav is the only
  // way to move through it, so those pages start expanded. A reader who has toggled the
  // sidebar themselves keeps their choice everywhere.
  const onApi = window.location.pathname.includes("/generated/api/");
  const stored = read();
  const wantsCollapsed = stored === null ? !onApi : stored !== "false";

  // below 960px the sidebar is an off-canvas drawer and the button is hidden
  const collapsed = wantsCollapsed && window.matchMedia("(min-width: 960px)").matches;
  if (collapsed) {
    document.documentElement.classList.add("bwb-sidebar-collapsed");
  }

  document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.getElementById("pst-primary-sidebar");
    const button = document.getElementById("pst-collapse-sidebar-button");
    if (collapsed && sidebar && button) {
      sidebar.classList.add("pst-squeeze");
      button.setAttribute("aria-expanded", "false");
    }
    // same geometry either way, so handing over is invisible
    requestAnimationFrame(() =>
      document.documentElement.classList.remove("bwb-sidebar-collapsed"),
    );
    if (!button) return;
    button.addEventListener("click", function () {
      // the theme flips aria-expanded after the transition, so this is the pre-click state
      try {
        localStorage.setItem(KEY, String(button.getAttribute("aria-expanded") === "true"));
      } catch (e) {
        // storage unavailable, the choice just does not persist
      }
    });
  });
})();
