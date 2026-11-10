/*
 * hermes-skill-router has no web dashboard UI.
 *
 * The dashboard plugin loader injects the manifest's default `entry` (this file) as a script once
 * `dashboard/manifest.json` is discovered, so the file exists to make that request resolve. The
 * plugin's only dashboard half is the read-only /api/plugins/hermes-skill-router/suggestions route;
 * the Hermes Desktop pane is a separate runtime plugin and the Ink TUI half is a user widget. The
 * manifest hides the tab, and this module registers nothing, imports nothing and touches no global.
 */
