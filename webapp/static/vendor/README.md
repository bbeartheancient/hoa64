# Vendored third-party JS

Pinned copies, served statically by the webapp (no build step, no CDN at
runtime). Three.js is MIT-licensed (© 2010–2024 Three.js authors).

| File | Version | Source |
| --- | --- | --- |
| `three.module.js` | 0.170.0 | https://unpkg.com/three@0.170.0/build/three.module.js |
| `OrbitControls.js` | 0.170.0 | https://unpkg.com/three@0.170.0/examples/jsm/controls/OrbitControls.js |

Downloaded 2026-08-13. `OrbitControls.js` imports only the bare specifier
`three`, resolved via the import map in `index.html` — no edits were made
to either file.
