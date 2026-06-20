/**
 * Vanilla-canvas dithered wave field — a 2D-canvas port of the
 * react-three-fiber Dither shader's visual concept (Perlin-style
 * flow field + Bayer ordered dithering + limited palette), without
 * the WebGL/Three.js dependency tree that Streamlit/static-HTML
 * can't load anyway.
 *
 * Signature element: starts heavily dithered / low color-count
 * ("RESOLVING SIGNAL…") and sharpens in resolution and color depth
 * over the first ~2.5s, like a sensor warming up — then settles
 * into a slow ambient wave for the rest of the page.
 */

(function () {
  const canvas = document.getElementById('radar-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d', { alpha: false });

  const prefersReducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)'
  ).matches;

  let width = 0;
  let height = 0;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);

  // Bayer 8x8 ordered-dither matrix, normalized 0..1
  const BAYER = [
    0, 32, 8, 40, 2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44, 4, 36, 14, 46, 6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
    3, 35, 11, 43, 1, 33, 9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47, 7, 39, 13, 45, 5, 37,
    63, 31, 55, 23, 61, 29, 53, 21,
  ].map((v) => v / 64);

  // Base palette derived from the dashboard's zone colors, dark end
  // weighted heavily since this sits behind text.
  const PALETTE = [
    [10, 14, 15],     // base
    [20, 28, 30],     // surface-2
    [30, 56, 52],     // dim green wash
    [58, 70, 56],     // dim yellow wash
    [70, 45, 38],     // dim red wash
  ];

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
  }

  window.addEventListener('resize', resize);
  resize();

  // Cheap 2D value-noise (not true Perlin, but smooth + fast enough
  // for an ambient background at low pixel-grid resolution).
  function hash(x, y) {
    const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
    return s - Math.floor(s);
  }

  function noise(x, y) {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const tl = hash(xi, yi);
    const tr = hash(xi + 1, yi);
    const bl = hash(xi, yi + 1);
    const br = hash(xi + 1, yi + 1);
    const u = xf * xf * (3 - 2 * xf);
    const v = yf * yf * (3 - 2 * yf);
    return tl * (1 - u) * (1 - v) + tr * u * (1 - v) + bl * (1 - u) * v + br * u * v;
  }

  let startTime = null;
  const RESOLVE_DURATION = 2500; // ms — "sensor warming up"

  function draw(now) {
    if (startTime === null) startTime = now;
    const elapsed = now - startTime;
    const resolveT = Math.min(elapsed / RESOLVE_DURATION, 1); // 0 -> 1

    // Low-res grid that we scale up — coarser early (less resolved),
    // finer once "resolved". Matches the pixelSize behavior of the
    // original shader's dither pass.
    const cellSize = prefersReducedMotion
      ? 10
      : Math.round(14 - resolveT * 6); // 14px blocky -> 8px finer

    const cols = Math.ceil(width / cellSize) + 1;
    const rows = Math.ceil(height / cellSize) + 1;

    const t = prefersReducedMotion ? 0 : elapsed * 0.00012;
    const freq = 0.012;

    // Color depth ramps from coarse (3 levels) to fuller (5 levels)
    // as resolveT progresses — visually "sharpening".
    const colorLevels = prefersReducedMotion
      ? PALETTE.length
      : Math.max(2, Math.round(2 + resolveT * (PALETTE.length - 2)));

    ctx.save();
    ctx.scale(dpr, dpr);

    for (let yi = 0; yi < rows; yi++) {
      for (let xi = 0; xi < cols; xi++) {
        const nx = xi * freq;
        const ny = yi * freq;
        let n = noise(nx + t, ny - t * 0.7);
        n += 0.5 * noise(nx * 2.3 + t * 1.4, ny * 2.3 - t);
        n = n / 1.5;

        const bx = xi % 8;
        const by = yi % 8;
        const threshold = BAYER[by * 8 + bx];

        let level = n + (threshold - 0.5) * 0.18;
        level = Math.max(0, Math.min(1, level));

        const paletteIdx = Math.min(
          colorLevels - 1,
          Math.floor(level * colorLevels)
        );
        const color = PALETTE[Math.min(paletteIdx, PALETTE.length - 1)];

        ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        ctx.fillRect(xi * cellSize, yi * cellSize, cellSize, cellSize);
      }
    }

    ctx.restore();

    const label = document.querySelector('.hero-resolve-label');
    if (label) {
      if (resolveT >= 1) {
        label.style.opacity = '0';
      }
    }

    if (!prefersReducedMotion || elapsed < RESOLVE_DURATION) {
      requestAnimationFrame(draw);
    }
  }

  requestAnimationFrame(draw);
})();
