/**
 * Vanilla-JS animated counter — ported from the React Counter
 * component's effect (digits roll up to a target value) without
 * the React/Framer Motion dependency.
 *
 * Counts up from 0 to data-target over ~1.4s with an ease-out
 * curve, triggered when the element scrolls into view. Respects
 * prefers-reduced-motion by snapping straight to the target.
 */

(function () {
  const prefersReducedMotion = window.matchMedia(
    '(prefers-reduced-motion: reduce)'
  ).matches;

  function easeOutExpo(t) {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
  }

  function animateCount(el) {
    const target = Number(el.dataset.target || '0');
    if (prefersReducedMotion) {
      el.textContent = target.toLocaleString();
      return;
    }

    const duration = 1400;
    let startTime = null;

    function step(now) {
      if (startTime === null) startTime = now;
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);
      const eased = easeOutExpo(t);
      const current = Math.round(eased * target);
      el.textContent = current.toLocaleString();
      if (t < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }

  function init() {
    const counters = document.querySelectorAll('[data-counter]');
    if (!counters.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateCount(entry.target);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );

    counters.forEach((el) => observer.observe(el));
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Exposed so main.js can set real data-target values once fetched
  // from the dashboard's actual pipeline outputs, then re-trigger.
  window.IntelliParkCounter = { animateCount };
})();
