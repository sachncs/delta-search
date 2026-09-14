/* DeltaSearch site interactions -------------------------------------------- */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* --- nav ---------------------------------------------------------------- */
  var nav = document.getElementById("nav");
  var burger = document.getElementById("nav-burger");
  var panel = document.getElementById("nav-panel");

  function setNavOpen(open) {
    document.body.classList.toggle("nav-open", open);
    if (burger) burger.setAttribute("aria-expanded", String(open));
  }

  function onScroll() {
    if (nav) nav.classList.toggle("is-scrolled", window.scrollY > 24);
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if (burger && panel) {
    burger.addEventListener("click", function () {
      setNavOpen(!document.body.classList.contains("nav-open"));
    });

    panel.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        setNavOpen(false);
      });
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") setNavOpen(false);
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 880) setNavOpen(false);
    });
  }

  /* --- reveal on scroll ---------------------------------------------------- */
  var reveals = document.querySelectorAll(".reveal");

  if ("IntersectionObserver" in window && !reduceMotion) {
    var revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    reveals.forEach(function (el) {
      revealObserver.observe(el);
    });
  } else {
    reveals.forEach(function (el) {
      el.classList.add("is-visible");
    });
  }

  /* --- animated counters ---------------------------------------------------- */
  function animateCount(el, target, decimals, suffix, duration) {
    if (reduceMotion) {
      el.textContent = target.toLocaleString() + (suffix || "");
      return;
    }
    var start = performance.now();
    function step(now) {
      var t = Math.min((now - start) / duration, 1);
      var eased = 1 - Math.pow(1 - t, 4); // ease-out quart
      var value = target * eased;
      el.textContent =
        value.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        }) +
        (suffix || "");
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  var counters = Array.prototype.slice.call(document.querySelectorAll("[data-count]"));

  if ("IntersectionObserver" in window) {
    var countObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var el = entry.target;
          var target = parseFloat(el.getAttribute("data-count"));
          if (!isFinite(target)) return;
          var decimals = parseInt(el.getAttribute("data-decimals") || "0", 10);
          var suffix = el.getAttribute("data-suffix") || "";
          animateCount(el, target, decimals, suffix, 1100);
          countObserver.unobserve(el);
        });
      },
      { threshold: 0.4 }
    );
    counters.forEach(function (el) {
      countObserver.observe(el);
    });
  } else {
    counters.forEach(function (el) {
      animateCount(el, parseFloat(el.getAttribute("data-count")), parseInt(el.getAttribute("data-decimals") || "0", 10), el.getAttribute("data-suffix") || "", 1);
    });
  }

  /* --- code tabs ------------------------------------------------------------- */
  var tabs = document.querySelectorAll(".code__tab");
  var panes = document.querySelectorAll(".code__pane");
  var copyBtn = document.querySelector(".code__copy");

  function activateTab(name) {
    tabs.forEach(function (tab) {
      var active = tab.getAttribute("data-tab") === name;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    panes.forEach(function (pane) {
      var show = pane.getAttribute("data-pane") === name;
      pane.hidden = !show;
    });
    var title = document.querySelector(".code__win-bar-title");
    if (title) title.textContent = name === "py" ? "solve.py" : "shell";
  }

  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      activateTab(tab.getAttribute("data-tab"));
    });
  });

  if (copyBtn) {
    function copyLegacy(text) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
      } catch (err) {
        /* noop */
      }
      document.body.removeChild(ta);
    }

    copyBtn.addEventListener("click", function () {
      var active = document.querySelector(".code__pane:not([hidden]) pre code");
      if (!active) return;
      var text = active.textContent;
      var flash = function () {
        copyBtn.textContent = "Copied ✓";
        setTimeout(function () {
          copyBtn.textContent = "Copy";
        }, 1800);
      };

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(flash, function () {
          copyLegacy(text);
          flash();
        });
      } else {
        copyLegacy(text);
        flash();
      }
    });
  }
})();