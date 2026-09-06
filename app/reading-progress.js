// Shared by the React page and the offline export. Scrolling never re-renders the lesson.
export function observeReadingProgress() {
  const links = [...document.querySelectorAll(".sidenav a")];
  const sections = links.map((link) => document.querySelector(link.hash));
  const progress = document.querySelector(".progress");
  let frame = 0;

  function update() {
    frame = 0;
    const height = document.documentElement.scrollHeight - innerHeight;
    progress.style.width = `${height > 0 ? Math.min(100, Math.max(0, scrollY / height * 100)) : 0}%`;
    let active = 0;
    sections.forEach((section, index) => {
      if (section.getBoundingClientRect().top <= 130) active = index;
    });
    links.forEach((link, index) => {
      link.classList.toggle("active", index === active);
      if (index === active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(update);
  }

  update();
  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  return () => {
    removeEventListener("scroll", schedule);
    removeEventListener("resize", schedule);
    cancelAnimationFrame(frame);
  };
}
