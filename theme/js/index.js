import Glide from "@glidejs/glide";

const carousels = document.querySelector(".glide");

if (carousels) {
  new Glide(".glide", {
    type: "carousel",
    perView: 7,
    focusAt: "center",
    gap: 0,
    autoplay: 1000,
    hoverpause: true,
    animationDuration: 500,
  }).mount();
}

const exploreModule = document.querySelector(".explore");
const exploreContent = document.querySelector(".explore__content");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        window.addEventListener("scroll", controlExploreContent);
        return;
      }

      window.removeEventListener("scroll", controlExploreContent);
    });
  },
  { threshold: 0 }
);

observer.observe(exploreModule);

function controlExploreContent() {
  const container = exploreModule.getBoundingClientRect();
  const containerTop = container.top;
  const containerHeight = exploreModule.clientHeight;
  const containerCenter = containerTop + containerHeight / 2;

  const viewportHeight = window.innerHeight;
  const viewportCenter = viewportHeight / 2;

  if (
    containerCenter >= (viewportCenter - containerHeight) / 2 &&
    containerCenter <= (viewportCenter + containerHeight) / 2
  ) {
    const scrollProgress = window.scrollY - containerTop;

    const normalizedScroll = Math.min(
      Math.max(scrollProgress / containerHeight, 0),
      1
    );

    const easeInOut =
      normalizedScroll < 0.5
        ? 2 * Math.pow(normalizedScroll, 2)
        : -1 + (4 - 2 * normalizedScroll) * normalizedScroll;

    const maxMove = containerHeight - exploreContent.clientHeight;
    const moveAmount = Math.max(
      0,
      Math.min(maxMove, scrollProgress * easeInOut * 0.4)
    );

    exploreContent.style.top = `${moveAmount}px`;
    return;
  }

  exploreContent.style.top = `0px`;
}
