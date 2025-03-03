// Mobile nav controls
const navButton = document.querySelector('.nav__menu--button');
const closeButton = document.querySelector('.nav__menu--close');
const mobileNavMenu = document.querySelector('.k-nav');

navButton.addEventListener('click', () => {
  mobileNavMenu.style.display = 'block';
  closeButton.style.display = 'block';
  navButton.style.display = 'none';
});

closeButton.addEventListener('click', () => {
  mobileNavMenu.style.display = 'none';
  closeButton.style.display = 'none';
  navButton.style.display = 'block';
});

const copyButtons = document.querySelectorAll('.code__copy--button');
copyButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const parent = button.parentNode;
    const text = parent.querySelector('.language-python').innerText;
    navigator.clipboard.writeText(text);
  });
});

// Search controls
const searchForms = document.querySelectorAll('.nav__search');
const mobileNavSearchIcon = document.querySelector('.nav__search--mobile');
const mobileNavSearchForm = document.querySelector('.nav__search-form--mobile');
const mobileNavControls = document.querySelector('.nav__controls--mobile');
const pageContainer = document.querySelector('.page__container');

mobileNavSearchIcon.addEventListener('click', () => {
  mobileNavControls.style.display = 'none';
  mobileNavSearchForm.style.display = 'block';
});

searchForms.forEach((search) => {
  search.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = search.querySelector('.nav__search--input').value;
    window.location = `/search.html?query=${text}`;
  });
});

pageContainer.addEventListener('click', () => {
  mobileNavControls.style.display = 'flex';
  mobileNavSearchForm.style.display = 'none';
});

// Parallax functionality
const exploreModule = document.querySelector('.explore');
const exploreContent = document.querySelector('.explore__content');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        window.addEventListener('scroll', controlExploreContent);
        return;
      }

      window.removeEventListener('scroll', controlExploreContent);
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
