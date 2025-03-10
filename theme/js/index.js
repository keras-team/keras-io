// Mobile nav controls
const navButton = document.querySelector('.nav__menu--button');
const closeButton = document.querySelector('.nav__menu--close');
const mobileNavMenu = document.querySelector('.k-nav');
const pageContainer = document.querySelector('.page__container');

navButton.addEventListener('click', () => {
  mobileNavMenu.style.display = 'block';
  closeButton.style.display = 'block';
  navButton.style.display = 'none';
  pageContainer.style.position = 'fixed';
});

closeButton.addEventListener('click', () => {
  mobileNavMenu.style.display = 'none';
  closeButton.style.display = 'none';
  navButton.style.display = 'block';
  pageContainer.style.position = 'static';
});

// Copy code
const copyButtons = document.querySelectorAll('.code__copy--button');
copyButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const parent = button.parentNode;
    const text = parent.querySelector('.language-python').innerText;
    const inputElement = document.createElement('textarea');
    console.log('text', text);
    inputElement.value = text;
    inputElement.setAttribute('class', 'visually-hidden');
    const body = document.body;
    body.appendChild(inputElement);
    inputElement.select();
    document.execCommand('copy');
    inputElement.remove();

    button.querySelector('.code__copy--tooltip').style.display = 'block';
    setTimeout(() => {
      button.querySelector('.code__copy--tooltip').style.display = 'none';
    }, 2000);
  });
});

// Search controls
const searchForms = document.querySelectorAll('.nav__search');
const mobileNavSearchIcon = document.querySelector('.nav__search--mobile');
const mobileNavSearchForm = document.querySelector('.nav__search-form--mobile');
const mobileNavControls = document.querySelector('.nav__controls--mobile');
const desktopSearch = document.querySelector('.nav__menu .nav__search');

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

mobileNavMenu.addEventListener('click', () => {
  mobileNavControls.style.display = 'flex';
  mobileNavSearchForm.style.display = 'none';
});

if (window.location.pathname.indexOf('search.html') > -1) {
  desktopSearch.style.display = 'none';
  mobileNavSearchIcon.style.visibility = 'hidden';
}

// position:sticky functionality (set margin-top so that the content is correctly centered vertically)
const exploreModule = document.querySelector('.explore');
const exploreContent = document.querySelector('.explore__content');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        window.addEventListener('resize', verticallyCenterExploreContent);
        return;
      }

      window.removeEventListener('resize', verticallyCenterExploreContent);
    });
  },
  { threshold: 0 }
);

if (exploreModule && window.innerWidth > 1199) {
  observer.observe(exploreModule);
  /* let's call it once initially to align it in case a screen never gets resized */
  verticallyCenterExploreContent();
}

function verticallyCenterExploreContent() {
  exploreContent.style.marginTop = `${Math.round(exploreContent.getBoundingClientRect().height / 2)}px`;
}
