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
function addCopyButtonsToCodeBlocks() {
  // Find all code blocks: .k-default-codeblock divs and standalone <pre> tags with <code>
  const wrappedCodeBlocks = document.querySelectorAll('.k-default-codeblock');
  const preElements = document.querySelectorAll('.k-content pre');
  
  // Combine both types of code blocks
  const allCodeBlocks = [...wrappedCodeBlocks];
  
  // Add standalone pre elements that aren't already inside k-default-codeblock
  preElements.forEach((pre) => {
    if (!pre.closest('.k-default-codeblock') && pre.querySelector('code')) {
      allCodeBlocks.push(pre);
    }
  });
  
  allCodeBlocks.forEach((block) => {
    // Skip if button already exists
    if (block.querySelector('.code__copy--button')) {
      return;
    }
    
    // Create a wrapper div if the block is a <pre> element
    let container = block;
    if (block.tagName === 'PRE') {
      const wrapper = document.createElement('div');
      wrapper.className = 'code-block-wrapper';
      wrapper.style.position = 'relative';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);
      container = wrapper;
    } else {
      // For k-default-codeblock divs
      block.style.position = 'relative';
    }
    
    // Create copy button
    const button = document.createElement('button');
    button.className = 'code__copy--button';
    button.setAttribute('aria-label', 'Copy code to clipboard');
    
    // Create icon element
    const icon = document.createElement('i');
    icon.className = 'icon--copy';
    button.appendChild(icon);
    
    // Create tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'code__copy--tooltip';
    tooltip.textContent = 'Copied!';
    button.appendChild(tooltip);
    
    // Add button to container
    container.insertBefore(button, container.firstChild);
    
    // Add click event listener
    button.addEventListener('click', () => {
      // Find the code element
      const codeElement = container.querySelector('pre code') || container.querySelector('pre');
      if (!codeElement) return;
      
      const text = codeElement.innerText || codeElement.textContent;
      
      // Use modern clipboard API if available
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
          showCopyTooltip(button);
        }).catch(() => {
          // Fallback method
          copyWithFallback(text, button);
        });
      } else {
        // Fallback for older browsers
        copyWithFallback(text, button);
      }
    });
  });
}

function copyWithFallback(text, button) {
  const inputElement = document.createElement('textarea');
  inputElement.value = text;
  inputElement.setAttribute('readonly', '');
  inputElement.style.position = 'absolute';
  inputElement.style.left = '-9999px';
  document.body.appendChild(inputElement);
  inputElement.select();
  
  try {
    document.execCommand('copy');
    showCopyTooltip(button);
  } catch (err) {
    console.error('Failed to copy text:', err);
  }
  
  inputElement.remove();
}

function showCopyTooltip(button) {
  const tooltip = button.querySelector('.code__copy--tooltip');
  if (tooltip) {
    tooltip.style.display = 'block';
    setTimeout(() => {
      tooltip.style.display = 'none';
    }, 2000);
  }
}

// Add copy buttons on page load
document.addEventListener('DOMContentLoaded', addCopyButtonsToCodeBlocks);

// Existing copy buttons (for landing page compatibility)
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
