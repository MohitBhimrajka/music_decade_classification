document.addEventListener('DOMContentLoaded', function() {
  // Initialize components
  setupDarkModeToggle();
  setupTableOfContents();
  setupScrollProgress();
  setupScrollAnimations();
  setupImageZoom();
  highlightCodeBlocks();
  setupMobileNavigation();
});

/**
 * Dark Mode Toggle
 */
function setupDarkModeToggle() {
  const themeToggle = document.getElementById('theme-toggle');
  if (!themeToggle) return;
  
  // Check for saved theme preference or respect OS preference
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const savedTheme = localStorage.getItem('theme');
  
  // Apply saved theme or OS preference
  if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
    document.body.classList.add('dark-mode');
    updateThemeIcon(true);
  }
  
  // Handle toggle click
  themeToggle.addEventListener('click', function() {
    const isDark = document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    updateThemeIcon(isDark);
  });
}

/**
 * Update theme toggle icon based on current mode
 */
function updateThemeIcon(isDark) {
  const themeToggle = document.getElementById('theme-toggle');
  if (!themeToggle) return;
  
  themeToggle.innerHTML = isDark 
    ? '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>'
    : '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';
}

/**
 * Mobile Navigation
 */
function setupMobileNavigation() {
  // Create mobile navigation elements if they don't exist
  if (!document.querySelector('.mobile-nav')) {
    createMobileNavElements();
  }
  
  const mobileNavToggle = document.querySelector('.mobile-nav-toggle');
  const mobileNav = document.querySelector('.mobile-nav');
  const mobileNavOverlay = document.querySelector('.mobile-nav-overlay');
  const closeNavButton = document.querySelector('.close-nav');
  
  if (!mobileNavToggle || !mobileNav || !mobileNavOverlay || !closeNavButton) return;
  
  // Open mobile nav
  mobileNavToggle.addEventListener('click', function() {
    mobileNav.classList.add('active');
    mobileNavOverlay.classList.add('active');
    document.body.style.overflow = 'hidden'; // Prevent scrolling
  });
  
  // Close functions
  function closeNav() {
    mobileNav.classList.remove('active');
    mobileNavOverlay.classList.remove('active');
    document.body.style.overflow = ''; // Re-enable scrolling
  }
  
  closeNavButton.addEventListener('click', closeNav);
  mobileNavOverlay.addEventListener('click', closeNav);
  
  // Close nav on ESC key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && mobileNav.classList.contains('active')) {
      closeNav();
    }
  });
  
  // Close nav when a link is clicked
  const mobileNavLinks = document.querySelectorAll('.mobile-nav a');
  mobileNavLinks.forEach(link => {
    link.addEventListener('click', closeNav);
  });
}

/**
 * Create mobile navigation elements
 */
function createMobileNavElements() {
  // Create mobile nav overlay
  const overlay = document.createElement('div');
  overlay.classList.add('mobile-nav-overlay');
  document.body.appendChild(overlay);
  
  // Create mobile nav
  const mobileNav = document.createElement('div');
  mobileNav.classList.add('mobile-nav');
  
  // Create mobile nav header
  const mobileNavHeader = document.createElement('div');
  mobileNavHeader.classList.add('mobile-nav-header');
  
  const siteTitle = document.querySelector('.site-title h1').textContent;
  const siteTitleElement = document.createElement('div');
  siteTitleElement.textContent = siteTitle;
  mobileNavHeader.appendChild(siteTitleElement);
  
  const closeButton = document.createElement('button');
  closeButton.classList.add('close-nav');
  closeButton.setAttribute('aria-label', 'Close menu');
  closeButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';
  mobileNavHeader.appendChild(closeButton);
  
  mobileNav.appendChild(mobileNavHeader);
  
  // Create mobile nav content
  const mobileNavContent = document.createElement('div');
  mobileNavContent.classList.add('mobile-nav-content');
  
  // Clone desktop navigation links
  const desktopNav = document.querySelector('header nav');
  if (desktopNav) {
    const navClone = desktopNav.cloneNode(true);
    mobileNavContent.appendChild(navClone.querySelector('ul'));
  }
  
  mobileNav.appendChild(mobileNavContent);
  document.body.appendChild(mobileNav);
}

/**
 * Table of Contents handling
 */
function setupTableOfContents() {
  const toc = document.querySelector('.toc');
  if (!toc) return;
  
  // Get all h2 and h3 elements from the content
  const content = document.querySelector('.content');
  const headings = content.querySelectorAll('h2, h3');
  
  // Create TOC structure
  const tocList = document.createElement('ul');
  
  headings.forEach((heading, index) => {
    // Create list item
    const li = document.createElement('li');
    
    // Add proper indentation for h3
    if (heading.tagName === 'H3') {
      li.classList.add('toc-h3');
    }
    
    // Create link
    const a = document.createElement('a');
    a.textContent = heading.textContent;
    
    // Ensure the heading has an id
    if (!heading.id && heading.textContent) {
      heading.id = heading.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    }
    
    a.href = `#${heading.id}`;
    li.appendChild(a);
    tocList.appendChild(li);
  });
  
  toc.appendChild(tocList);
  
  // Highlight TOC items on scroll
  window.addEventListener('scroll', highlightTocOnScroll);
}

/**
 * Highlight TOC items based on scroll position
 */
function highlightTocOnScroll() {
  const scrollY = window.scrollY;
  const headings = document.querySelectorAll('h2, h3');
  const tocLinks = document.querySelectorAll('.toc a');
  
  headings.forEach((heading, index) => {
    const rect = heading.getBoundingClientRect();
    const offsetTop = rect.top + scrollY;
    const nextHeading = headings[index + 1];
    const nextTop = nextHeading ? nextHeading.getBoundingClientRect().top + scrollY : document.body.scrollHeight;
    
    if (scrollY >= offsetTop - 100 && scrollY < nextTop - 100) {
      // Remove active class from all links
      tocLinks.forEach(link => {
        link.classList.remove('active');
      });
      
      // Add active class to current link
      const currentLink = document.querySelector(`.toc a[href="#${heading.id}"]`);
      if (currentLink) {
        currentLink.classList.add('active');
      }
    }
  });
}

/**
 * Progress bar for reading indication
 */
function setupScrollProgress() {
  const progressContainer = document.querySelector('.progress-container');
  const progressBar = document.querySelector('.progress-bar');
  if (!progressContainer || !progressBar) return;
  
  window.addEventListener('scroll', function() {
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (scrollTop / height) * 100;
    progressBar.style.width = scrolled + '%';
  });
}

/**
 * Scroll-triggered animations
 */
function setupScrollAnimations() {
  // Add sequential fade-in animations to sections
  const sections = document.querySelectorAll('section');
  
  sections.forEach((section, index) => {
    section.style.setProperty('--section-index', index);
  });
  
  // Highlight code blocks when visible in viewport
  const codeBlocks = document.querySelectorAll('pre');
  
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('animate-highlight');
      }
    });
  }, { threshold: 0.3 });
  
  codeBlocks.forEach(block => {
    observer.observe(block);
  });
}

/**
 * Add zoom functionality to images
 */
function setupImageZoom() {
  const figures = document.querySelectorAll('figure');
  
  figures.forEach(figure => {
    const img = figure.querySelector('img');
    if (!img) return;
    
    img.addEventListener('click', function() {
      // Create modal for image
      const modal = document.createElement('div');
      modal.classList.add('image-modal');
      
      // Create zoomed image
      const zoomedImg = document.createElement('img');
      zoomedImg.src = img.src;
      modal.appendChild(zoomedImg);
      
      // Add caption if exists
      const figcaption = figure.querySelector('figcaption');
      if (figcaption) {
        const caption = document.createElement('div');
        caption.classList.add('modal-caption');
        caption.textContent = figcaption.textContent;
        modal.appendChild(caption);
      }
      
      // Add close button
      const closeBtn = document.createElement('button');
      closeBtn.classList.add('modal-close');
      closeBtn.innerHTML = '&times;';
      modal.appendChild(closeBtn);
      
      // Add modal to body
      document.body.appendChild(modal);
      document.body.style.overflow = 'hidden';
      
      // Handle close
      closeBtn.addEventListener('click', closeModal);
      modal.addEventListener('click', function(e) {
        if (e.target === modal) closeModal();
      });
      
      function closeModal() {
        document.body.removeChild(modal);
        document.body.style.overflow = '';
      }
      
      // Close on escape key
      document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closeModal();
      });
    });
    
    // Add indicator that image is clickable
    img.style.cursor = 'zoom-in';
  });
  
  // Add CSS for the modal
  const style = document.createElement('style');
  style.textContent = `
    .image-modal {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0, 0, 0, 0.85);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 9999;
      cursor: zoom-out;
    }
    
    .image-modal img {
      max-width: 85%;
      max-height: 85%;
      object-fit: contain;
      border-radius: 4px;
      box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }
    
    .modal-caption {
      color: white;
      margin-top: 16px;
      font-size: 14px;
      max-width: 80%;
      text-align: center;
    }
    
    .modal-close {
      position: absolute;
      top: 20px;
      right: 20px;
      background: transparent;
      border: none;
      color: white;
      font-size: 32px;
      cursor: pointer;
    }
    
    .animate-highlight {
      animation: codeHighlight 0.5s ease forwards;
    }
    
    @keyframes codeHighlight {
      0% {
        transform: translateY(10px);
        opacity: 0.7;
      }
      100% {
        transform: translateY(0);
        opacity: 1;
      }
    }
  `;
  document.head.appendChild(style);
}

/**
 * Add language detection and proper highlighting to code blocks
 */
function highlightCodeBlocks() {
  const codeBlocks = document.querySelectorAll('pre code');
  
  codeBlocks.forEach(block => {
    // Try to detect language from class
    const parentPre = block.parentElement;
    let language = '';
    
    // Check for language class like 'language-python'
    const classes = block.className.split(' ');
    for (const cls of classes) {
      if (cls.startsWith('language-')) {
        language = cls.substring(9); // remove 'language-' prefix
        break;
      }
    }
    
    // Set data attribute on pre for styling
    if (language) {
      parentPre.setAttribute('data-language', language);
    } else {
      parentPre.setAttribute('data-language', 'code');
    }
  });
} 