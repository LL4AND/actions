// Get references to the image container and the images
const imageContainer = document.querySelector('.image-container');
const images = document.querySelectorAll('.image');

// Get the previous and next arrow buttons
const prevButton = document.getElementById('prev');
const nextButton = document.getElementById('next');

// Keep track of the current slide index
let currentIndex = 0;

// Function to move the slider to a specific image index
function moveToSlide(index) {
    // Prevent going out of bounds
    if (index < 0) {
        currentIndex = images.length - 1;
    } else if (index >= images.length) {
        currentIndex = 0;
    } else {
        currentIndex = index;
    }

    // Update the image container's transform to show the correct image
    imageContainer.style.transform = `translateX(-${currentIndex * 25}%)`;
}

// Event listener for the previous button
prevButton.addEventListener('click', () => {
    moveToSlide(currentIndex - 1);
});

// Event listener for the next button
nextButton.addEventListener('click', () => {
    moveToSlide(currentIndex + 1);
});

// Auto-slide functionality (auto slide every 3 seconds)
let autoSlideInterval;

function startAutoSlide() {
    // Clear existing interval (to avoid multiple intervals running at once)
    clearInterval(autoSlideInterval);

    // Start a new interval to move to the next slide every 3 seconds
    autoSlideInterval = setInterval(() => {
        moveToSlide(currentIndex + 1);
    }, 3000);
}

// Initialize auto-slide on page load
startAutoSlide();

// Optional: Restart auto-slide when user interacts with the slider
prevButton.addEventListener('click', startAutoSlide);
nextButton.addEventListener('click', startAutoSlide);

// Initialize the slider to show the first image
moveToSlide(currentIndex);


document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
      e.preventDefault();

      document.querySelector(this.getAttribute('href')).scrollIntoView({
          behavior: 'smooth'
      });
  });
});
