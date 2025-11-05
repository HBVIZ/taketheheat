# Liquid Cursor Reveal Effect

A Three.js project that creates a liquid-like reveal effect where moving your cursor reveals the image underneath with a fluid, mask-based animation.

## Features

- Two-image reveal effect with liquid-like masking
- Smooth cursor tracking with interpolation
- Real-time shader-based liquid distortion
- Specular highlights for enhanced liquid appearance
- Responsive design that works on desktop and mobile

## Setup

1. Place your images in the project root directory:
   - `top-image.jpg` - The image that will be revealed (top layer)
   - `bottom-image.jpg` - The image that shows underneath (bottom layer)

2. Open `index.html` in a modern web browser

3. If you don't have images, the project will automatically create colored placeholder textures

## Customization

You can adjust the reveal effect by modifying the shader uniforms in `main.js`:

- `uRevealRadius`: Controls the size of the reveal circle (default: 0.15)
- Mouse interpolation speed: Change the lerp value in the `animate()` function

## Files

- `index.html` - Main HTML file with shader scripts
- `main.js` - Three.js scene setup and animation loop
- `shaders/fragment.glsl` - Fragment shader (also embedded in HTML)
- `shaders/vertex.glsl` - Vertex shader (also embedded in HTML)

## Browser Support

Requires a modern browser with WebGL support. Uses ES6 modules and Three.js.

