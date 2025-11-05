uniform sampler2D uTopTexture;
uniform sampler2D uBottomTexture;
uniform vec2 uMouse;
uniform vec2 uResolution;
uniform float uTime;
uniform float uRevealRadius;

varying vec2 vUv;

void main() {
    vec2 uv = vUv;
    
    // Calculate distance from mouse position
    vec2 mousePos = uMouse;
    float dist = distance(uv, mousePos);
    
    // Create smooth reveal mask with liquid-like effect
    // Using smoothstep for smooth edges
    float reveal = smoothstep(uRevealRadius, uRevealRadius * 0.3, dist);
    
    // Add some liquid-like distortion based on distance
    float distortion = sin(dist * 10.0 - uTime * 2.0) * 0.01;
    vec2 distortedUv = uv + vec2(distortion);
    
    // Sample both textures
    vec4 topColor = texture2D(uTopTexture, distortedUv);
    vec4 bottomColor = texture2D(uBottomTexture, distortedUv);
    
    // Mix based on reveal mask
    vec4 finalColor = mix(bottomColor, topColor, reveal);
    
    gl_FragColor = finalColor;
}

