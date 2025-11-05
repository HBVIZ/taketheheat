'use strict';

let canvas, gl, ext;
let config = {
  TEXTURE_DOWNSAMPLE: 1,
  DENSITY_DISSIPATION: 0.98,
  VELOCITY_DISSIPATION: 0.99,
  PRESSURE_DISSIPATION: 0.8,
  PRESSURE_ITERATIONS: 25,
  CURL: 28,
  SPLAT_RADIUS: 0.004 };

let pointers = [];
let splatStack = [];
let topImage = null;
let bottomImage = null;
let programs = {};
let textureWidth, textureHeight, density, velocity, divergence, curl, pressure;
let initialized = false;
let lastTime = Date.now();
let blit = null;

// Define GLProgram class first (before any usage)
class GLProgram {
  constructor(vertexShader, fragmentShader) {
    this.uniforms = {};
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw gl.getProgramInfoLog(this.program);
    const uniformCount = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
      const uniformName = gl.getActiveUniform(this.program, i).name;
      this.uniforms[uniformName] = gl.getUniformLocation(this.program, uniformName);
    }
  }
  bind() {
    gl.useProgram(this.program);
  }
}

// Helper functions (defined before init)
function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
    throw gl.getShaderInfoLog(shader);
  return shader;
}

function pointerPrototype() {
  this.id = -1;
  this.x = 0;
  this.y = 0;
  this.dx = 0;
  this.dy = 0;
  this.down = false;
  this.moved = false;
  this.color = [30, 0, 300];
}

pointers.push(new pointerPrototype());

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

function init() {
  canvas = document.getElementById('fluid-canvas');
  if (!canvas) {
    console.error('Canvas element not found!');
    return;
  }
  
  canvas.width = canvas.clientWidth || window.innerWidth;
  canvas.height = canvas.clientHeight || window.innerHeight;

  try {
    const context = getWebGLContext(canvas);
    gl = context.gl;
    ext = context.ext;
    
    if (!gl) {
      console.error('WebGL not supported!');
      return;
    }
    
    // Initialize WebGL resources
    initWebGLResources();
    
    // Setup event listeners
    setupEventListeners();
    
    // Load images
    loadImages();
    
    // Start update loop
    update();
  } catch (error) {
    console.error('Failed to initialize WebGL:', error);
  }
}

function initWebGLResources() {
  // Compile shaders
  const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
    precision highp float;
    precision mediump sampler2D;

    attribute vec2 aPosition;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;

    void main () {
        vUv = aPosition * 0.5 + 0.5;
        vL = vUv - vec2(texelSize.x, 0.0);
        vR = vUv + vec2(texelSize.x, 0.0);
        vT = vUv + vec2(0.0, texelSize.y);
        vB = vUv - vec2(0.0, texelSize.y);
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
  `);

  const clearShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    uniform sampler2D uTexture;
    uniform float value;
    void main () {
        gl_FragColor = value * texture2D(uTexture, vUv);
    }
  `);

  const displayShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    uniform sampler2D uTexture;
    uniform sampler2D uTopTexture;
    uniform sampler2D uBottomTexture;
    void main () {
        vec4 density = texture2D(uTexture, vUv);
        float mask = density.r;
        // Flip texture coordinates vertically (WebGL uses bottom-left origin)
        vec2 flippedUv = vec2(vUv.x, 1.0 - vUv.y);
        vec4 topColor = texture2D(uTopTexture, flippedUv);
        vec4 bottomColor = texture2D(uBottomTexture, flippedUv);
        // Swap: show bottom image initially, reveal top image where there's density
        vec4 finalColor = mix(bottomColor, topColor, smoothstep(0.0, 0.1, mask));
        gl_FragColor = finalColor;
    }
  `);

  const splatShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    uniform sampler2D uTarget;
    uniform float aspectRatio;
    uniform vec3 color;
    uniform vec2 point;
    uniform float radius;
    void main () {
        vec2 p = vUv - point.xy;
        p.x *= aspectRatio;
        vec3 splat = exp(-dot(p, p) / radius) * color;
        vec3 base = texture2D(uTarget, vUv).xyz;
        gl_FragColor = vec4(base + splat, 1.0);
    }
  `);

  const advectionShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform float dt;
    uniform float dissipation;
    void main () {
        vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
        gl_FragColor = dissipation * texture2D(uSource, coord);
        gl_FragColor.a = 1.0;
    }
  `);

  const divergenceShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;
    vec2 sampleVelocity (in vec2 uv) {
        vec2 multiplier = vec2(1.0, 1.0);
        if (uv.x < 0.0) { uv.x = 0.0; multiplier.x = -1.0; }
        if (uv.x > 1.0) { uv.x = 1.0; multiplier.x = -1.0; }
        if (uv.y < 0.0) { uv.y = 0.0; multiplier.y = -1.0; }
        if (uv.y > 1.0) { uv.y = 1.0; multiplier.y = -1.0; }
        return multiplier * texture2D(uVelocity, uv).xy;
    }
    void main () {
        float L = sampleVelocity(vL).x;
        float R = sampleVelocity(vR).x;
        float T = sampleVelocity(vT).y;
        float B = sampleVelocity(vB).y;
        float div = 0.5 * (R - L + T - B);
        gl_FragColor = vec4(div, 0.0, 0.0, 1.0);
    }
  `);

  const curlShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;
    void main () {
        float L = texture2D(uVelocity, vL).y;
        float R = texture2D(uVelocity, vR).y;
        float T = texture2D(uVelocity, vT).x;
        float B = texture2D(uVelocity, vB).x;
        float vorticity = R - L - T + B;
        gl_FragColor = vec4(vorticity, 0.0, 0.0, 1.0);
    }
  `);

  const vorticityShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float curl;
    uniform float dt;
    void main () {
        float T = texture2D(uCurl, vT).x;
        float B = texture2D(uCurl, vB).x;
        float C = texture2D(uCurl, vUv).x;
        vec2 force = vec2(abs(T) - abs(B), 0.0);
        force *= 1.0 / length(force + 0.00001) * curl * C;
        vec2 vel = texture2D(uVelocity, vUv).xy;
        gl_FragColor = vec4(vel + force * dt, 0.0, 1.0);
    }
  `);

  const pressureShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;
    vec2 boundary (in vec2 uv) {
        uv = min(max(uv, 0.0), 1.0);
        return uv;
    }
    void main () {
        float L = texture2D(uPressure, boundary(vL)).x;
        float R = texture2D(uPressure, boundary(vR)).x;
        float T = texture2D(uPressure, boundary(vT)).x;
        float B = texture2D(uPressure, boundary(vB)).x;
        float C = texture2D(uPressure, vUv).x;
        float divergence = texture2D(uDivergence, vUv).x;
        float pressure = (L + R + B + T - divergence) * 0.25;
        gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);
    }
  `);

  const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, `
    precision highp float;
    precision mediump sampler2D;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;
    vec2 boundary (in vec2 uv) {
        uv = min(max(uv, 0.0), 1.0);
        return uv;
    }
    void main () {
        float L = texture2D(uPressure, boundary(vL)).x;
        float R = texture2D(uPressure, boundary(vR)).x;
        float T = texture2D(uPressure, boundary(vT)).x;
        float B = texture2D(uPressure, boundary(vB)).x;
        vec2 velocity = texture2D(uVelocity, vUv).xy;
        velocity.xy -= vec2(R - L, T - B);
        gl_FragColor = vec4(velocity, 0.0, 1.0);
    }
  `);

  // Create programs
  programs.clear = new GLProgram(baseVertexShader, clearShader);
  programs.display = new GLProgram(baseVertexShader, displayShader);
  programs.splat = new GLProgram(baseVertexShader, splatShader);
  programs.advection = new GLProgram(baseVertexShader, advectionShader);
  programs.divergence = new GLProgram(baseVertexShader, divergenceShader);
  programs.curl = new GLProgram(baseVertexShader, curlShader);
  programs.vorticity = new GLProgram(baseVertexShader, vorticityShader);
  programs.pressure = new GLProgram(baseVertexShader, pressureShader);
  programs.gradientSubtract = new GLProgram(baseVertexShader, gradientSubtractShader);

  // Setup blit
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  blit = (destination) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  };

  // Initialize framebuffers
  initFramebuffers();
}

function getWebGLContext(canvas) {
  const params = { alpha: false, depth: false, stencil: false, antialias: false };
  let gl = canvas.getContext('webgl2', params);
  const isWebGL2 = !!gl;
  if (!isWebGL2)
    gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);

  let halfFloat;
  let supportLinearFiltering;
  if (isWebGL2) {
    gl.getExtension('EXT_color_buffer_float');
    supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
  } else {
    halfFloat = gl.getExtension('OES_texture_half_float');
    supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
  }

  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
  let formatRGBA, formatRG, formatR;

  if (isWebGL2) {
    formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
    formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
    formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
  } else {
    formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
  }

  return {
    gl,
    ext: { formatRGBA, formatRG, formatR, halfFloatTexType, supportLinearFiltering }
  };
}

function getSupportedFormat(gl, internalFormat, format, type) {
  if (!supportRenderTextureFormat(gl, internalFormat, format, type)) {
    switch (internalFormat) {
      case gl.R16F:
        return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
      case gl.RG16F:
        return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
      default:
        return null;
    }
  }
  return { internalFormat, format };
}

function supportRenderTextureFormat(gl, internalFormat, format, type) {
  let texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);
  let fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  return status === gl.FRAMEBUFFER_COMPLETE;
}

function initFramebuffers() {
  textureWidth = gl.drawingBufferWidth >> config.TEXTURE_DOWNSAMPLE;
  textureHeight = gl.drawingBufferHeight >> config.TEXTURE_DOWNSAMPLE;
  const texType = ext.halfFloatTexType;
  const rgba = ext.formatRGBA;
  const rg = ext.formatRG;
  const r = ext.formatR;
  density = createDoubleFBO(2, textureWidth, textureHeight, rgba.internalFormat, rgba.format, texType, ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST);
  velocity = createDoubleFBO(0, textureWidth, textureHeight, rg.internalFormat, rg.format, texType, ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST);
  divergence = createFBO(4, textureWidth, textureHeight, r.internalFormat, r.format, texType, gl.NEAREST);
  curl = createFBO(5, textureWidth, textureHeight, r.internalFormat, r.format, texType, gl.NEAREST);
  pressure = createDoubleFBO(6, textureWidth, textureHeight, r.internalFormat, r.format, texType, gl.NEAREST);
}

function createFBO(texId, w, h, internalFormat, format, type, param) {
  gl.activeTexture(gl.TEXTURE0 + texId);
  let texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
  let fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.viewport(0, 0, w, h);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return [texture, fbo, texId];
}

function createDoubleFBO(texId, w, h, internalFormat, format, type, param) {
  let fbo1 = createFBO(texId, w, h, internalFormat, format, type, param);
  let fbo2 = createFBO(texId + 1, w, h, internalFormat, format, type, param);
  return {
    get read() { return fbo1; },
    get write() { return fbo2; },
    swap() { let temp = fbo1; fbo1 = fbo2; fbo2 = temp; }
  };
}

function loadImages() {
  const topImg = new Image();
  const bottomImg = new Image();
  topImg.crossOrigin = 'anonymous';
  bottomImg.crossOrigin = 'anonymous';
  topImg.onload = () => { topImage = topImg; checkImagesReady(); };
  bottomImg.onload = () => { bottomImage = bottomImg; checkImagesReady(); };
  topImg.onerror = () => { console.warn('Failed to load top-image.jpg, using placeholder'); topImage = createPlaceholderImage(true); checkImagesReady(); };
  bottomImg.onerror = () => { console.warn('Failed to load bottom-image.jpg, using placeholder'); bottomImage = createPlaceholderImage(false); checkImagesReady(); };
  topImg.src = 'top-image.jpg';
  bottomImg.src = 'bottom-image.jpg';
}

function createPlaceholderImage(isTop) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 512, 512);
  gradient.addColorStop(0, isTop ? '#ff6b6b' : '#4ecdc4');
  gradient.addColorStop(1, isTop ? '#ee5a6f' : '#44a08d');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 512, 512);
  ctx.fillStyle = 'white';
  ctx.font = '48px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(isTop ? 'TOP IMAGE' : 'BOTTOM IMAGE', 256, 256);
  return canvas;
}

function checkImagesReady() {
  if (topImage && bottomImage && gl) {
    initTextures();
    update();
  }
}

function initTextures() {
  const topTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, topTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, topImage);
  const bottomTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, bottomTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bottomImage);
  window.topTexture = topTexture;
  window.bottomTexture = bottomTexture;
}

function update() {
  if (!gl || !canvas || !blit || !programs.display) {
    requestAnimationFrame(update);
    return;
  }
  if (!topImage || !bottomImage) {
    requestAnimationFrame(update);
    return;
  }
  if (!initialized) {
    initialized = true;
    multipleSplats(parseInt(Math.random() * 20) + 5);
  }
  resizeCanvas();
  const dt = Math.min((Date.now() - lastTime) / 1000, 0.016);
  lastTime = Date.now();
  gl.viewport(0, 0, textureWidth, textureHeight);
  if (splatStack.length > 0) multipleSplats(splatStack.pop());

  programs.advection.bind();
  gl.uniform2f(programs.advection.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
  gl.uniform1i(programs.advection.uniforms.uVelocity, velocity.read[2]);
  gl.uniform1i(programs.advection.uniforms.uSource, velocity.read[2]);
  gl.uniform1f(programs.advection.uniforms.dt, dt);
  gl.uniform1f(programs.advection.uniforms.dissipation, config.VELOCITY_DISSIPATION);
  blit(velocity.write[1]);
  velocity.swap();

  gl.uniform1i(programs.advection.uniforms.uVelocity, velocity.read[2]);
  gl.uniform1i(programs.advection.uniforms.uSource, density.read[2]);
  gl.uniform1f(programs.advection.uniforms.dissipation, config.DENSITY_DISSIPATION);
  blit(density.write[1]);
  density.swap();

  for (let i = 0; i < pointers.length; i++) {
    const pointer = pointers[i];
    if (pointer.moved) {
      splat(pointer.x, pointer.y, pointer.dx, pointer.dy, pointer.color);
      pointer.moved = false;
    }
  }

  programs.curl.bind();
  gl.uniform2f(programs.curl.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
  gl.uniform1i(programs.curl.uniforms.uVelocity, velocity.read[2]);
  blit(curl[1]);

  programs.vorticity.bind();
  gl.uniform2f(programs.vorticity.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
  gl.uniform1i(programs.vorticity.uniforms.uVelocity, velocity.read[2]);
  gl.uniform1i(programs.vorticity.uniforms.uCurl, curl[2]);
  gl.uniform1f(programs.vorticity.uniforms.curl, config.CURL);
  gl.uniform1f(programs.vorticity.uniforms.dt, dt);
  blit(velocity.write[1]);
  velocity.swap();

  programs.divergence.bind();
  gl.uniform2f(programs.divergence.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
  gl.uniform1i(programs.divergence.uniforms.uVelocity, velocity.read[2]);
  blit(divergence[1]);

  programs.clear.bind();
  let pressureTexId = pressure.read[2];
  gl.activeTexture(gl.TEXTURE0 + pressureTexId);
  gl.bindTexture(gl.TEXTURE_2D, pressure.read[0]);
  gl.uniform1i(programs.clear.uniforms.uTexture, pressureTexId);
  gl.uniform1f(programs.clear.uniforms.value, config.PRESSURE_DISSIPATION);
  blit(pressure.write[1]);
  pressure.swap();

  programs.pressure.bind();
  gl.uniform2f(programs.pressure.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
  gl.uniform1i(programs.pressure.uniforms.uDivergence, divergence[2]);
  pressureTexId = pressure.read[2];
  gl.uniform1i(programs.pressure.uniforms.uPressure, pressureTexId);
  gl.activeTexture(gl.TEXTURE0 + pressureTexId);
  for (let i = 0; i < config.PRESSURE_ITERATIONS; i++) {
    gl.bindTexture(gl.TEXTURE_2D, pressure.read[0]);
    blit(pressure.write[1]);
    pressure.swap();
  }

  programs.gradientSubtract.bind();
  gl.uniform2f(programs.gradientSubtract.uniforms.texelSize, 1.0 / textureWidth, 1.0 / textureHeight);
  gl.uniform1i(programs.gradientSubtract.uniforms.uPressure, pressure.read[2]);
  gl.uniform1i(programs.gradientSubtract.uniforms.uVelocity, velocity.read[2]);
  blit(velocity.write[1]);
  velocity.swap();

  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  programs.display.bind();
  gl.uniform1i(programs.display.uniforms.uTexture, density.read[2]);
  gl.uniform1i(programs.display.uniforms.uTopTexture, 7);
  gl.uniform1i(programs.display.uniforms.uBottomTexture, 8);
  gl.activeTexture(gl.TEXTURE7);
  gl.bindTexture(gl.TEXTURE_2D, window.topTexture);
  gl.activeTexture(gl.TEXTURE8);
  gl.bindTexture(gl.TEXTURE_2D, window.bottomTexture);
  blit(null);
  requestAnimationFrame(update);
}

function splat(x, y, dx, dy, color) {
  programs.splat.bind();
  gl.uniform1i(programs.splat.uniforms.uTarget, velocity.read[2]);
  gl.uniform1f(programs.splat.uniforms.aspectRatio, canvas.width / canvas.height);
  gl.uniform2f(programs.splat.uniforms.point, x / canvas.width, 1.0 - y / canvas.height);
  gl.uniform3f(programs.splat.uniforms.color, dx, -dy, 1.0);
  gl.uniform1f(programs.splat.uniforms.radius, config.SPLAT_RADIUS);
  blit(velocity.write[1]);
  velocity.swap();
  gl.uniform1i(programs.splat.uniforms.uTarget, density.read[2]);
  gl.uniform3f(programs.splat.uniforms.color, color[0] * 0.3, color[1] * 0.3, color[2] * 0.3);
  blit(density.write[1]);
  density.swap();
}

function multipleSplats(amount) {
  for (let i = 0; i < amount; i++) {
    const color = [Math.random() * 10, Math.random() * 10, Math.random() * 10];
    const x = canvas.width * Math.random();
    const y = canvas.height * Math.random();
    const dx = 1000 * (Math.random() - 0.5);
    const dy = 1000 * (Math.random() - 0.5);
    splat(x, y, dx, dy, color);
  }
}

function setupEventListeners() {
  // Always create splats on mouse move (no need to click)
  canvas.addEventListener('mousemove', e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Calculate velocity from previous position
    pointers[0].dx = (x - pointers[0].x) * 10.0;
    pointers[0].dy = (y - pointers[0].y) * 10.0;
    pointers[0].x = x;
    pointers[0].y = y;
    
    // Always create splat on hover (don't require mouse down)
    pointers[0].moved = true;
    pointers[0].down = true; // Keep down true for continuous effect
    if (!pointers[0].color || pointers[0].color[0] === 30) {
      pointers[0].color = [Math.random() + 0.2, Math.random() + 0.2, Math.random() + 0.2];
    }
  });

  canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
      if (i >= pointers.length) pointers.push(new pointerPrototype());
      let pointer = pointers[i];
      const x = touches[i].clientX - rect.left;
      const y = touches[i].clientY - rect.top;
      pointer.dx = (x - pointer.x) * 10.0;
      pointer.dy = (y - pointer.y) * 10.0;
      pointer.x = x;
      pointer.y = y;
      pointer.moved = true;
      pointer.down = true;
      if (!pointer.color || pointer.color[0] === 30) {
        pointer.color = [Math.random() + 0.2, Math.random() + 0.2, Math.random() + 0.2];
      }
    }
  }, false);

  canvas.addEventListener('mousedown', () => {
    pointers[0].down = true;
    pointers[0].color = [Math.random() + 0.2, Math.random() + 0.2, Math.random() + 0.2];
  });

  canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const touches = e.targetTouches;
    for (let i = 0; i < touches.length; i++) {
      if (i >= pointers.length) pointers.push(new pointerPrototype());
      pointers[i].id = touches[i].identifier;
      pointers[i].down = true;
      pointers[i].x = touches[i].clientX - rect.left;
      pointers[i].y = touches[i].clientY - rect.top;
      pointers[i].color = [Math.random() + 0.2, Math.random() + 0.2, Math.random() + 0.2];
    }
  });

  // When mouse leaves canvas, stop creating splats
  canvas.addEventListener('mouseleave', () => {
    pointers[0].down = false;
    pointers[0].moved = false;
  });

  window.addEventListener('mouseup', () => {
    // Don't stop on mouseup - keep effect going on hover
    // pointers[0].down = false;
  });

  window.addEventListener('touchend', e => {
    const touches = e.changedTouches;
    for (let i = 0; i < touches.length; i++)
      for (let j = 0; j < pointers.length; j++)
        if (touches[i].identifier == pointers[j].id)
          pointers[j].down = false;
  });
}

function resizeCanvas() {
  if (canvas.width != canvas.clientWidth || canvas.height != canvas.clientHeight) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    initFramebuffers();
    initialized = false;
  }
}
