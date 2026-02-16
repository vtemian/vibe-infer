# GPU Inference from Scratch — Lesson Plan

Build a forward pass for a tiny MNIST classifier using WebGPU.
No frameworks. Just: load weights -> matmul -> activation -> matmul -> softmax -> prediction.

## Lessons

- [x] **1. WebGPU Bootstrap**
  Get a GPU device from the browser. Prove the GPU is alive.
  - `navigator.gpu.requestAdapter()` — handle to physical GPU
  - `adapter.requestDevice()` — logical connection to command it

- [x] **2. First Compute Shader (add two numbers)**
  Learn the full GPU compute pipeline: buffers, shaders, pipelines, dispatch, readback.
  - Create GPU buffers (input_a, input_b, output)
  - Write a WGSL compute shader that adds two numbers
  - Create a compute pipeline and bind group
  - Dispatch work to the GPU
  - Read results back to JS

- [ ] **3. Matrix Multiplication Kernel**
  The core operation of neural network inference. ~80% of what inference does.
  - Understand why matmul maps to GPU parallelism
  - Write a WGSL shader that multiplies two matrices
  - Learn workgroups and thread indexing (`global_invocation_id`)
  - Test with small hardcoded matrices before scaling up

- [ ] **4. ReLU Activation Kernel**
  Element-wise GPU operation. Dead simple but important.
  - Understand what ReLU does: `max(0, x)`
  - Write a shader that applies ReLU to every element in a buffer
  - Learn about element-wise parallelism on the GPU

- [ ] **5. Softmax Kernel**
  Trickier — requires a reduction across elements. Teaches GPU coordination.
  - Understand softmax: `exp(x_i) / sum(exp(x_j))`
  - Handle numerical stability (subtract max before exp)
  - Implement reduction on the GPU

- [ ] **6. Chain the Forward Pass**
  Wire all kernels together into a real inference pipeline.
  - Layer 1: matmul(input[784], W1[784x128]) + bias1[128] -> ReLU
  - Layer 2: matmul(hidden[128], W2[128x10]) + bias2[10] -> softmax
  - Output: 10 probabilities, argmax = predicted digit

- [ ] **7. Load Real Weights & Test Images**
  The payoff. Load pre-trained weights, classify actual digits.
  - Export weights from a trained PyTorch model (or use pre-exported)
  - Load weight files as ArrayBuffers in JS
  - Upload to GPU buffers
  - Feed test MNIST images through the pipeline
  - See it predict correctly

- [ ] **8. Interactive Demo (bonus)**
  Draw a digit on a canvas, classify it in real-time.
  - Add a drawable canvas to the HTML
  - Capture canvas pixels, resize to 28x28, normalize
  - Run inference on the drawing
  - Display prediction and confidence
